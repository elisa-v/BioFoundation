import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
from safetensors.torch import load_file
import torch_optimizer as torch_optim
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy, Precision, Recall, AUROC,
    AveragePrecision, CohenKappa, F1Score
)
from util.train_utils import RobustQuartileNormalize

class FinetuneTaskTCN(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning TCN models, with support for:

    - Classification types:
        - `bc`: Binary Classification (2 classes)
        - `mcc`: Multi-Class Classification (k-classes k>2)
  
    - Metric logging during training, validation, and testing, including accuracy, precision, recall, F1 score, AUROC, and more
    - Optional input normalization with configurable normalization functions
    - Custom optimizer support including SGD, Adam, AdamW, and LAMB
    - Learning rate schedulers with configurable scheduling strategies
    - Loss: CrossEntropyLoss for both (binary is handled as num_classes=2).
    
    Note: This task is optimized for TCN models and does not use layer-wise learning rate decay.
    """
    def __init__(self, hparams):
        """
        Initialize the FinetuneTaskTCN module.

        Args:
            hparams (DictConfig): Hyperparameters and configuration loaded via Hydra.
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        # enable Tensor Core matmul on compatible GPUs (e.g., RTX 30xx).
        if torch.cuda.is_available(): # 'medium' (good balance between speed and numerical stability)
            torch.set_float32_matmul_precision('medium')

        self.model = hydra.utils.instantiate(self.hparams.model)
        self.num_classes = self.hparams.model.num_classes
        self.classification_type = self.hparams.model.classification_type

        # Input normalization
        if self.hparams.input_normalization is not None and self.hparams.input_normalization.normalize:
            self.normalize = True
            self.normalize_fct = RobustQuartileNormalize(
                self.hparams.input_normalization.quartile_normalization_lower_val,
                self.hparams.input_normalization.quartile_normalization_upper_val
            )

        # Loss function (CE for both bc and mcc)
        if self.classification_type in {"bc", "mcc"}:
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported classification_type: {self.classification_type}")


        self.criterion = nn.CrossEntropyLoss()

        # Classification mode detection
        if not isinstance(self.num_classes, int):
            raise TypeError("Number of classes must be an integer.")
        elif self.num_classes < 2:
            raise ValueError("Number of classes must be at least 2.")
        elif self.num_classes == 2:
            self.classification_task = "binary"
        else:
            self.classification_task = "multiclass"

        # Metrics
        label_metrics = MetricCollection([
            Accuracy(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            Recall(task='multiclass', num_classes=self.num_classes, average="macro"),
            Precision(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            F1Score(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            CohenKappa(task=self.classification_task, num_classes=self.num_classes)
        ])
        logit_metrics = MetricCollection([
            AUROC(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            AveragePrecision(task=self.classification_task, num_classes=self.num_classes, average="macro"),
        ])
        self.train_label_metrics = label_metrics.clone(prefix='train_')
        self.val_label_metrics = label_metrics.clone(prefix='val_')
        self.test_label_metrics = label_metrics.clone(prefix='test_')
        self.train_logit_metrics = logit_metrics.clone(prefix='train_')
        self.val_logit_metrics = logit_metrics.clone(prefix='val_')
        self.test_logit_metrics = logit_metrics.clone(prefix='test_')

    def _step(self, X):
        """
        Perform forward pass and post-process predictions.

        Args:
            X (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
            dict: Dictionary containing predicted labels, probabilities, and logits.
        """
        y_pred_logits = self.model(X)

        if self.classification_type in ("bc", "mcc", "ml"):
            y_pred_probs = torch.softmax(y_pred_logits, dim=1)
            y_pred_label = torch.argmax(y_pred_probs, dim=1)
        else:
            raise ValueError(
                f"Unsupported classification_type: {self.classification_type}. "
                "Expected 'bc' or 'mcc'."
            )

        return {
            'label': y_pred_label,
            'probs': y_pred_probs,
            'logits': y_pred_logits,
        }

    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.squeeze(1) if y.dim() > 1 else y  # Convert from [batch, 1] to [batch] safely
        if self.normalize:
            X = self.normalize_fct(X)

        y_pred = self._step(X)
        loss = self.criterion(y_pred['logits'], y)

        self.train_label_metrics(y_pred['label'], y)
        self.train_logit_metrics(self._handle_binary(y_pred['logits']), y)
        self.log_dict(self.train_label_metrics, on_step=True, on_epoch=False)
        self.log_dict(self.train_logit_metrics, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y = y.squeeze(1) if y.dim() > 1 else y  # Convert from [batch, 1] to [batch] safely
        if self.normalize:
            X = self.normalize_fct(X)

        y_pred = self._step(X)
        loss = self.criterion(y_pred['logits'], y)

        self.val_label_metrics(y_pred['label'], y)
        self.val_logit_metrics(self._handle_binary(y_pred['logits']), y)
        self.log_dict(self.val_label_metrics, on_step=False, on_epoch=True)
        self.log_dict(self.val_logit_metrics, on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y = y.squeeze(1) if y.dim() > 1 else y  # Convert from [batch, 1] to [batch] safely
        if self.normalize:
            X = self.normalize_fct(X)

        y_pred = self._step(X)
        loss = self.criterion(y_pred['logits'], y)

        self.test_label_metrics(y_pred['label'], y)
        self.test_logit_metrics(self._handle_binary(y_pred['logits']), y)
        self.log_dict(self.test_label_metrics, on_step=False, on_epoch=True)
        self.log_dict(self.test_logit_metrics, on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        """
        Custom scheduler step function for step-based LR schedulers
        """
        scheduler.step_update(num_updates=self.global_step)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler (simplified for TCN, no layerwise decay).

        Returns:
            dict: Configuration dictionary with optimizer and LR scheduler.
        """
        base_lr = self.hparams.optimizer.lr

        if self.hparams.optimizer.optim == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=base_lr, momentum=self.hparams.optimizer.get('momentum', 0.9))
        elif self.hparams.optimizer.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, weight_decay=self.hparams.optimizer.get('weight_decay', 0.0))
        elif self.hparams.optimizer.optim == 'AdamW':
            betas = self.hparams.optimizer.get('betas', (0.9, 0.999))
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=self.hparams.optimizer.get('weight_decay', 0.0), betas=betas)
        elif self.hparams.optimizer.optim == 'LAMB':
            optimizer = torch_optim.Lamb(self.model.parameters(), lr=base_lr)
        else:
            raise NotImplementedError("No valid optimizer name")

        scheduler_type = self.hparams.get('scheduler_type', 'cosine')
        if scheduler_type == "multi_step_lr":
            scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        else:
            scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer,
                                                total_training_opt_steps=self.trainer.estimated_stepping_batches)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def _handle_binary(self, preds):
        """
        Special handling for binary classification probabilities.

        Args:
            preds (torch.Tensor): Logit outputs of shape [batch_size, 2].

        Returns:
            torch.Tensor: Probabilities for the positive class of shape [batch_size].
        """
        if self.classification_task == 'binary':
            return preds[:, 1]  # Extract positive class probabilities
        else:
            return preds
