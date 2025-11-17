
# BCI2A Motor-Imagery: Pipeline + TCN Baseline

This contribution adds an end-to-end pipeline for **BCI Competition IV – Dataset 2a** (22 EEG + 3 EOG) with:
- preprocessing (0.5–40 Hz band-pass, EOG regression, artifact rejection by flags),
- flexible train/val/test splits,
- a baseline **Temporal Convolutional Network (TCN)** and Hydra configs.

---

## What’s included
- `scripts/process_eeg_bci2a.py`: load `.mat`, filter, EOG-regress, cut MI windows, export **pickles**.
- `make_datasets/make_hdf5.py`: existing functions to bundle pickles → **HDF5** (`train/val/test.h5`) 
- `models/tcn_baseline.py`: simple, clean TCN classifier (`(B, 22, T) → (B, num_classes)`), where B is the batch size and T the temporal dimension.
- `config/data_module/finetune_data_module_bci2a.yaml`: HDF5 data module (no squeeze, finetune mode).
- `config/experiment/BCI2A_TCN_LR.yaml` and `BCI2A_TCN_4C.yaml`: 2 different experiments for **binary classification** (left and right hand imaginary movements) or **4-class classification** (left and right hand + feet and tongue imaginary movements).

### Data layout (recommended)
```bash
BIOFOUNDATION/
└── data/                          
    ├── raw/
    │   └── A01T.mat     # A01T.mat ... A09E.mat
    └── BCI2A_data/
        ├── processed/
        │   ├── train/
        │   │   └── example.pkl # each: {'X': [22, T], 'y': int}
        │   ├── val/
        │   │   └── example.pkl 
        │   └── test/.pkl
        │       └── example.pkl 
        ├── train.h5
        ├── val.h5
        └── test.h5

```
**Labels**
- **LR** task: `0 = Left`, `1 = Right`
- **4C** task: `0 = Left`, `1 = Right`, `2 = Feet`, `3 = Tongue`

**Signal shape**: saved per sample as **[22, T]** (channels first; T≈625 for 2.5 s at 250 Hz).  
**MI window**: **3.5–6.0 s** post-cue (2.5 s).

---

## Model & Training Design Choices

### Architecture (TCN Baseline)
- **Input shape:** `(B, 22, T)`; 
- **Backbone:** Temporal Convolutional Network (TCN)
  - **Causal 1D convs** with **dilations** `1, 2, 4, 8` (default `num_levels=4`) to cover short–mid temporal context.
  - **Kernel size:** `3` (configurable).
  - **Residual blocks** with BN + ReLU + Dropout (default `p=0.1`).
  - **Causal padding + Chomp** to preserve sequence length without leakage from the future.
  - **Global Avg Pool** over time → fixed-length embedding.
- **Head:** single `Linear(hidden_channels → num_classes)` for logits.
- **Why TCN?** Efficient 1D convolutions, stable training, parallelizable (vs. RNNs), and strong for MI windows where temporal locality matters.

### Signal Preprocessing 
- Band-pass **0.5–40 Hz** on EEG, 22 channels only for classification.
- **EOG regression** (least squares) from 3 EOG channels:  
  \( S = Y - U(U^\top U)^{-1}U^\top Y \)
- **MI window:** 3.5–6.0 s (2.5s duration) post-cue.
- **Saved sample:** `X ∈ ℝ^{22×T}`, `y ∈ {0,1}` (LR) or `{0,1,2,3}` (4C).

### Loss / Optimizer / Scheduler
- **Task:** classification (binary or 4-class) → `CrossEntropyLoss`.
- **Optimizer:** `AdamW` (default `lr=1e-3`, `weight_decay=1e-4`).
- **Scheduler:** cosine (warmup optional via config).
- **Batch size / Epochs:** set via Hydra experiment config (e.g., `batch_size=64`, `max_epochs=100`).

### Training Framework
- **PyTorch** + **PyTorch Lightning** for training loops & checkpointing.
- **Hydra** for reproducible, composable configs.
- **Single-GPU** training supported out-of-the-box.


---

## Quick Start

1) **Download data**. Get BCI IV-2a from the BNCI Horizon site [text](https://bnci-horizon-2020.eu/database/data-sets) and place `.mat` files under:
```bash
<ABS_PATH>/data/BCI2A_data/raw
```

2) **Preprocess raw data**. Choose a split:

- **Leave-one-subject-out (LOO)**, test S9:
```bash
python scripts/process_eeg_bci2a.py \
  --mat-root "<ABS_PATH>/data/BCI2A_data/raw" \
  --out-root "<ABS_PATH>/data/BCI2A_data/processed" \
  --split-mode loo --held-out-subject 9 --task LR 
```

- **Original Splitting: Train/Val from T, Test from E:**:
```bash
python scripts/process_eeg_bci2a.py \
  --mat-root "<ABS_PATH>/data/BCI2A_data/raw" \
  --out-root "<ABS_PATH>/data/BCI2A_data/processed" \
  --split-mode TvsE --task LR --seed 123
```
Optional subset: --subjects "1,2,5,7"

3) **Convert pickles to HDF5**:
```bash
python make_datasets/make_hdf5.py --prepath <OUT_ROOT> --dataset BCI2A
```

4) **Train TCN model**

Change DATA_PATH and CHECKPOINT_DIR in **run_train.py**

Run:
```bash
python run_train.py +experiment=BCI2A_TCN_LR
# for 4-class: +experiment=BCI2A_TCN_4C
```

TensorBoard + checkpoints are stored under:
```bash
CHECKPOINT_DIR/<tag>/<timestamp>/
```

