Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Scripts

## Process EEG BCI Motor Imagery IV Competition (2a)

**process_eeg_bci2a.py** loads **BCI Competition IV-2a** `.mat` files (`A0{S}{T|E}.mat`), applies basic preprocessing, extracts motor-imagery (MI) windows, and saves per-trial pickles for ML.

---

### Features
- EEG band-pass **0.5–40 Hz** (Butterworth, zero-phase)
- **EOG regression** (least-squares) to remove ocular artifacts
- MI window extraction (default **start=3.5 s**, **dur=2.5 s**)
- Tasks: **LR** (Left vs Right) or **4C** (L/R/F/T)
- Split modes:
  - `loo`: **Leave-One-Out** by subject (held-out subject = test)
  - `TvsE`: Train/Val from **T** files, Test from **E** files (original split)

---

### Download the data
1. Request/download **BCI Competition IV-2a** from the BNCI repository:  
   https://bnci-horizon-2020.eu/database/data-sets
2. Place the `.mat` files as:
```bash
<RAW_ROOT>/
A01T.mat A01E.mat
A02T.mat A02E.mat
...
A09T.mat A09E.mat
```

### Arguments
- `--mat-root` (required): Path to the folder containing the raw .mat files (e.g., .../data/raw with A01T.mat, A01E.mat, …).
- `--out-root` (required): Output directory for processed pickles. The script creates train/, val/, test/ inside this path.
- `--subjects` (optional, default: all 1..9). Comma-separated list of subjects to include, e.g. "1,2,5,7".
- `--task` (optional, default: LR, choices: LR, 4C). **LR**: binary Left vs Right, **4C**: 4-class (Left, Right, Feet, Tongue)
- `--split-mode` (optional, default: loo, choices: loo, TvsE). **loo**: leave-one-subject-out (train/val from all except - `--held-out-subject`, test on held-out subject). **TvsE**: use *T.mat for train/val and *E.mat for test (per subject)
- `--held-out-subject` (optional, default: 9). Subject used as test set when --split-mode loo is selected.
- `--val-ratio` (optional, default: 0.2). Validation fraction within the training pool (subject-balanced stratified split at sample level).
- `--seed` (optional, default: 123). RNG seed for reproducible splits/shuffling.
- `--imag-offset` (optional, default: 3.5). Motor-imagery window start (seconds after trial onset).
- `--imag-dur` (optional, default: 2.5). Motor-imagery window length in seconds.

The script writes .pkl files to <OUT_ROOT>:
```bash
<OUT_ROOT>/
  train/  *.pkl
  val/    *.pkl
  test/   *.pkl
```
Each pickle contains:
- X: np.ndarray of shape [T, C] = [time_samples, 22 EEG] (after band-pass + EOG regression)
- y: int label (LR: 0=Left, 1=Right; 4C: 1=Left, 2=Right, 3=Feet, 4=Tongue)

### Quick run
Process all subjects with defaults (LOO with S9 as test; LR task; 3.5–6.0 s window):
- Modify "RAW_ROOT" with the path to your raw data
- Modify "OUT_ROOT" with the path to the output folder (it should end as "\processed")

```bash
python python scripts/process_eeg_bci2a.py \
--mat-root <RAW_ROOT> \
--out-root <OUT_ROOT>
```

### 1) Leave-One-Out (S9 as test, 15% val from remaining subjects)
```bash
python scripts/process_eeg_bci2a.py \
  --mat-root <RAW_ROOT> \
  --out-root <OUT_ROOT> \
  --split-mode loo \
  --held-out-subject 9 \
  --val-ratio 0.15 \
  --task LR \
  --seed 123
```

### 2) Train/Val from T, Test from E (all subjects), 4-class
```bash
python scripts/process_eeg_bci2a.py \
  --mat-root <RAW_ROOT> \
  --out-root <OUT_ROOT> \
  --split-mode TvsE \
  --task 4C \
  --seed 123
```

### 3) Subset of subjects (1,2,3,4)
```bash
python scripts/process_eeg_bci2a.py \
  --mat-root <RAW_ROOT> \
  --out-root <OUT_ROOT> \
  --split-mode TvsE \
  --subjects "1,2,3,4" \
  --task LR
```
