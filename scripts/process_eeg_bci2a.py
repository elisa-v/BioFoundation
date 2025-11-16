# -------------------------------------------------------------
# BCI Competition IV-2a (9 subjects) -> MI trials (L/R/F/T)
# - Loads A0{S}T.mat (training set)
# - Band-pass EEG 0.5–40 Hz
# - Regress out EOG (least squares)
# - Remove artifact-marked trials
# - Slice MI window [t0+3.5s, +2.5s] -> (T,22)
# -------------------------------------------------------------

from pathlib import Path
import numpy as np

from scipy.io import loadmat
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy.io.matlab import mat_struct

import os
from os.path import join as pjoin
import argparse

from typing import Dict, List, Literal, Tuple

import pickle


def load_bci_competition_data(data_dir, subject_number) -> mat_struct:
    """Load BCI Competition IV Dataset 2a and convert to the expected format."""
    mat_fname = pjoin(data_dir, f'A0{subject_number}T.mat')
    mat_contents = sio.loadmat(mat_fname, squeeze_me=True, struct_as_record=False)
    data = mat_contents['data']
    print(f"Loaded data for Subject {subject_number} from {mat_fname}")

    return data

def load_subject_runs(mat_root: str, subject: int, suffix: Literal["T", "E"]) -> list:
    """Load runs for one subject and one split-file type (T or E)."""
    mat_fname = pjoin(mat_root, f"A0{subject}{suffix}.mat")
    if not os.path.exists(mat_fname):
        print(f"  ! Missing file: {mat_fname}")
        return []
    mat = sio.loadmat(mat_fname, squeeze_me=True, struct_as_record=False)
    return list(mat["data"]) 

def parse_mat_name(path: str) -> tuple[int, str]:
    """Return (subject_id, phase) from 'A0{S}{T|E}.mat'."""
    base = os.path.splitext(os.path.basename(path))[0]  # e.g. 'A01T'
    return int(base[1:3]), base[3]                      # (1, 'T') or (1, 'E')

def load_runs_from_mat(mat_path: str) -> list[mat_struct]:
    """Load the 'data' cell array from a single .mat file (T or E)."""
    import scipy.io as sio
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return m["data"]

def bandpass_filter(
    data: np.ndarray,
    fs: float | int,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 5,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="bandpass")
    return filtfilt(b, a, data, axis=0)


def regress_out_eog(
    eeg: np.ndarray,
    eog: np.ndarray,
) -> np.ndarray:
    """ Remove EOG components from EEG via linear regression."""
    oto = eog.T @ eog          # [n_eog, n_eog]
    ots = eog.T @ eeg          # [n_eog, n_eeg]
    b = np.linalg.pinv(oto) @ ots

    eeg_clean = eeg - eog @ b  # [T, n_eeg]
    return eeg_clean


def extract_imagery_segment(
    eeg: np.ndarray,
    start_pos: int,
    fs: int,
    imag_offset_s: float,
    imag_dur_s: float
) -> np.ndarray:
    """Extract the motor imagery segment for a trial."""
    imag_offset = int(imag_offset_s * fs)
    imag_len = int(imag_dur_s * fs)

    start = start_pos + imag_offset
    end = start + imag_len

    if end > eeg.shape[0]:
        return None

    return eeg[start:end, :]

def summarize_subject_data(subject_data) -> None:
    """Print a compact summary of #trials per class and total."""
    counts = {k: len(v) for k, v in subject_data.items()}
    total = sum(counts.values())
    print("\nSUMMARY (kept trials)")
    print(f"L (left):   {counts.get('L',0)}")
    print(f"R (right):  {counts.get('R',0)}")
    print(f"F (feet):   {counts.get('F',0)}")
    print(f"T (tongue): {counts.get('T',0)}")
    print(f"TOTAL:      {total}\n")


def process_bci_data(
    data: List[mat_struct],
    imag_offset_s: float = 3.5, # without visual cue overlapping
    imag_dur_s: float = 2.5
) -> Dict[str, List[np.ndarray]]:
    """Process BCI Competition IV 2a matlab data into MI trials.
    Steps:
      1. Split EEG (22 ch) and EOG (3 ch)
      2. Band-pass filter 0.5-40 Hz (EEG)
      3. Regress out EOG from EEG
      4. Extract MI windows 
      5. Discard artifact trials based on visual inspection."""
    
    subject_data: Dict[str, List[np.ndarray]] = {
        "L": [],
        "R": [],
        "F": [],
        "T": [],
    }
    
    for run_idx, session in enumerate(data):

        if not hasattr(session, "X"): 
            continue

        # print(session._fieldnames)
        X = session.X  
        fs = int(session.fs)  
        y = session.y  
        trial_pos = session.trial  
        artifacts = session.artifacts 

        if y.size == 0 or trial_pos.size == 0:
            print(f"  Run {run_idx}: no trials (calibration), skipping.")
            continue

        print(f"  Run {run_idx}: X={X.shape}, trials={y.shape[0]}")
        
        # 1. Split EEG and EOG
        eeg = X[:, :22] # 22-channels are EEG, notch filter at 50Hz and bd 0.5-100Hz already applied
        eog = X[:, 22:] # 3-channels are EOG, notch filter at 50Hz and bd 0.5-100Hz already applied

        # 2. Band-pass filter EEG 
        eeg_bp = bandpass_filter(eeg, fs, low=0.5, high=40.0, order=5)

        # 3. Regress out EOG from EEG
        eeg_clean = regress_out_eog(eeg_bp, eog)

        imag_len_samples = int(imag_dur_s * fs)
        imag_offset_samples = int(imag_offset_s * fs)

        for t_idx, (start_pos, label) in enumerate(zip(trial_pos, y)):
            # 4. Extract motor imagery segment
            start_pos = int(start_pos)
            start = start_pos + imag_offset_samples
            end = start + imag_len_samples
            if end > eeg_clean.shape[0]:
                continue

            mi_seg = eeg_clean[start:end, :]  

            # 4. Discard artifact trials
            has_artifact = (artifacts.size == y.size and artifacts[t_idx] != 0)

            if has_artifact: 
                continue

            if label == 1:
                subject_data["L"].append(mi_seg)
            elif label == 2:
                subject_data["R"].append(mi_seg)
            elif label == 3:
                subject_data["F"].append(mi_seg)
            elif label == 4:
                subject_data["T"].append(mi_seg)

    return subject_data

def trials_to_samples(subject_data: Dict[str, List[np.ndarray]],
                       task: str = "LR") -> Tuple[List[np.ndarray], List[int]]:
    """Convert the dict of class-> trials into flat (X,y) lists. 
    X is returned as [C, T] to match the saving convention."""
    X_list, y_list = [], []

    if task == "LR":
        # Left=0, Right=1
        for seg in subject_data.get("L", []):
            X_list.append(seg.T)        # [C, T]
            y_list.append(0)
        for seg in subject_data.get("R", []):
            X_list.append(seg.T)
            y_list.append(1)
    elif task == "4C":
        # 4-class (L=0,R=1,F=2,T=3) if you want later
        for k, lab in zip(["L","R","F","T"], [0,1,2,3]):
            for seg in subject_data.get(k, []):
                X_list.append(seg.T)
                y_list.append(lab)
    else:
        raise ValueError(f"Unknown task '{task}'")

    return X_list, y_list

def export_to_pickles(X_list: List[np.ndarray],
                      y_list: List[int],
                      out_dir: str | Path,
                      split: str,
                      prefix: str) -> None:
    """Save each (X,y) pair as one pickle:
      out_dir/data/processed/{split}/<prefix>_<idx>.pkl"""
    base = Path(out_dir) / split
    base.mkdir(parents=True, exist_ok=True)

    for i, (X, y) in enumerate(zip(X_list, y_list)):
        sample = {"X": X.astype(np.float32), "y": int(y)}
        p = base / f"{prefix}_{i:05d}.pkl"
        with open(p, "wb") as f:
            pickle.dump(sample, f)

def list_session_fields(session: mat_struct) -> List[str]:
    """Quick inspector for a single run struct."""
    return list(getattr(session, "_fieldnames", []))

def split_subjects_leave_one_out(all_subjects: List[int], held_out: int
) -> Tuple[List[int], List[int]]:
    trval = [s for s in all_subjects if s != held_out]
    te = [held_out]
    return trval, te

def split_train_val_by_subjects(subjects: List[int], val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split a list of subjects into training and validation sets."""
    rng = np.random.default_rng(seed)
    subs = subjects.copy()
    rng.shuffle(subs)
    k = int(round(len(subs) * (1 - val_ratio)))
    return subs[:k], subs[k:]

def index_bci2a_files(mat_root: str, subjects: List[int]) -> Dict[int, Dict[str, List[str]]]:
    """Create an index of available .mat files for each subject."""
    idx = {}
    for s in subjects:
        idx[s] = {
            "trainT": [os.path.join(mat_root, f"A0{s}T.mat")],  # training runs (with labels)
            "testE":  [os.path.join(mat_root, f"A0{s}E.mat")],  # test runs (no labels in original comp)
        }
    return idx

def split_mode_leave_one_subject_out(
    mat_root: str,
    subjects: List[int],
    held_out_subject: int,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Split data in leave-one-subject-out mode."""
    file_idx = index_bci2a_files(mat_root, subjects)

    trainval_subjects, test_subjects = split_subjects_leave_one_out(subjects, held_out_subject)
    trval_paths = [file_idx[s]["trainT"][0] for s in trainval_subjects] + [file_idx[s]["testE"][0] for s in trainval_subjects]
    test_paths  = [file_idx[s]["trainT"][0] for s in test_subjects] + [file_idx[s]["testE"][0] for s in test_subjects]

    np.random.seed(seed)
    np.random.shuffle(trval_paths)
    train_files, val_files = np.split(np.array(trval_paths), [int(len(trval_paths) * (1-val_ratio))])
    train_paths = train_files.tolist()
    val_paths   = val_files.tolist()

    # print(f"Training paths ({len(train_paths)}): {train_paths}")
    # print(f"Validation paths ({len(val_paths)}): {val_paths}")
    # print(f"Test paths ({len(test_paths)}): {test_paths}")

    return train_paths, val_paths, test_paths

def split_mode_T_as_train_val_E_as_test(
    mat_root: str,
    subjects: List[int],
    val_ratio: float = 0.2,
    seed: int = 123
) -> Tuple[List[str], List[str], List[str]]:
    """Split data in T-as-train/val and E-as-test mode following original split"""
    file_idx = index_bci2a_files(mat_root, subjects)
    # Subject-level val split
    tr_subs, val_subs = split_train_val_by_subjects(subjects, val_ratio, seed)
    train_paths = [file_idx[s]["trainT"][0] for s in tr_subs]
    val_paths   = [file_idx[s]["trainT"][0] for s in val_subs]
    test_paths  = [file_idx[s]["testE"][0]  for s in subjects]

    # print(f"Training paths ({len(train_paths)}): {train_paths}")
    # print(f"Validation paths ({len(val_paths)}): {val_paths}")
    # print(f"Test paths ({len(test_paths)}): {test_paths}")

    return train_paths, val_paths, test_paths


def split_and_process_data(
    mat_root: str,
    out_root: str,
    subjects: List[int] | None = None,
    task: str = "LR", 
    split_mode: str = "loo", # "loo" (leave-one-out) or "TvsE" (subjects both in train/val and test)
    held_out_subject: int = 9,  # used only for loo
    val_ratio: float = 0.2,
    seed: int = 123
) -> None:
    """Split and process BCI Competition IV-2a data into pickles."""
    if subjects is None:
        subjects = list(range(1, 10))

    if split_mode == "loo":
        train_paths, val_paths, test_paths = split_mode_leave_one_subject_out(
            mat_root, subjects, held_out_subject, val_ratio, seed
        )
    elif split_mode == "TvsE":
        train_paths, val_paths, test_paths = split_mode_T_as_train_val_E_as_test(
            mat_root, subjects, val_ratio, seed
        )
    else:
        raise ValueError("split_mode must be 'loo' or 'TvsE'")

    def process_list(paths, split_name: str, seed: int = 123) -> None:
        rng = np.random.default_rng(seed)

        for p in paths:
            subj, phase = parse_mat_name(p)            
            runs = load_runs_from_mat(p)              
            print(f"Processing Subject {subj} Phase {phase} ({p}), #runs={len(runs)}")
            subject_data = process_bci_data(
                runs, imag_offset_s=3.5, imag_dur_s=2.5
            )
            X, y = trials_to_samples(subject_data, task=task)

            # Reproducible shuffle while preserving [C, T] per sample
            idx = rng.permutation(len(y))
            X_shuf = [X[i] for i in idx]
            y_shuf = [int(y[i]) for i in idx]

            subj_prefix = f"S{subj:02d}_{phase}_{task}"
            export_to_pickles(X_shuf, y_shuf, out_root, split_name, subj_prefix)

    process_list(train_paths, "train")
    process_list(val_paths,   "val")
    process_list(test_paths,  "test")


def _parse_subjects(s: str | None) -> list[int]:
    """Parse a comma-separated string of subject IDs into a list of integers."""
    if not s:
        return list(range(1, 10))  # default: 1..9
    return [int(x) for x in s.split(",")]

def main():
    p = argparse.ArgumentParser(description="BCI IV-2a → MI trial pickles")
    p.add_argument("--mat-root", required=True, help="Folder containing A0{S}{T|E}.mat files")
    p.add_argument("--out-root", required=True, help="Output folder for processed pickles")
    p.add_argument("--subjects", default=None,
                   help="Comma-separated subjects (e.g. '1,2,3'); default=1..9")
    p.add_argument("--task", default="LR", choices=["LR", "4C"], help="Classification task")
    p.add_argument("--split-mode", default="loo", choices=["loo", "TvsE"],
                   help="CV mode: leave-one-out or Train/Val on T, Test on E")
    p.add_argument("--held-out-subject", type=int, default=9,
                   help="Held-out subject id for LOO")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation fraction")
    p.add_argument("--seed", type=int, default=123, help="RNG seed")
    p.add_argument("--imag-offset", type=float, default=3.5, help="MI window start (s)")
    p.add_argument("--imag-dur", type=float, default=2.5, help="MI window length (s)")
    args = p.parse_args()

    subjects = _parse_subjects(args.subjects)

    split_and_process_data(
        mat_root=str(Path(args.mat_root)),
        out_root=str(Path(args.out_root)),
        subjects=subjects,
        task=args.task,
        split_mode=args.split_mode,
        held_out_subject=args.held_out_subject,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("Processing complete.")

if __name__ == "__main__":
    main()


    
