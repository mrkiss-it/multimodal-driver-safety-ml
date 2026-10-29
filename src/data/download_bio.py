"""Module: Bio-signal data ingestion (WESAD) for the time-series subsystem.

Author: Phạm Thành Trung (MSSV: 26410141) - Bio-Signal & Time-Series Specialist
Course: Máy học nâng cao (Advanced Machine Learning) - UIT

WESAD (Schmidt et al., ICMI 2018) is NOT downloaded automatically: the official
archive is several GB and must be fetched manually. This module therefore:
  1. documents where the data must be placed (``data/raw/bio/WESAD/S*/S*.pkl``),
  2. reports which subjects are present/missing,
  3. can generate a SYNTHETIC dataset with the exact WESAD pickle structure so the
     whole team can develop and test the pipeline before the real data is ready.

Synthetic data is always written to a separate directory
(``data/raw/bio/wesad_synthetic``) and every file carries ``synthetic=True`` so it
can never be confused with real recordings.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_BIO_DIR = PROJECT_ROOT / "data" / "raw" / "bio"
WESAD_DIR = RAW_BIO_DIR / "WESAD"
SYNTHETIC_DIR = RAW_BIO_DIR / "wesad_synthetic"

# WESAD has 15 usable subjects: S1 and S12 are not part of the public release.
WESAD_SUBJECTS: List[str] = [f"S{i}" for i in list(range(2, 12)) + list(range(13, 18))]

# Sampling rates (Hz) of the wrist (Empatica E4) signals and the label stream.
BVP_FS = 64
ACC_FS = 32
EDA_FS = 4
TEMP_FS = 4
LABEL_FS = 700

# WESAD label codes -> description.
WESAD_LABELS: Dict[int, str] = {
    0: "not defined / transition",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore",
    6: "ignore",
    7: "ignore",
}

DATASET_METADATA: Dict[str, Any] = {
    "name": "WESAD - Wearable Stress and Affect Detection",
    "reference": "Schmidt, Reiss, Duerichen, Marberger, Van Laerhoven. "
                 "Introducing WESAD, a Multimodal Dataset for Wearable Stress and "
                 "Affect Detection. ACM ICMI 2018, pp. 400-408.",
    "subjects": WESAD_SUBJECTS,
    "wrist_signals": {"BVP": BVP_FS, "ACC": ACC_FS, "EDA": EDA_FS, "TEMP": TEMP_FS},
    "label_rate_hz": LABEL_FS,
    "expected_layout": "data/raw/bio/WESAD/S2/S2.pkl ... S17/S17.pkl",
    "how_to_get": "Search 'WESAD' on the UCI Machine Learning Repository or the "
                  "University of Siegen page, download the archive, extract it so "
                  "that each subject folder sits directly under data/raw/bio/WESAD/.",
}


def get_bio_dirs() -> Dict[str, Path]:
    """Create (if needed) and return the directories used by the bio subsystem."""
    dirs = {"raw_bio": RAW_BIO_DIR, "wesad": WESAD_DIR, "synthetic": SYNTHETIC_DIR}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def check_wesad_status(root: Optional[Path] = None) -> Dict[str, Any]:
    """Report which WESAD subject files exist under ``root``.

    Args:
        root: Directory that should contain ``S*/S*.pkl``. Defaults to the real
            WESAD directory.

    Returns:
        Dict with the found/missing subjects, total size in MB and a readiness flag.
    """
    root = Path(root) if root is not None else WESAD_DIR
    found, missing = [], []
    total_bytes = 0
    for subject in WESAD_SUBJECTS:
        pkl = root / subject / f"{subject}.pkl"
        if pkl.is_file():
            found.append(subject)
            total_bytes += pkl.stat().st_size
        else:
            missing.append(subject)
    return {
        "root": str(root),
        "found_subjects": found,
        "missing_subjects": missing,
        "n_found": len(found),
        "n_expected": len(WESAD_SUBJECTS),
        "size_mb": round(total_bytes / (1024 * 1024), 2),
        "is_complete": len(missing) == 0,
    }


# --------------------------------------------------------------------------- #
# Synthetic WESAD generator
# --------------------------------------------------------------------------- #
# (wesad_label, duration_s, base_hr_bpm). Values are plausible resting/arousal
# levels, NOT measurements. Meditation is lower than baseline, stress is higher.
_PROTOCOL: List[Tuple[int, int, float]] = [
    (1, 180, 70.0),
    (2, 150, 98.0),
    (3, 120, 76.0),
    (4, 150, 62.0),
]
_TRANSITION_S = 5


def _build_timeline(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-second label and target-HR arrays for one synthetic subject."""
    labels: List[int] = []
    targets: List[float] = []
    for label, duration, base in _PROTOCOL:
        labels += [label] * duration
        targets += [base] * duration
        labels += [0] * _TRANSITION_S
        targets += [base] * _TRANSITION_S
    return np.asarray(labels, dtype=np.uint8), np.asarray(targets, dtype=float)


def _synth_subject(subject: str, seed: int) -> Dict[str, Any]:
    """Generate one synthetic subject in the WESAD pickle layout.

    Besides the WESAD keys, a ``meta`` entry stores the ground truth (true HR per
    second and the injected artifact intervals) so unit tests can verify the parser.
    """
    rng = np.random.default_rng(seed)
    labels_1hz, target_hr = _build_timeline(rng)
    n_sec = len(labels_1hz)

    # Smooth ramps between states + subject offset + slow AR(1) drift + RSA.
    kernel = np.ones(10) / 10.0
    smooth = np.convolve(np.pad(target_hr, (5, 5), mode="edge"), kernel, mode="same")[5:-5]
    drift = np.zeros(n_sec)
    for i in range(1, n_sec):
        drift[i] = 0.95 * drift[i - 1] + rng.normal(0.0, 0.6)
    t_sec = np.arange(n_sec) + 0.5
    true_hr = smooth + rng.normal(0.0, 4.0) + drift + 1.5 * np.sin(2 * np.pi * 0.25 * t_sec)
    true_hr = np.clip(true_hr, 45.0, 180.0)

    # BVP: pulse wave whose instantaneous phase follows the HR curve.
    n_bvp = n_sec * BVP_FS
    t_bvp = np.arange(n_bvp) / BVP_FS
    hr_bvp = np.interp(t_bvp, t_sec, true_hr)
    phase = 2 * np.pi * np.cumsum(hr_bvp / 60.0) / BVP_FS
    bvp = np.sin(phase) + 0.30 * np.sin(2 * phase + 0.8)
    bvp = bvp * rng.uniform(0.8, 1.4) + 0.15 * np.sin(2 * np.pi * 0.1 * t_bvp)
    bvp = bvp + rng.normal(0.0, 0.06, n_bvp)

    # ACC: gravity on z + small noise.
    n_acc = n_sec * ACC_FS
    acc = np.zeros((n_acc, 3))
    acc[:, 2] = 64.0
    acc += rng.normal(0.0, 1.5, acc.shape)

    # Inject artifacts in fixed, non-overlapping zones (integer-second boundaries).
    dropout_intervals = [
        (int(rng.integers(40, 200)), 6),
        (int(rng.integers(300, 500)), 6),
    ]
    motion_intervals = [
        (int(rng.integers(220, 270)), 8),
        (int(rng.integers(520, 580)), 8),
    ]
    for start, length in dropout_intervals:
        bvp[start * BVP_FS:(start + length) * BVP_FS] = 0.0
    for start, length in motion_intervals:
        sl_b = slice(start * BVP_FS, (start + length) * BVP_FS)
        tb = t_bvp[sl_b]
        bvp[sl_b] += rng.normal(0.0, 2.0, tb.size) + 1.5 * np.sin(2 * np.pi * 1.2 * tb + rng.uniform(0, 6))
        sl_a = slice(start * ACC_FS, (start + length) * ACC_FS)
        acc[sl_a] += rng.normal(0.0, 25.0, acc[sl_a].shape)

    eda = np.abs(rng.normal(2.0, 0.1, n_sec * EDA_FS))
    temp = rng.normal(33.0, 0.1, n_sec * TEMP_FS)
    label_700 = np.repeat(labels_1hz, LABEL_FS).astype(np.uint8)

    return {
        "subject": subject,
        "synthetic": True,
        "signal": {
            "chest": {},  # chest (RespiBAN) signals are not needed by this project
            "wrist": {
                "BVP": bvp.reshape(-1, 1),
                "ACC": acc,
                "EDA": eda.reshape(-1, 1),
                "TEMP": temp.reshape(-1, 1),
            },
        },
        "label": label_700,
        "meta": {
            "true_hr_1hz": true_hr,
            "dropout_intervals": dropout_intervals,
            "motion_intervals": motion_intervals,
            "note": "SYNTHETIC data for pipeline development only.",
        },
    }


def generate_synthetic_wesad(
    out_dir: Optional[Path] = None,
    subjects: Optional[List[str]] = None,
    seed: int = 2026,
) -> List[Path]:
    """Write synthetic WESAD-format pickles to ``out_dir/S*/S*.pkl``.

    Args:
        out_dir: Destination root. Defaults to ``data/raw/bio/wesad_synthetic``.
        subjects: Subject IDs to create. Defaults to all 15 WESAD subjects.
        seed: Base RNG seed (each subject uses ``seed + index``) for reproducibility.

    Returns:
        List of written pickle paths.
    """
    out_dir = Path(out_dir) if out_dir is not None else SYNTHETIC_DIR
    subjects = subjects if subjects is not None else WESAD_SUBJECTS
    written: List[Path] = []
    for idx, subject in enumerate(subjects):
        data = _synth_subject(subject, seed + idx)
        target = out_dir / subject / f"{subject}.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(data, f)
        written.append(target)
    return written


def print_download_guide() -> None:
    """Print where and how to obtain the real WESAD data."""
    print("=" * 80)
    print("HƯỚNG DẪN LẤY DỮ LIỆU SINH TRẮC HỌC WESAD (BIO-SIGNAL INGESTION)")
    print("Phụ trách: Phạm Thành Trung (Bio-Signal & Time-Series Specialist)")
    print("=" * 80)
    print(f"Bộ dữ liệu : {DATASET_METADATA['name']}")
    print(f"Trích dẫn  : {DATASET_METADATA['reference']}")
    print(f"Đối tượng  : {len(WESAD_SUBJECTS)} người ({WESAD_SUBJECTS[0]}..{WESAD_SUBJECTS[-1]}, không có S1, S12)")
    print(f"Cách lấy   : {DATASET_METADATA['how_to_get']}")
    print(f"Cấu trúc   : {DATASET_METADATA['expected_layout']}")
    print("=" * 80)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="WESAD ingestion helper")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="Generate SYNTHETIC WESAD-format data (for pipeline testing only).")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    get_bio_dirs()
    print_download_guide()
    print("\n[Trạng thái dữ liệu THẬT]:")
    print(json.dumps(check_wesad_status(WESAD_DIR), indent=2, ensure_ascii=False))

    if args.generate_synthetic:
        paths = generate_synthetic_wesad(SYNTHETIC_DIR, seed=args.seed)
        print(f"\n[!] Đã sinh {len(paths)} file DỮ LIỆU TỔNG HỢP (synthetic) tại {SYNTHETIC_DIR}")
        print("    Chỉ dùng để phát triển/kiểm thử pipeline - KHÔNG phải số liệu đo thật.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
