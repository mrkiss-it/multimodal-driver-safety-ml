"""Module: WESAD parser - wrist BVP -> 1 Hz heart-rate series aligned with labels.

Author: Phạm Thành Trung (MSSV: 26410141) - Bio-Signal & Time-Series Specialist
Course: Máy học nâng cao (Advanced Machine Learning) - UIT

Pipeline per subject:
  1. load ``S*.pkl`` (WESAD layout; BVP 64 Hz, ACC 32 Hz, labels 700 Hz),
  2. band-pass filter BVP (Butterworth 0.5-4 Hz) and detect pulse peaks,
  3. convert inter-beat intervals (IBI) to heart rate and resample to 1 Hz,
  4. down-sample the 700 Hz labels to 1 Hz (a second with mixed labels = transition),
  5. flag signal dropouts, motion artifacts and implausible HR per second.

Project-label mapping (an explicit PROXY assumption, see ``map_to_project_label``):
WESAD records stress/amusement/meditation, not driving states. We use baseline and
amusement as "Normal", meditation (low HR) as a proxy for "Drowsy", and stress
(high HR) as a proxy for the "panic" pattern of the Assault class. This is a
modelling assumption that must be stated in the report, not a ground-truth equivalence.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_BIO_DIR = PROJECT_ROOT / "data" / "raw" / "bio"
REPORTS_DIR = PROJECT_ROOT / "reports"

BVP_FS = 64
ACC_FS = 32
LABEL_FS = 700

HR_MIN_BPM = 40.0
HR_MAX_BPM = 200.0
DROPOUT_STD_THRESHOLD = 1e-3   # raw-BVP std per second below this => flat/dropout
MOTION_STD_THRESHOLD = 10.0    # std of |ACC| per second (E4 units, 64 = 1 g)
IBI_COVERAGE_S = 2.5           # a second needs an IBI within this distance
ARTIFACT_MARGIN_S = 1          # seconds around an artifact also treated as suspect

WESAD_LABEL_NAMES: Dict[int, str] = {
    0: "transition", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation",
}


def load_wesad_pickle(path: Path) -> Dict[str, Any]:
    """Load one WESAD subject pickle (Python-2 pickles need ``latin1`` encoding)."""
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def bandpass(x: np.ndarray, fs: float, low: float = 0.5, high: float = 4.0,
             order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter.

    Args:
        x: 1-D signal.
        fs: Sampling rate in Hz.
        low: Lower cut-off in Hz.
        high: Upper cut-off in Hz.
        order: Filter order.

    Returns:
        Filtered signal with the same length. Signals too short for ``filtfilt``
        are returned mean-removed instead of raising.
    """
    x = np.asarray(x, dtype=float).ravel()
    b, a = butter(order, [low, high], btype="band", fs=fs)
    if x.size <= 3 * max(len(a), len(b)):
        return x - np.mean(x) if x.size else x
    return filtfilt(b, a, x)


def detect_pulse_peaks(bvp_filt: np.ndarray, fs: float) -> np.ndarray:
    """Return sample indices of pulse peaks in a band-passed BVP signal."""
    if bvp_filt.size == 0:
        return np.array([], dtype=int)
    spread = np.percentile(bvp_filt, 95) - np.percentile(bvp_filt, 5)
    if spread <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(bvp_filt, distance=int(0.3 * fs), prominence=0.3 * spread)
    return peaks


def peaks_to_hr_1hz(peaks: np.ndarray, fs: float, n_sec: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pulse peaks to a 1 Hz HR curve.

    IBIs outside the plausible range are discarded; HR is interpolated to the
    centre of every second, but only where an IBI exists within ``IBI_COVERAGE_S``
    (so gaps are NaN instead of being filled with invented values).

    Returns:
        (hr_bpm, covered): arrays of length ``n_sec``.
    """
    hr = np.full(n_sec, np.nan)
    covered = np.zeros(n_sec, dtype=bool)
    if len(peaks) < 3:
        return hr, covered
    t_peaks = peaks / fs
    ibi = np.diff(t_peaks)
    t_mid = (t_peaks[:-1] + t_peaks[1:]) / 2.0
    ok = (ibi >= 60.0 / HR_MAX_BPM) & (ibi <= 60.0 / HR_MIN_BPM)
    if ok.sum() < 2:
        return hr, covered
    t_mid, hr_inst = t_mid[ok], 60.0 / ibi[ok]
    t_grid = np.arange(n_sec) + 0.5
    hr = np.interp(t_grid, t_mid, hr_inst)
    nearest = np.min(np.abs(t_grid[:, None] - t_mid[None, :]), axis=1)
    covered = nearest <= IBI_COVERAGE_S
    hr[~covered] = np.nan
    return hr, covered


def labels_to_1hz(label_700: np.ndarray, n_sec: int, fs: int = LABEL_FS) -> np.ndarray:
    """Down-sample labels to 1 Hz. A second whose samples are not all equal -> 0."""
    label_700 = np.asarray(label_700).ravel()
    out = np.zeros(n_sec, dtype=np.int16)
    for s in range(n_sec):
        seg = label_700[s * fs:(s + 1) * fs]
        if seg.size == fs and np.all(seg == seg[0]):
            out[s] = int(seg[0])
    return out


def per_second_std(x: np.ndarray, fs: int, n_sec: int) -> np.ndarray:
    """Standard deviation of ``x`` inside each 1-second window."""
    x = np.asarray(x, dtype=float).ravel()
    out = np.zeros(n_sec)
    for s in range(n_sec):
        seg = x[s * fs:(s + 1) * fs]
        out[s] = np.std(seg) if seg.size else 0.0
    return out


def _dilate(mask: np.ndarray, margin: int) -> np.ndarray:
    """Extend True regions by ``margin`` seconds on both sides."""
    if margin <= 0 or not mask.any():
        return mask.copy()
    out = mask.copy()
    for shift in range(1, margin + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def map_to_project_label(wesad_label: int) -> int:
    """Map a WESAD label to the project classes (PROXY mapping, see module doc).

    Returns:
        0 = Normal (baseline, amusement), 1 = Drowsy-proxy (meditation),
        2 = Assault/panic-proxy (stress), -1 = excluded (transition/ignored).
    """
    return {1: 0, 3: 0, 4: 1, 2: 2}.get(int(wesad_label), -1)


def parse_subject(pkl_path: Path) -> pd.DataFrame:
    """Parse one subject into a 1 Hz table.

    Columns: subject, t_sec, hr_bpm, label_wesad, label_name, project_label,
    is_dropout, is_motion, is_valid.
    """
    data = load_wesad_pickle(pkl_path)
    wrist = data["signal"]["wrist"]
    bvp = np.asarray(wrist["BVP"], dtype=float).ravel()
    acc = np.asarray(wrist["ACC"], dtype=float)
    n_sec = min(len(bvp) // BVP_FS, len(acc) // ACC_FS, len(data["label"]) // LABEL_FS)

    bvp = bvp[:n_sec * BVP_FS]
    acc_mag = np.linalg.norm(acc[:n_sec * ACC_FS], axis=1)

    is_dropout = per_second_std(bvp, BVP_FS, n_sec) < DROPOUT_STD_THRESHOLD
    is_motion = per_second_std(acc_mag, ACC_FS, n_sec) > MOTION_STD_THRESHOLD

    peaks = detect_pulse_peaks(bandpass(bvp, BVP_FS), BVP_FS)
    hr, covered = peaks_to_hr_1hz(peaks, BVP_FS, n_sec)

    labels = labels_to_1hz(data["label"], n_sec)
    in_range = (hr >= HR_MIN_BPM) & (hr <= HR_MAX_BPM)
    suspect = _dilate(is_dropout | is_motion, ARTIFACT_MARGIN_S)
    is_valid = covered & in_range & ~suspect

    return pd.DataFrame({
        "subject": data.get("subject", pkl_path.stem),
        "t_sec": np.arange(n_sec),
        "hr_bpm": np.round(hr, 2),
        "label_wesad": labels,
        "label_name": [WESAD_LABEL_NAMES.get(int(v), "ignore") for v in labels],
        "project_label": [map_to_project_label(v) for v in labels],
        "is_dropout": is_dropout,
        "is_motion": is_motion,
        "is_valid": is_valid,
    })


def find_subject_pickles(root: Path) -> List[Path]:
    """Return ``root/S*/S*.pkl`` files sorted by subject number."""
    files = [p for p in Path(root).glob("S*/S*.pkl") if p.is_file()]
    return sorted(files, key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0)


def summarize_subject(df: pd.DataFrame) -> Dict[str, Any]:
    """Quality summary of one parsed subject."""
    n = len(df)
    valid = df[df["is_valid"]]
    return {
        "subject": str(df["subject"].iloc[0]) if n else "",
        "n_seconds": int(n),
        "valid_seconds": int(len(valid)),
        "valid_rate": round(len(valid) / n, 4) if n else 0.0,
        "dropout_seconds": int(df["is_dropout"].sum()),
        "motion_seconds": int(df["is_motion"].sum()),
        "hr_mean_valid": round(float(valid["hr_bpm"].mean()), 2) if len(valid) else None,
        "hr_by_label": {
            WESAD_LABEL_NAMES[k]: round(float(g["hr_bpm"].mean()), 2)
            for k, g in valid.groupby("label_wesad") if k in WESAD_LABEL_NAMES and k != 0
        },
    }


def parse_directory(root: Path, out_dir: Path, source: str,
                    report_path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse all subjects under ``root``; write one CSV per subject and a JSON report.

    Args:
        root: Directory containing ``S*/S*.pkl``.
        out_dir: Where the per-subject CSV files are written.
        source: ``"real"`` or ``"synthetic"`` - recorded in the report so the origin
            of the numbers is never ambiguous.
        report_path: Optional JSON report path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for pkl in find_subject_pickles(root):
        df = parse_subject(pkl)
        df.to_csv(out_dir / f"{pkl.stem}.csv", index=False)
        summaries.append(summarize_subject(df))

    total = sum(s["n_seconds"] for s in summaries)
    valid = sum(s["valid_seconds"] for s in summaries)
    report = {
        "parsed_by": "Phạm Thành Trung (Bio-Signal & Time-Series Specialist)",
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "n_subjects": len(summaries),
        "total_seconds": total,
        "valid_seconds": valid,
        "valid_rate": round(valid / total, 4) if total else 0.0,
        "subjects": summaries,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Parse WESAD pickles into 1 Hz HR CSVs")
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    args = parser.parse_args(argv)

    root = RAW_BIO_DIR / ("WESAD" if args.source == "real" else "wesad_synthetic")
    out_dir = RAW_BIO_DIR / ("hr_csv" if args.source == "real" else "hr_csv_synthetic")
    report_path = REPORTS_DIR / ("bio_data_summary.json" if args.source == "real"
                                 else "bio_data_summary_synthetic.json")
    if not find_subject_pickles(root):
        print(f"[!] Không tìm thấy file S*/S*.pkl trong {root}")
        return 1
    report = parse_directory(root, out_dir, args.source, report_path)
    print(f"Nguồn dữ liệu: {args.source} | số đối tượng: {report['n_subjects']} | "
          f"tỷ lệ giây hợp lệ: {report['valid_rate']:.2%}")
    print(f"CSV: {out_dir}\nBáo cáo: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
