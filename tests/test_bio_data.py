"""Unit tests for the bio-signal pipeline (WESAD ingestion + parsing).

Author: Phạm Thành Trung (Bio-Signal & Time-Series Specialist)

All tests write to ``tmp_path`` so they never touch real data under ``data/raw``.
Accuracy tests run on SYNTHETIC data; they verify the code logic, not real-world
performance on WESAD.
"""

import json
import pickle

import numpy as np
import pytest

from src.data import download_bio as db
from src.data import parse_wesad as pw


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    """Two synthetic subjects shared by the tests in this module."""
    root = tmp_path_factory.mktemp("wesad_synth")
    db.generate_synthetic_wesad(root, subjects=["S2", "S3"], seed=123)
    return root


# ------------------------------ download_bio -------------------------------- #
def test_wesad_subject_list():
    assert len(db.WESAD_SUBJECTS) == 15
    assert "S1" not in db.WESAD_SUBJECTS and "S12" not in db.WESAD_SUBJECTS
    assert db.WESAD_SUBJECTS[0] == "S2" and db.WESAD_SUBJECTS[-1] == "S17"


def test_status_empty_dir(tmp_path):
    status = db.check_wesad_status(tmp_path)
    assert status["n_found"] == 0
    assert len(status["missing_subjects"]) == 15
    assert status["is_complete"] is False


def test_status_detects_files(synth_root):
    status = db.check_wesad_status(synth_root)
    assert status["found_subjects"] == ["S2", "S3"]
    assert status["n_found"] == 2
    assert status["is_complete"] is False
    assert status["size_mb"] > 0


def test_synthetic_pickle_structure(synth_root):
    with open(synth_root / "S2" / "S2.pkl", "rb") as f:
        data = pickle.load(f)
    assert data["subject"] == "S2"
    assert data["synthetic"] is True
    wrist = data["signal"]["wrist"]
    n_sec = len(data["label"]) // db.LABEL_FS
    assert len(wrist["BVP"]) == n_sec * db.BVP_FS
    assert len(wrist["ACC"]) == n_sec * db.ACC_FS
    assert wrist["ACC"].shape[1] == 3
    assert set(np.unique(data["label"])) <= {0, 1, 2, 3, 4}


def test_synthetic_is_reproducible(tmp_path):
    a = db.generate_synthetic_wesad(tmp_path / "a", subjects=["S4"], seed=7)[0]
    b = db.generate_synthetic_wesad(tmp_path / "b", subjects=["S4"], seed=7)[0]
    with open(a, "rb") as fa, open(b, "rb") as fb:
        da, dbb = pickle.load(fa), pickle.load(fb)
    assert np.array_equal(da["signal"]["wrist"]["BVP"], dbb["signal"]["wrist"]["BVP"])


# ------------------------------- parse_wesad -------------------------------- #
def test_bandpass_removes_dc_keeps_pulse():
    fs = 64
    t = np.arange(0, 20, 1 / fs)
    x = 5.0 + np.sin(2 * np.pi * 1.2 * t)
    y = pw.bandpass(x, fs)
    assert abs(np.mean(y)) < 0.05
    assert np.std(y) > 0.5


def test_bandpass_short_signal_does_not_raise():
    assert pw.bandpass(np.array([1.0, 2.0, 3.0]), 64).shape == (3,)
    assert pw.bandpass(np.array([]), 64).size == 0


def test_peaks_to_hr_exact_rate_and_gap():
    fs = 64
    t_peaks = np.arange(0.4, 60.0, 0.8)           # 75 BPM
    t_peaks = t_peaks[(t_peaks < 20) | (t_peaks > 40)]  # 20 s gap
    hr, covered = pw.peaks_to_hr_1hz(np.round(t_peaks * fs).astype(int), fs, 60)
    # Peak times are quantised to 1 sample (1/64 s): the worst-case HR error is
    # 75 * (1/64) / 0.8 ~= 1.46 BPM, so a 1.5 BPM tolerance is the exact bound.
    assert np.allclose(hr[covered][:10], 75.0, atol=1.5)
    assert np.isnan(hr[30]) and not covered[30]


def test_peaks_to_hr_too_few_peaks():
    hr, covered = pw.peaks_to_hr_1hz(np.array([10, 80]), 64, 10)
    assert np.all(np.isnan(hr)) and not covered.any()


def test_labels_to_1hz_mixed_second_is_transition():
    fs = pw.LABEL_FS
    lab = np.concatenate([np.full(fs, 1), np.full(fs // 2, 1), np.full(fs // 2, 2), np.full(fs, 2)])
    out = pw.labels_to_1hz(lab, 3)
    assert out.tolist() == [1, 0, 2]


def test_project_label_mapping():
    assert pw.map_to_project_label(1) == 0
    assert pw.map_to_project_label(3) == 0
    assert pw.map_to_project_label(4) == 1
    assert pw.map_to_project_label(2) == 2
    assert pw.map_to_project_label(0) == -1
    assert pw.map_to_project_label(7) == -1


def test_parse_subject_columns_and_alignment(synth_root):
    df = pw.parse_subject(synth_root / "S2" / "S2.pkl")
    assert list(df.columns) == ["subject", "t_sec", "hr_bpm", "label_wesad", "label_name",
                                "project_label", "is_dropout", "is_motion", "is_valid"]
    assert (df["t_sec"].diff().dropna() == 1).all()
    assert df.loc[df["is_valid"], "hr_bpm"].between(pw.HR_MIN_BPM, pw.HR_MAX_BPM).all()


def test_parse_subject_hr_accuracy_vs_ground_truth(synth_root):
    path = synth_root / "S2" / "S2.pkl"
    with open(path, "rb") as f:
        true_hr = pickle.load(f)["meta"]["true_hr_1hz"]
    df = pw.parse_subject(path)
    v = df["is_valid"].values
    err = np.abs(df["hr_bpm"].values[v] - true_hr[: len(df)][v])
    assert v.mean() > 0.85
    assert err.mean() < 2.0          # measured ~0.5 BPM on synthetic data
    assert err.max() < 6.0


def test_parse_subject_flags_injected_artifacts(synth_root):
    path = synth_root / "S3" / "S3.pkl"
    with open(path, "rb") as f:
        meta = pickle.load(f)["meta"]
    df = pw.parse_subject(path)
    for start, length in meta["dropout_intervals"]:
        assert df["is_dropout"].values[start:start + length].all()
    for start, length in meta["motion_intervals"]:
        assert df["is_motion"].values[start:start + length].all()
        assert not df["is_valid"].values[start:start + length].any()


def test_hr_orders_by_state(synth_root):
    """Stress must be higher than baseline, meditation lower (protocol design)."""
    summary = pw.summarize_subject(pw.parse_subject(synth_root / "S2" / "S2.pkl"))
    hr = summary["hr_by_label"]
    assert hr["stress"] > hr["baseline"] > hr["meditation"]


def test_parse_directory_writes_outputs(synth_root, tmp_path):
    report_path = tmp_path / "report.json"
    report = pw.parse_directory(synth_root, tmp_path / "csv", "synthetic", report_path)
    assert report["is_synthetic"] is True
    assert report["n_subjects"] == 2
    assert (tmp_path / "csv" / "S2.csv").is_file()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["data_source"] == "synthetic"
    assert 0.0 < saved["valid_rate"] <= 1.0


def test_find_subject_pickles_sorted(synth_root):
    names = [p.stem for p in pw.find_subject_pickles(synth_root)]
    assert names == ["S2", "S3"]
