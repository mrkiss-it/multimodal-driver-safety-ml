"""
Unit Tests for HCMC Local Helmet Pilot Samples, Schema Validation, and Week 2 EDA Notebook.

Course: Máy học nâng cao (Advanced Machine Learning) - UIT, VNU-HCM
Author: Hoàng Cao Sơn (MSSV: 25730061) - Data Engineering & Explainable AI (XAI)
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.local_samples import (
    HELMET_TYPES,
    LIGHTING_CONDITIONS,
    PROJECT_ROOT,
    REQUIRED_MANIFEST_COLUMNS,
    check_local_samples_status,
    generate_synthetic_local_pilot,
    main as local_samples_main,
    validate_local_samples,
)


def test_pilot_generation_count_and_70_20_10_stratification(tmp_path: Path) -> None:
    """50 pilot samples must follow 35:10:5 (70:20:10) and cover all 5 helmet types."""
    out_dir = tmp_path / "pilot_50"
    df = generate_synthetic_local_pilot(output_dir=out_dir, num_samples=50, seed=42)

    assert len(df) == 50
    assert list(df.columns) == list(REQUIRED_MANIFEST_COLUMNS)
    assert bool(df["is_synthetic"].all()) is True
    assert df["subject_id"].nunique() == 10

    counts = df["class_label"].value_counts().to_dict()
    assert counts == {0: 35, 1: 10, 2: 5}

    helmet_counts = df["helmet_type"].value_counts().to_dict()
    for h_type in HELMET_TYPES:
        assert helmet_counts[h_type] == 10

    lighting_present = set(df["lighting_condition"].unique())
    assert lighting_present == set(LIGHTING_CONDITIONS)

    # Drowsiness head pitch must strictly exceed Normal head pitch on average
    pitch_means = df.groupby("class_label")["head_pitch_deg_est"].mean()
    assert float(pitch_means[1]) > 25.0
    assert float(pitch_means[0]) < 20.0


def test_pilot_generation_reproducibility(tmp_path: Path) -> None:
    """Identical seeds must generate identical manifest records and image bytes."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    df_a = generate_synthetic_local_pilot(output_dir=dir_a, num_samples=20, seed=99)
    df_b = generate_synthetic_local_pilot(output_dir=dir_b, num_samples=20, seed=99)

    pd.testing.assert_frame_equal(df_a, df_b)

    first_rel = str(df_a.iloc[0]["file_name"])
    assert (dir_a / first_rel).read_bytes() == (dir_b / first_rel).read_bytes()


def test_validate_local_samples_passes_valid_pilot(tmp_path: Path) -> None:
    """validate_local_samples must report 100% integrity on freshly generated pilot."""
    out_dir = tmp_path / "valid_pilot"
    generate_synthetic_local_pilot(output_dir=out_dir, num_samples=30, seed=7)

    result = validate_local_samples(out_dir)
    assert result["is_valid"] is True
    assert result["all_synthetic"] is True
    assert result["valid_images"] == 30
    assert result["missing_images"] == 0
    assert result["corrupt_images"] == 0
    assert result["integrity_rate_pct"] == 100.0
    # Daylight sunny images must be brighter than night glare images
    luma = result["mean_brightness_by_lighting"]
    assert luma["daylight_sunny"] > luma["night_glare"]


def test_validate_local_samples_detects_corrupt_and_missing_files(tmp_path: Path) -> None:
    """Corrupted and missing image files must be caught and flagged as invalid."""
    out_dir = tmp_path / "broken_pilot"
    df = generate_synthetic_local_pilot(output_dir=out_dir, num_samples=15, seed=11)

    corrupt_target = out_dir / str(df.iloc[0]["file_name"])
    corrupt_target.write_bytes(b"NOT_A_VALID_JPEG_HEADER")

    missing_target = out_dir / str(df.iloc[1]["file_name"])
    missing_target.unlink()

    result = validate_local_samples(out_dir)
    assert result["is_valid"] is False
    assert result["corrupt_images"] == 1
    assert result["missing_images"] == 1
    assert result["valid_images"] == 13
    assert len(result["errors"]) == 2


def test_check_local_samples_status_empty_real_dir(tmp_path: Path) -> None:
    """Empty real directory must honestly report real_field_data_collected == False."""
    real_dir = tmp_path / "real_local"
    synth_dir = tmp_path / "synth_local"
    generate_synthetic_local_pilot(output_dir=synth_dir, num_samples=10, seed=1)

    status = check_local_samples_status(real_dir=real_dir, synthetic_dir=synth_dir)
    assert status["real_image_count"] == 0
    assert status["real_field_data_collected"] is False
    assert (real_dir / ".gitkeep").is_file()
    assert status["synthetic_pilot_image_count"] == 10
    assert status["synthetic_manifest_exists"] is True


def test_invalid_parameters_raise_value_error(tmp_path: Path) -> None:
    """Defensive checks on num_samples, seed, and missing manifest."""
    with pytest.raises(ValueError, match="num_samples"):
        generate_synthetic_local_pilot(output_dir=tmp_path / "too_small", num_samples=5)

    with pytest.raises(ValueError, match="seed"):
        generate_synthetic_local_pilot(output_dir=tmp_path / "neg_seed", num_samples=10, seed=-3)

    with pytest.raises(FileNotFoundError):
        validate_local_samples(tmp_path / "nonexistent_dir")


def test_local_samples_cli_report_output(tmp_path: Path) -> None:
    """CLI entry point writes valid JSON report with data_transparency_notice."""
    out_report = tmp_path / "pilot_report.json"
    code = local_samples_main(["--output-report", str(out_report)])
    assert code == 0
    assert out_report.is_file()
    payload = json.loads(out_report.read_text(encoding="utf-8"))
    assert payload["week"] == 2
    assert "SYNTHETIC" in payload["data_transparency_notice"]
    assert payload["pilot_validation"]["is_valid"] is True


def test_eda_notebook_structure_and_executed_outputs() -> None:
    """Verify 01_eda_and_data_preprocessing.ipynb is valid nbformat v4 and cleanly executed."""
    nb_path = PROJECT_ROOT / "notebooks" / "01_eda_and_data_preprocessing.ipynb"
    assert nb_path.is_file(), "Missing notebooks/01_eda_and_data_preprocessing.ipynb"

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]

    assert len(code_cells) >= 4
    assert len(md_cells) >= 4

    for idx, cell in enumerate(code_cells):
        assert cell.get("execution_count") is not None, (
            f"Code cell {idx} in EDA notebook has not been executed."
        )
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", (
                f"Code cell {idx} raised error: {out.get('ename')}: {out.get('evalue')}"
            )
