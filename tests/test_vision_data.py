"""
Unit Tests for Vision Data Ingestion and Verification
Author: Nguyễn Duy Khang (Computer Vision Lead)
"""

import pytest
from pathlib import Path
from PIL import Image
from src.data.download_vision import get_vision_dirs, check_dataset_status, generate_sample_vision_data
from src.data.verify_vision_data import verify_image_file, profile_directory, run_full_verification

def test_get_vision_dirs():
    dirs = get_vision_dirs()
    assert "raw_vision" in dirs
    assert "nthu" in dirs
    assert "weapons" in dirs
    assert "local_helmets" in dirs
    for p in dirs.values():
        assert p.exists()
        assert p.is_dir()

def test_check_dataset_status():
    status = check_dataset_status()
    assert "NTHU-DDD" in status
    assert "Roboflow-Weapons" in status
    assert "Local-Helmets" in status
    for item in status.values():
        assert "image_count" in item
        assert "size_mb" in item
        assert "is_ready" in item

def test_verify_valid_image(tmp_path):
    img_path = tmp_path / "test_valid.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(img_path)

    is_valid, err, dim, ch = verify_image_file(img_path)
    assert is_valid is True
    assert err is None
    assert dim == (100, 100)
    assert ch == 3

def test_verify_corrupt_image(tmp_path):
    corrupt_path = tmp_path / "corrupt.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_AN_IMAGE_FILE_RANDOM_BYTES_123456")

    is_valid, err, dim, ch = verify_image_file(corrupt_path)
    assert is_valid is False
    assert err is not None
    assert dim is None

def test_run_full_verification():
    report = run_full_verification()
    assert "verified_by" in report
    assert "subsystems" in report
    assert "overall_summary" in report
    assert report["overall_summary"]["total_valid"] >= 0
