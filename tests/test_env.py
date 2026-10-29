"""
Automated Environment, Dependency, Reproducibility, and Hardware/CUDA Unit Tests.

Course: Máy học nâng cao (Advanced Machine Learning) - UIT, VNU-HCM
Author: Vũ Văn Duy (MSSV: 26410031) - ML Modeling & Evaluation Lead
"""

import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.losses import MultiClassFocalLoss
from src.utils.env_check import (
    PROJECT_ROOT,
    main as env_check_main,
    probe_hardware_and_cuda,
    resolve_compute_device,
    set_global_seed,
)

CORE_REQUIRED_IMPORTS = (
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "torch",
    "lightgbm",
    "xgboost",
    "PIL",
    "cv2",
    "matplotlib",
    "seaborn",
    "docx",
    "pytest",
)


def test_python_runtime_version() -> None:
    """Project requires Python >= 3.10 for modern type annotations and ML wheels."""
    assert sys.version_info >= (3, 10), (
        f"Expected Python >= 3.10, found {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_environment_spec_files_exist_and_complete() -> None:
    """Verify requirements.txt and environment.yml exist and list all core packages."""
    req_path = PROJECT_ROOT / "requirements.txt"
    conda_path = PROJECT_ROOT / "environment.yml"

    assert req_path.is_file(), "Missing requirements.txt at project root."
    assert conda_path.is_file(), "Missing environment.yml at project root."

    req_text = req_path.read_text(encoding="utf-8").lower()
    conda_text = conda_path.read_text(encoding="utf-8").lower()

    expected_req_tokens = (
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "torchvision",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "pillow",
        "opencv-python-headless",
        "albumentations",
        "matplotlib",
        "seaborn",
        "shap",
        "python-docx",
        "pytest",
    )
    for token in expected_req_tokens:
        assert token in req_text, f"Package '{token}' missing from requirements.txt"

    assert "name: driver-safety-ml" in conda_text
    assert "python=3.10" in conda_text
    for token in ("pytorch", "torchvision", "lightgbm", "xgboost", "albumentations", "shap"):
        assert token in conda_text, f"Package '{token}' missing from environment.yml"


@pytest.mark.parametrize("module_name", CORE_REQUIRED_IMPORTS)
def test_core_ml_and_scientific_imports(module_name: str) -> None:
    """Verify every core scientific, GBDT, vision, and testing library imports cleanly."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_torchvision_mobilenetv3_smoke() -> None:
    """Verify MobileNetV3-Small backbone runs forward pass on 3x96x96 project input."""
    torchvision = pytest.importorskip(
        "torchvision",
        reason="torchvision not installed; run 'pip install -r requirements.txt' in .venv.",
    )
    set_global_seed(42)
    model = torchvision.models.mobilenet_v3_small(weights=None, num_classes=3)
    model.eval()
    dummy_images = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        logits = model(dummy_images)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_albumentations_spatial_pipeline_smoke() -> None:
    """Verify Albumentations spatial transform runs on a 96x96x3 RGB frame."""
    A = pytest.importorskip(
        "albumentations",
        reason="albumentations not installed; run 'pip install -r requirements.txt' in .venv.",
    )
    set_global_seed(42)
    pipeline = A.Compose([A.MotionBlur(blur_limit=5, p=1.0)])
    dummy_frame = np.full((96, 96, 3), fill_value=128, dtype=np.uint8)
    transformed = pipeline(image=dummy_frame)["image"]
    assert transformed.shape == (96, 96, 3)
    assert transformed.dtype == np.uint8


def test_shap_tree_explainer_smoke() -> None:
    """Verify LightGBM + SHAP TreeExplainer integration on 8-D HRV-like features."""
    shap = pytest.importorskip(
        "shap",
        reason="shap not installed; run 'pip install -r requirements.txt' in .venv.",
    )
    import lightgbm as lgb

    rng = np.random.default_rng(42)
    x_dummy = rng.normal(size=(30, 8))
    y_dummy = np.array([0] * 15 + [1] * 10 + [2] * 5)

    clf = lgb.LGBMClassifier(
        n_estimators=5, max_depth=3, min_child_samples=3, random_state=42, verbose=-1
    )
    clf.fit(x_dummy, y_dummy)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_dummy[:4])
    assert shap_values is not None


def test_global_seed_reproducibility() -> None:
    """Verify set_global_seed makes Python, NumPy, and PyTorch RNGs reproducible."""
    set_global_seed(1234)
    py_a = random.random()
    np_a = np.random.randn(3, 3)
    th_a = torch.randn(3, 3)

    set_global_seed(1234)
    py_b = random.random()
    np_b = np.random.randn(3, 3)
    th_b = torch.randn(3, 3)

    assert py_a == py_b
    assert np.array_equal(np_a, np_b)
    assert torch.equal(th_a, th_b)

    with pytest.raises(ValueError, match="non-negative"):
        set_global_seed(-1)


def test_compute_device_resolution_and_autograd() -> None:
    """Verify device resolution (CUDA or CPU fallback) and PyTorch autograd step."""
    cpu_device = resolve_compute_device(prefer_cuda=False)
    assert cpu_device == torch.device("cpu")

    active_device = resolve_compute_device(prefer_cuda=True)
    expected_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert active_device.type == expected_type

    hw_info = probe_hardware_and_cuda()
    assert hw_info["cuda_available"] is torch.cuda.is_available()
    assert hw_info["resolved_device"].startswith(expected_type)

    set_global_seed(42)
    net = torch.nn.Sequential(
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    ).to(active_device)
    criterion = MultiClassFocalLoss(gamma=2.0).to(active_device)

    x = torch.randn(6, 16, device=active_device)
    y = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long, device=active_device)

    loss = criterion(net(x), y)
    loss.backward()

    assert torch.isfinite(loss).item()
    for param in net.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all().item()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No NVIDIA CUDA GPU detected on this workstation; CPU fallback verified.",
)
def test_cuda_gpu_kernel_when_available() -> None:
    """Execute real CUDA tensor algebra, focal loss, and synchronization when GPU exists."""
    device = torch.device("cuda:0")
    a = torch.randn(32, 32, device=device)
    b = torch.randn(32, 3, device=device, requires_grad=True)
    logits = a @ b
    targets = torch.randint(0, 3, (32,), device=device)

    loss = MultiClassFocalLoss(gamma=2.0)(logits, targets)
    loss.backward()
    torch.cuda.synchronize(device)

    assert logits.is_cuda
    assert b.grad is not None and torch.isfinite(b.grad).all().item()


def test_env_check_cli_writes_valid_json(tmp_path: Path) -> None:
    """Verify env_check CLI writes a valid JSON report without touching tracked files."""
    out_json = tmp_path / "env_verification.json"
    exit_code = env_check_main(["--output", str(out_json), "--seed", "42"])
    assert exit_code == 0
    assert out_json.is_file()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["week"] == 2
    assert "hardware_and_runtime" in payload
    assert "library_summary" in payload
    assert payload["loss_mathematical_audit"]["ce_vs_focal_gamma0_abs_diff"] < 1e-6
    assert payload["loss_mathematical_audit"]["backward_gradients_finite"] is True
