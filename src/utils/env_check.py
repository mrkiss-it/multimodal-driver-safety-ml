"""
Environment, Reproducibility, and Custom Loss Verification Utility.

Course: Máy học nâng cao (Advanced Machine Learning) - UIT, VNU-HCM
Lecturer: ThS./TS. Đặng Việt Dũng
Author: Vũ Văn Duy (MSSV: 26410031) - ML Modeling & Evaluation Lead
"""

import argparse
import importlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.losses import (
    ClassBalancedFocalLoss,
    CombinedFocalCostLoss,
    CostSensitiveBayesLoss,
    MultiClassFocalLoss,
    bayes_optimal_predict,
)

# Canonical import-name -> distribution label mapping aligned with requirements.txt
REQUIRED_LIBRARIES: Tuple[Tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python / opencv-python-headless"),
    ("albumentations", "albumentations"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("shap", "shap"),
    ("docx", "python-docx"),
    ("pytest", "pytest"),
)


def set_global_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible ML experiments.

    Args:
        seed: Non-negative integer random seed (default ``42``).
        deterministic: Configure cuDNN for deterministic kernel selection when
            running on CUDA devices.

    Returns:
        The applied integer seed.

    Raises:
        ValueError: If ``seed < 0``.
    """
    if seed < 0:
        raise ValueError(f"Random seed must be non-negative, got {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

    return seed


def resolve_compute_device(prefer_cuda: bool = True) -> torch.device:
    """Resolve the active PyTorch compute device with graceful CPU fallback.

    Args:
        prefer_cuda: Return ``cuda`` when an NVIDIA CUDA GPU is available;
            otherwise fall back cleanly to ``cpu``.

    Returns:
        Active :class:`torch.device`.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def probe_installed_libraries() -> Dict[str, Dict[str, Any]]:
    """Import and record version status for all standardized project libraries."""
    results: Dict[str, Dict[str, Any]] = {}
    for import_name, dist_label in REQUIRED_LIBRARIES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "installed")
            results[import_name] = {
                "distribution": dist_label,
                "installed": True,
                "version": str(version),
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - diagnostic capture
            results[import_name] = {
                "distribution": dist_label,
                "installed": False,
                "version": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def probe_hardware_and_cuda() -> Dict[str, Any]:
    """Collect OS, CPU, and PyTorch CUDA/GPU diagnostic metadata."""
    cuda_available = bool(torch.cuda.is_available())
    device_names: List[str] = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            device_names.append(torch.cuda.get_device_name(idx))

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
        "in_virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_compiled_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_names": device_names,
        "resolved_device": str(resolve_compute_device(prefer_cuda=True)),
    }


def run_loss_audit_benchmark(seed: int = 42) -> Dict[str, Any]:
    """Run deterministic mathematical verification benchmarks on custom losses."""
    set_global_seed(seed)
    device = resolve_compute_device(prefer_cuda=True)

    logits = torch.tensor(
        [
            [3.0, -1.0, -2.0],   # High-confidence Normal
            [0.2, 0.3, -0.5],    # Ambiguous Drowsy sample
            [2.0, 0.1, -1.5],    # Dangerous False-Negative Assault (predicted Normal!)
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

    ce_loss = torch.nn.CrossEntropyLoss()(logits, targets)
    fl_gamma0 = MultiClassFocalLoss(gamma=0.0)(logits, targets)
    fl_gamma2 = MultiClassFocalLoss(gamma=2.0)(logits, targets)
    cb_fl = ClassBalancedFocalLoss(
        samples_per_class=[4200, 1200, 600], beta=0.999, gamma=2.0
    )(logits, targets)
    bayes_cost = CostSensitiveBayesLoss()(logits, targets)
    combined = CombinedFocalCostLoss(mu=0.05, gamma=2.0)(logits, targets)
    combined.backward()

    grad_is_finite = bool(
        logits.grad is not None and torch.isfinite(logits.grad).all().item()
    )

    # Inference threshold verification for posterior P = [0.85, 0.05, 0.10]
    sample_posterior = torch.tensor(
        [[0.85, 0.05, 0.10], [0.90, 0.09, 0.01], [0.995, 0.004, 0.001]],
        dtype=torch.float32,
        device=device,
    )
    naive_preds = torch.argmax(sample_posterior, dim=-1).tolist()
    bayes_preds_t, exp_costs_t = bayes_optimal_predict(
        sample_posterior, from_logits=False, return_expected_costs=True
    )

    cb_module = ClassBalancedFocalLoss(
        samples_per_class=[4200, 1200, 600], beta=0.999, gamma=2.0
    )

    return {
        "device_tested": str(device),
        "ce_vs_focal_gamma0_abs_diff": float(
            torch.abs(ce_loss - fl_gamma0).detach().cpu().item()
        ),
        "cross_entropy_mean": round(float(ce_loss.detach().cpu().item()), 6),
        "focal_loss_gamma2_mean": round(float(fl_gamma2.detach().cpu().item()), 6),
        "class_balanced_focal_mean": round(float(cb_fl.detach().cpu().item()), 6),
        "class_balanced_weights_70_20_10": [
            round(float(w), 6) for w in cb_module.alpha.tolist()
        ],
        "expected_bayes_cost_mean": round(float(bayes_cost.detach().cpu().item()), 6),
        "combined_focal_cost_mean": round(float(combined.detach().cpu().item()), 6),
        "backward_gradients_finite": grad_is_finite,
        "bayes_decision_demo": {
            "posteriors": sample_posterior.detach().cpu().tolist(),
            "argmax_predictions": naive_preds,
            "bayes_optimal_predictions": bayes_preds_t.detach().cpu().tolist(),
            "expected_conditional_costs": [
                [round(float(x), 4) for x in row]
                for row in exp_costs_t.detach().cpu().tolist()
            ],
        },
    }


def collect_environment_report(seed: int = 42) -> Dict[str, Any]:
    """Assemble full environment, library, hardware, and loss verification report."""
    libs = probe_installed_libraries()
    installed_count = sum(1 for item in libs.values() if item["installed"])
    total_count = len(libs)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "maintainer": "Vu Van Duy (MSSV: 26410031) - ML Modeling & Evaluation Lead",
        "week": 2,
        "hardware_and_runtime": probe_hardware_and_cuda(),
        "library_summary": {
            "installed_count": installed_count,
            "total_required": total_count,
            "all_installed": installed_count == total_count,
            "packages": libs,
        },
        "loss_mathematical_audit": run_loss_audit_benchmark(seed=seed),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point to verify environment & losses and write JSON summary."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Verify Python ML environment, hardware/CUDA, and custom losses."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "env_and_loss_verification.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for loss verification benchmark.",
    )
    args = parser.parse_args(argv)

    report = collect_environment_report(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    hw = report["hardware_and_runtime"]
    lib_sum = report["library_summary"]
    audit = report["loss_mathematical_audit"]

    print(
        f"[ENV CHECK] Python {hw['python_version']} | "
        f"PyTorch {hw['torch_version']} | "
        f"Resolved device: {hw['resolved_device']} (CUDA available: {hw['cuda_available']})"
    )
    print(
        f"[LIBRARIES] {lib_sum['installed_count']}/{lib_sum['total_required']} "
        f"standardized packages installed."
    )
    print(
        f"[LOSS AUDIT] |CE - FL(gamma=0)| = {audit['ce_vs_focal_gamma0_abs_diff']:.2e} | "
        f"CB weights (70:20:10) = {audit['class_balanced_weights_70_20_10']} | "
        f"Gradients finite = {audit['backward_gradients_finite']}"
    )
    print(f"[REPORT] Saved verification JSON -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
