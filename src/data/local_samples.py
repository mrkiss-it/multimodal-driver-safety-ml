"""
Local Vietnam (Ho Chi Minh City) Helmet Sample Management, Schema Validation,
and Synthetic Pilot Generator for Pipeline Verification.

Course: Máy học nâng cao (Advanced Machine Learning) - UIT, VNU-HCM
Lecturer: ThS./TS. Đặng Việt Dũng
Author: Hoàng Cao Sơn (MSSV: 25730061) - Data Engineering & Explainable AI (XAI)

Academic Integrity Note:
    Real street/campus field photos collected in Ho Chi Minh City belong in
    ``data/raw/local_samples/``. When real field images have not yet been
    transferred to disk, ``generate_synthetic_local_pilot()`` creates 50
    structured synthetic pilot samples in ``data/raw/local_samples_synthetic/``
    with ``is_synthetic=True`` on every record so the end-to-end ingestion,
    EDA, and Week 3 DataLoader pipelines can be dry-run and unit-tested honestly.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REAL_LOCAL_SAMPLES_DIR: Path = PROJECT_ROOT / "data" / "raw" / "local_samples"
SYNTHETIC_LOCAL_SAMPLES_DIR: Path = (
    PROJECT_ROOT / "data" / "raw" / "local_samples_synthetic"
)
DEFAULT_REPORT_PATH: Path = (
    PROJECT_ROOT / "reports" / "local_samples_pilot_summary.json"
)

CLASS_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Drowsiness",
    2: "Assault",
}

HELMET_TYPES: Tuple[str, ...] = (
    "grab_green",
    "be_yellow",
    "shopeefood_orange",
    "three_quarter_visor",
    "half_helmet_mask",
)

HELMET_RGB_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "grab_green": (0, 155, 65),
    "be_yellow": (240, 195, 20),
    "shopeefood_orange": (235, 90, 30),
    "three_quarter_visor": (45, 75, 135),
    "half_helmet_mask": (70, 75, 85),
}

LIGHTING_CONDITIONS: Tuple[str, ...] = (
    "daylight_sunny",
    "tree_shadow",
    "dusk_streetlight",
    "night_glare",
)

LIGHTING_BG_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "daylight_sunny": (212, 224, 235),
    "tree_shadow": (148, 165, 155),
    "dusk_streetlight": (105, 98, 118),
    "night_glare": (52, 56, 70),
}

OCCLUSION_TYPES: Tuple[str, ...] = (
    "sunglasses_and_mask",
    "tinted_visor_and_mask",
    "cloth_mask_only",
    "medical_mask_and_glasses",
)

REQUIRED_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "subject_id",
    "class_label",
    "class_name",
    "helmet_type",
    "lighting_condition",
    "occlusion_type",
    "head_pitch_deg_est",
    "has_rear_passenger",
    "is_synthetic",
    "file_name",
    "width",
    "height",
)

VALID_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def check_local_samples_status(
    real_dir: Path = REAL_LOCAL_SAMPLES_DIR,
    synthetic_dir: Path = SYNTHETIC_LOCAL_SAMPLES_DIR,
) -> Dict[str, Any]:
    """Check availability of real HCMC field samples vs synthetic pilot samples.

    Args:
        real_dir: Directory reserved for real HCMC field images.
        synthetic_dir: Directory storing synthetic pilot images and manifest.

    Returns:
        Dictionary summarizing file counts and real-data collection status.
    """
    real_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = real_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")

    real_images = [
        p
        for p in sorted(real_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    synth_images = (
        [
            p
            for p in sorted(synthetic_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        if synthetic_dir.exists()
        else []
    )

    return {
        "real_dir": str(real_dir),
        "real_image_count": len(real_images),
        "real_field_data_collected": len(real_images) > 0,
        "synthetic_pilot_dir": str(synthetic_dir),
        "synthetic_pilot_image_count": len(synth_images),
        "synthetic_manifest_exists": (synthetic_dir / "manifest.csv").is_file(),
    }


def _allocate_stratified_labels(num_samples: int) -> List[int]:
    """Allocate exact 70% Normal : 20% Drowsiness : 10% Assault class labels."""
    n_normal = int(round(0.70 * num_samples))
    n_drowsy = int(round(0.20 * num_samples))
    n_assault = num_samples - n_normal - n_drowsy
    if n_assault < 1:
        n_assault = 1
        n_normal = max(1, num_samples - n_drowsy - n_assault)
    return [0] * n_normal + [1] * n_drowsy + [2] * n_assault


def _render_synthetic_helmet_scene(
    rng: np.random.Generator,
    class_label: int,
    helmet_type: str,
    lighting: str,
    occlusion: str,
    pitch_deg: float,
    has_passenger: bool,
    image_size: Tuple[int, int],
) -> Image.Image:
    """Render a deterministic synthetic HCMC helmet test frame."""
    width, height = image_size
    base_bg = np.array(LIGHTING_BG_PALETTE[lighting], dtype=np.int32)
    jitter = rng.integers(-10, 11, size=3)
    bg_color = tuple(int(np.clip(c, 15, 245)) for c in (base_bg + jitter))

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Street horizon / handlebar lower dash silhouette
    draw.rectangle([0, int(height * 0.84), width, height], fill=(38, 42, 48))

    # Optional rear passenger silhouette before foreground driver
    if has_passenger:
        p_cx = int(width * 0.56)
        p_cy = int(height * 0.36)
        draw.ellipse(
            [p_cx - 24, p_cy - 22, p_cx + 24, p_cy + 22],
            fill=(95, 105, 120),
            outline=(50, 55, 65),
            width=2,
        )
        draw.rectangle([p_cx - 28, p_cy + 20, p_cx + 28, int(height * 0.84)], fill=(75, 82, 95))

    # Driver torso & shoulders
    cx = int(width * 0.48 + rng.integers(-5, 6))
    draw.polygon(
        [
            (cx - 44, int(height * 0.84)),
            (cx - 32, int(height * 0.58)),
            (cx + 32, int(height * 0.58)),
            (cx + 44, int(height * 0.84)),
        ],
        fill=(45, 95, 65) if helmet_type == "grab_green" else (60, 72, 92),
    )

    # Helmet vertical drop proportional to forward head pitch (nodding / micro-sleep)
    vertical_shift = int(np.clip((pitch_deg - 8.0) * 0.95, -6, 34))
    lateral_shift = int(rng.integers(-6, 7)) if class_label == 1 else 0
    h_cx = cx + lateral_shift
    h_cy = int(height * 0.40) + vertical_shift

    helmet_rgb = HELMET_RGB_PALETTE[helmet_type]
    draw.ellipse(
        [h_cx - 30, h_cy - 28, h_cx + 30, h_cy + 26],
        fill=helmet_rgb,
        outline=(20, 24, 30),
        width=3,
    )

    # Occlusion: sunglasses / visor + facial mask (typical 95% Vietnam riders)
    if "visor" in occlusion or helmet_type == "three_quarter_visor":
        draw.rectangle(
            [h_cx - 24, h_cy - 6, h_cx + 24, h_cy + 14],
            fill=(30, 38, 48),
            outline=(15, 18, 22),
        )
    else:
        draw.rectangle([h_cx - 20, h_cy - 4, h_cx + 20, h_cy + 6], fill=(22, 22, 26))

    draw.rectangle(
        [h_cx - 19, h_cy + 8, h_cx + 19, h_cy + 25],
        fill=(175, 215, 238) if "medical" in occlusion else (225, 225, 230),
        outline=(110, 130, 145),
    )

    # Class 2 (Assault): rear arm reaching across neck + prop metallic object
    if class_label == 2:
        draw.line(
            [(width - 18, int(height * 0.62)), (h_cx + 6, h_cy + 24)],
            fill=(115, 75, 65),
            width=8,
        )
        draw.polygon(
            [
                (h_cx + 10, h_cy + 18),
                (h_cx + 42, h_cy - 14),
                (h_cx + 48, h_cy - 8),
                (h_cx + 16, h_cy + 24),
            ],
            fill=(228, 232, 240),
            outline=(90, 95, 105),
        )

    # Diagonal tree-shadow band or night-streetlight glare spot
    if lighting == "tree_shadow":
        draw.polygon(
            [(0, int(height * 0.18)), (width, int(height * 0.42)), (width, int(height * 0.58)), (0, int(height * 0.34))],
            fill=(85, 98, 92),
        )
    elif lighting == "night_glare":
        gx = int(width * 0.82)
        gy = int(height * 0.16)
        draw.ellipse([gx - 14, gy - 14, gx + 14, gy + 14], fill=(250, 242, 195))

    return img


def generate_synthetic_local_pilot(
    output_dir: Path = SYNTHETIC_LOCAL_SAMPLES_DIR,
    num_samples: int = 50,
    seed: int = 42,
    image_size: Tuple[int, int] = (160, 160),
) -> pd.DataFrame:
    """Generate 50 structured synthetic HCMC helmet pilot samples & manifest.csv.

    Args:
        output_dir: Destination folder for synthetic `.jpg` frames and `manifest.csv`.
        num_samples: Total pilot samples to generate (must be >= 10, default 50).
        seed: Non-negative RNG seed for deterministic reproducibility.
        image_size: `(width, height)` in pixels (default `(160, 160)`).

    Returns:
        DataFrame containing the generated sample manifest.

    Raises:
        ValueError: If `num_samples < 10` or `seed < 0`.
    """
    if num_samples < 10:
        raise ValueError(f"num_samples must be >= 10, got {num_samples}.")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_subdir = output_dir / "images"
    images_subdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    labels = _allocate_stratified_labels(num_samples)

    num_subjects = min(10, num_samples)
    subjects = [f"VN_SYN_S{(i % num_subjects) + 1:02d}" for i in range(num_samples)]

    records: List[Dict[str, Any]] = []
    for idx in range(num_samples):
        cls = labels[idx]
        helmet = HELMET_TYPES[idx % len(HELMET_TYPES)]
        lighting = LIGHTING_CONDITIONS[idx % len(LIGHTING_CONDITIONS)]
        occlusion = OCCLUSION_TYPES[idx % len(OCCLUSION_TYPES)]
        subject_id = subjects[idx]

        if cls == 0:
            pitch_deg = round(float(rng.uniform(3.0, 17.5)), 2)
            has_passenger = bool(idx % 2 == 0)
        elif cls == 1:
            pitch_deg = round(float(rng.uniform(26.5, 42.0)), 2)
            has_passenger = bool(idx % 3 == 0)
        else:
            pitch_deg = round(float(rng.uniform(8.0, 24.0)), 2)
            has_passenger = True

        sample_id = f"VN_PILOT_{idx + 1:03d}"
        file_name = f"images/{sample_id.lower()}_c{cls}_{helmet}.jpg"
        abs_img_path = output_dir / file_name

        img = _render_synthetic_helmet_scene(
            rng=rng,
            class_label=cls,
            helmet_type=helmet,
            lighting=lighting,
            occlusion=occlusion,
            pitch_deg=pitch_deg,
            has_passenger=has_passenger,
            image_size=image_size,
        )
        img.save(abs_img_path, format="JPEG", quality=92)

        records.append(
            {
                "sample_id": sample_id,
                "subject_id": subject_id,
                "class_label": int(cls),
                "class_name": CLASS_NAMES[cls],
                "helmet_type": helmet,
                "lighting_condition": lighting,
                "occlusion_type": occlusion,
                "head_pitch_deg_est": pitch_deg,
                "has_rear_passenger": has_passenger,
                "is_synthetic": True,
                "file_name": file_name,
                "width": int(image_size[0]),
                "height": int(image_size[1]),
            }
        )

    df = pd.DataFrame(records, columns=list(REQUIRED_MANIFEST_COLUMNS))
    manifest_path = output_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    return df


def validate_local_samples(
    samples_dir: Path,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate manifest schema, image integrity, RGB channels, and distributions.

    Args:
        samples_dir: Root directory containing `manifest.csv` and sample images.
        manifest_path: Optional explicit path to `manifest.csv`.

    Returns:
        Comprehensive validation and statistical profiling dictionary.
    """
    m_path = manifest_path if manifest_path is not None else (samples_dir / "manifest.csv")
    if not m_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {m_path}")

    df = pd.read_csv(m_path)
    missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {missing_cols}")

    total_rows = int(len(df))
    valid_images = 0
    missing_images = 0
    corrupt_images = 0
    errors: List[Dict[str, str]] = []
    brightness_by_lighting: Dict[str, List[float]] = {}

    for _, row in df.iterrows():
        img_path = samples_dir / str(row["file_name"])
        if not img_path.is_file():
            missing_images += 1
            errors.append({"sample_id": str(row["sample_id"]), "error": "missing_file"})
            continue

        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                w, h = im.size
                bands = len(im.getbands())
                mean_luma = float(np.asarray(im, dtype=np.float32).mean())

            if bands != 3 or w != int(row["width"]) or h != int(row["height"]):
                corrupt_images += 1
                errors.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "error": f"dimension_or_band_mismatch ({w}x{h}, bands={bands})",
                    }
                )
                continue

            valid_images += 1
            l_key = str(row["lighting_condition"])
            brightness_by_lighting.setdefault(l_key, []).append(mean_luma)
        except Exception as exc:
            corrupt_images += 1
            errors.append({"sample_id": str(row["sample_id"]), "error": str(exc)})

    class_counts = {
        CLASS_NAMES.get(int(k), str(k)): int(v)
        for k, v in df["class_label"].value_counts().sort_index().items()
    }
    class_ratios = {
        k: round(v / total_rows, 4) if total_rows > 0 else 0.0
        for k, v in class_counts.items()
    }
    helmet_counts = {
        str(k): int(v) for k, v in df["helmet_type"].value_counts().sort_index().items()
    }
    lighting_counts = {
        str(k): int(v)
        for k, v in df["lighting_condition"].value_counts().sort_index().items()
    }
    mean_brightness = {
        k: round(float(np.mean(vals)), 2) for k, vals in sorted(brightness_by_lighting.items())
    }
    pitch_by_class = {
        CLASS_NAMES.get(int(k), str(k)): round(float(v), 2)
        for k, v in df.groupby("class_label")["head_pitch_deg_est"].mean().items()
    }

    is_valid = (
        total_rows > 0
        and valid_images == total_rows
        and missing_images == 0
        and corrupt_images == 0
    )

    return {
        "samples_dir": str(samples_dir),
        "manifest_path": str(m_path),
        "all_synthetic": bool(df["is_synthetic"].all()),
        "total_manifest_rows": total_rows,
        "valid_images": valid_images,
        "missing_images": missing_images,
        "corrupt_images": corrupt_images,
        "integrity_rate_pct": round((valid_images / total_rows * 100.0), 2)
        if total_rows > 0
        else 0.0,
        "is_valid": is_valid,
        "unique_subjects": int(df["subject_id"].nunique()),
        "class_counts": class_counts,
        "class_ratios": class_ratios,
        "helmet_type_counts": helmet_counts,
        "lighting_condition_counts": lighting_counts,
        "mean_brightness_by_lighting": mean_brightness,
        "mean_head_pitch_deg_by_class": pitch_by_class,
        "errors": errors,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point to generate/validate local pilot samples and write JSON report."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Manage and validate HCMC local helmet samples & synthetic pilot."
    )
    parser.add_argument(
        "--generate-pilot",
        action="store_true",
        help="Generate 50 synthetic pilot samples in data/raw/local_samples_synthetic/.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of synthetic pilot samples to generate (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic pilot generation (default: 42).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to write JSON verification summary.",
    )
    args = parser.parse_args(argv)

    if args.generate_pilot:
        generate_synthetic_local_pilot(
            output_dir=SYNTHETIC_LOCAL_SAMPLES_DIR,
            num_samples=args.num_samples,
            seed=args.seed,
        )

    status = check_local_samples_status()
    pilot_validation = validate_local_samples(SYNTHETIC_LOCAL_SAMPLES_DIR)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "maintainer": "Hoang Cao Son (MSSV: 25730061) - Data Engineering & XAI",
        "week": 2,
        "data_transparency_notice": (
            "Real HCMC street/campus photos have NOT yet been captured; "
            "data/raw/local_samples/ currently has 0 real photos. "
            "All 50 pilot images validated below are SYNTHETIC dry-run samples "
            "(is_synthetic=True) stored in data/raw/local_samples_synthetic/."
        ),
        "directory_status": status,
        "pilot_validation": pilot_validation,
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"[LOCAL SAMPLES] Real field photos: {status['real_image_count']} | "
        f"Synthetic pilot photos: {pilot_validation['valid_images']}/{pilot_validation['total_manifest_rows']} "
        f"(integrity={pilot_validation['integrity_rate_pct']}%)"
    )
    print(f"[STRATIFICATION] Classes={pilot_validation['class_counts']} | Subjects={pilot_validation['unique_subjects']}")
    print(f"[REPORT] Saved JSON summary -> {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
