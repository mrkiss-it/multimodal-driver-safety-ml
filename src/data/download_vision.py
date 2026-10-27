"""
Module: Data Ingestion and Management for Computer Vision Subsystem
Author: Nguyễn Duy Khang (MSSV: 26410055) - Computer Vision Lead
Course: Máy học nâng cao (Advanced Machine Learning) - UIT
Lecturer: ThS./TS. Đặng Việt Dũng
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Root directory of Đồ án
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_VISION_DIR = PROJECT_ROOT / "data" / "raw" / "vision"

NTHU_DIR = RAW_VISION_DIR / "drowsiness_nthu"
WEAPONS_DIR = RAW_VISION_DIR / "weapons_roboflow"
LOCAL_HELMETS_DIR = RAW_VISION_DIR / "local_vietnam_helmets"

DATASET_METADATA = {
    "NTHU-DDD": {
        "full_name": "National Tsing Hua University Drowsy Driver Dataset",
        "description": "Video/Image sequences of drivers wearing sunglasses, facial masks, and nodding heads",
        "official_url": "http://cv.cs.nthu.edu.tw/php/CallForPaper/case.php?id=12",
        "kaggle_mirror": "https://www.kaggle.com/datasets/robinreni/nthu-driver-drowsiness-detection",
        "kaggle_cmd": "kaggle datasets download -d robinreni/nthu-driver-drowsiness-detection",
        "target_dir": str(NTHU_DIR)
    },
    "Roboflow-Weapons": {
        "full_name": "Knife & Weapon Detection Dataset (Roboflow Universe)",
        "description": ">4,200 labeled images of knives, blades, and dangerous threatening objects",
        "official_url": "https://universe.roboflow.com/roboflow-universe-projects/knife-detection",
        "roboflow_workspace": "roboflow-universe-projects",
        "roboflow_project": "knife-detection",
        "target_dir": str(WEAPONS_DIR)
    },
    "Local-Helmets-TPHCM": {
        "full_name": "Vietnam Two-Wheel Ride-Hailing Helmet & Passenger Dataset",
        "description": "Field dataset collected in Ho Chi Minh City streets with various helmets and masks",
        "target_dir": str(LOCAL_HELMETS_DIR)
    }
}


def get_vision_dirs() -> Dict[str, Path]:
    """Trả về từ điển đường dẫn các thư mục lưu trữ dữ liệu thị giác."""
    dirs = {
        "raw_vision": RAW_VISION_DIR,
        "nthu": NTHU_DIR,
        "weapons": WEAPONS_DIR,
        "local_helmets": LOCAL_HELMETS_DIR
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def check_dataset_status() -> Dict[str, Dict]:
    """Kiểm tra số lượng tệp ảnh hiện có trong từng thư mục dữ liệu thị giác.

    Returns:
        Dict chứa thống kê số lượng file ảnh, dung lượng (MB) của từng nguồn.
    """
    dirs = get_vision_dirs()
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    status = {}

    for name, path in [("NTHU-DDD", dirs["nthu"]), 
                       ("Roboflow-Weapons", dirs["weapons"]), 
                       ("Local-Helmets", dirs["local_helmets"])]:
        images = [f for f in path.rglob("*") if f.suffix.lower() in valid_extensions]
        total_bytes = sum(f.stat().st_size for f in images)
        status[name] = {
            "path": str(path),
            "image_count": len(images),
            "size_mb": round(total_bytes / (1024 * 1024), 2),
            "is_ready": len(images) > 0
        }

    return status


def generate_sample_vision_data(num_samples_per_class: int = 10) -> Dict[str, int]:
    """Tạo bộ dữ liệu ảnh mẫu mô phỏng phục vụ kiểm thử đơn vị và chạy CI/CD pipeline.

    Tạo ra 3 nhóm ảnh mẫu tương ứng với 3 nhãn phân loại:
    - Normal (0): Nón bảo hiểm thẳng, không có hung khí.
    - Drowsiness (1): Nón bảo hiểm nghiêng gục > 25 độ.
    - Assault (2): Người ngồi sau vươn tay / hung khí sắc nhọn.

    Args:
        num_samples_per_class: Số lượng ảnh mẫu sinh ra cho mỗi lớp.

    Returns:
        Dict số lượng ảnh mẫu đã được tạo cho từng thư mục.
    """
    dirs = get_vision_dirs()
    created_counts = {"nthu_normal": 0, "nthu_drowsy": 0, "weapons": 0, "local": 0}

    # 1. Sinh ảnh mẫu NTHU Normal & Drowsy
    for i in range(num_samples_per_class):
        # Normal: Mũ bảo hiểm thẳng (oval ở giữa)
        img_norm = Image.new("RGB", (160, 160), color=(180, 200, 220))
        draw_norm = ImageDraw.Draw(img_norm)
        # Nón bảo hiểm tròn thẳng
        draw_norm.ellipse([50, 40, 110, 100], fill=(30, 80, 150), outline=(10, 30, 80), width=3)
        # Kính râm đen che mắt
        draw_norm.rectangle([60, 70, 100, 82], fill=(20, 20, 20))
        # Khẩu trang xanh y tế
        draw_norm.rectangle([55, 88, 105, 115], fill=(135, 206, 235), outline=(100, 150, 200))
        img_norm.save(dirs["nthu"] / f"sample_normal_{i:03d}.jpg")
        created_counts["nthu_normal"] += 1

        # Drowsy: Mũ bảo hiểm cúi gục (chúc xuống dưới và ngả nghiêng)
        img_drowsy = Image.new("RGB", (160, 160), color=(160, 180, 200))
        draw_drowsy = ImageDraw.Draw(img_drowsy)
        # Nón bảo hiểm chúc thấp hẳn xuống toạ độ y=80-140
        draw_drowsy.ellipse([45, 80, 115, 140], fill=(30, 80, 150), outline=(10, 30, 80), width=3)
        # Kính chắn gió nón bảo hiểm cúi gập
        draw_drowsy.arc([45, 80, 115, 140], start=30, end=150, fill=(10, 10, 10), width=4)
        img_drowsy.save(dirs["nthu"] / f"sample_drowsy_{i:03d}.jpg")
        created_counts["nthu_drowsy"] += 1

    # 2. Sinh ảnh mẫu Roboflow Weapon / Threat
    for i in range(num_samples_per_class):
        img_weapon = Image.new("RGB", (160, 160), color=(100, 100, 110))
        draw_weapon = ImageDraw.Draw(img_weapon)
        # Hình mẫu dao nhọn kim loại
        draw_weapon.polygon([(40, 130), (120, 50), (135, 40), (125, 65)], fill=(220, 220, 230), outline=(80, 80, 80))
        # Cán dao đen
        draw_weapon.rectangle([30, 120, 50, 145], fill=(20, 20, 20))
        img_weapon.save(dirs["weapons"] / f"sample_weapon_{i:03d}.jpg")
        created_counts["weapons"] += 1

    # 3. Sinh ảnh mẫu Local Vietnam Street
    for i in range(num_samples_per_class):
        img_local = Image.new("RGB", (160, 160), color=(210, 215, 210))
        draw_local = ImageDraw.Draw(img_local)
        # Nón Grab xanh lá cây đặc trưng xe ôm công nghệ Việt Nam
        draw_local.ellipse([50, 45, 110, 105], fill=(0, 150, 60), outline=(0, 90, 30), width=3)
        draw_local.rectangle([58, 85, 102, 115], fill=(240, 240, 240))  # Khẩu trang vải
        img_local.save(dirs["local_helmets"] / f"sample_vn_driver_{i:03d}.jpg")
        created_counts["local"] += 1

    return created_counts


def print_download_guide():
    """In bản hướng dẫn chi tiết cách tải dữ liệu từ các nguồn mở chính thức."""
    print("=" * 80)
    print("HƯỚNG DẪN TẢI DỮ LIỆU THỊ GIÁC (COMPUTER VISION DATASET INGESTION)")
    print("Phụ trách: Nguyễn Duy Khang (Computer Vision Lead)")
    print("=" * 80)
    for name, meta in DATASET_METADATA.items():
        print(f"\n[+] Nguồn: {name} - {meta['full_name']}")
        print(f"    Mô tả: {meta['description']}")
        if "kaggle_cmd" in meta:
            print(f"    Lệnh Kaggle CLI: {meta['kaggle_cmd']}")
            print(f"    Thư mục đích:    {meta['target_dir']}")
        elif "official_url" in meta:
            print(f"    Liên kết tải:    {meta['official_url']}")
            print(f"    Thư mục đích:    {meta['target_dir']}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_download_guide()
    status = check_dataset_status()
    print("\n[Trạng thái thư mục hiện tại]:")
    print(json.dumps(status, indent=2))
    
    # Tạo mẫu thử nghiệm nếu chưa có ảnh
    total_imgs = sum(s["image_count"] for s in status.values())
    if total_imgs == 0:
        print("\n[!] Chưa có ảnh trong thư mục. Đang tạo dữ liệu mẫu thử nghiệm...")
        created = generate_sample_vision_data(num_samples_per_class=15)
        print(f"[+] Đã tạo thành công dữ liệu mẫu kiểm thử: {created}")
        print("\n[Trạng thái sau khi tạo dữ liệu mẫu]:")
        print(json.dumps(check_dataset_status(), indent=2))
