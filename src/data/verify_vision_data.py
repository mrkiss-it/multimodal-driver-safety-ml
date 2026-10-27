"""
Module: Vision Data Quality Verification and Statistical Profiling
Author: Nguyễn Duy Khang (MSSV: 26410055) - Computer Vision Lead
Course: Máy học nâng cao (Advanced Machine Learning) - UIT
Lecturer: ThS./TS. Đặng Việt Dũng
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_VISION_DIR = PROJECT_ROOT / "data" / "raw" / "vision"
REPORTS_DIR = PROJECT_ROOT / "reports"


def verify_image_file(image_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[int]]:
    """Kiểm tra tính toàn vẹn của một tệp ảnh.

    Returns:
        (is_valid, error_msg, (width, height), channels)
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Kiểm tra tính toàn vẹn file
        
        # Mở lại để lấy kích thước và kênh màu
        with Image.open(image_path) as img:
            w, h = img.size
            channels = len(img.getbands())
            return True, None, (w, h), channels
    except Exception as e:
        return False, str(e), None, None


def profile_directory(dir_path: Path) -> Dict[str, Any]:
    """Phân tích thống kê toàn diện các tệp ảnh trong một thư mục."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in dir_path.rglob("*") if f.suffix.lower() in valid_exts]

    if not image_files:
        return {
            "total_files": 0,
            "corrupt_files": 0,
            "valid_files": 0,
            "resolutions": {},
            "channels": {},
            "avg_width": 0,
            "avg_height": 0,
            "avg_size_kb": 0
        }

    valid_count = 0
    corrupt_count = 0
    widths = []
    heights = []
    sizes_kb = []
    channel_counts = {}
    corrupt_list = []

    for img_p in image_files:
        size_kb = img_p.stat().st_size / 1024.0
        sizes_kb.append(size_kb)

        is_valid, err, dim, ch = verify_image_file(img_p)
        if is_valid and dim is not None:
            valid_count += 1
            widths.append(dim[0])
            heights.append(dim[1])
            ch_str = f"{ch}-channel"
            channel_counts[ch_str] = channel_counts.get(ch_str, 0) + 1
        else:
            corrupt_count += 1
            corrupt_list.append({"file": str(img_p.name), "error": err})

    return {
        "total_files": len(image_files),
        "valid_files": valid_count,
        "corrupt_files": corrupt_count,
        "corrupt_details": corrupt_list,
        "channel_distribution": channel_counts,
        "avg_width": round(sum(widths) / len(widths), 1) if widths else 0,
        "avg_height": round(sum(heights) / len(heights), 1) if heights else 0,
        "min_resolution": f"{min(widths)}x{min(heights)}" if widths else "N/A",
        "max_resolution": f"{max(widths)}x{max(heights)}" if widths else "N/A",
        "avg_size_kb": round(sum(sizes_kb) / len(sizes_kb), 2) if sizes_kb else 0
    }


def run_full_verification() -> Dict[str, Any]:
    """Chạy toàn bộ quy trình kiểm định và phân tích chất lượng dữ liệu ảnh."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "verified_by": "Nguyễn Duy Khang (Computer Vision Lead)",
        "subsystems": {}
    }

    subdirs = {
        "nthu_ddd": RAW_VISION_DIR / "drowsiness_nthu",
        "roboflow_weapons": RAW_VISION_DIR / "weapons_roboflow",
        "local_vietnam_helmets": RAW_VISION_DIR / "local_vietnam_helmets"
    }

    total_valid = 0
    total_corrupt = 0

    for key, path in subdirs.items():
        if path.exists():
            prof = profile_directory(path)
            report["subsystems"][key] = prof
            total_valid += prof["valid_files"]
            total_corrupt += prof["corrupt_files"]

    report["overall_summary"] = {
        "total_images_scanned": total_valid + total_corrupt,
        "total_valid": total_valid,
        "total_corrupt": total_corrupt,
        "integrity_rate": f"{(total_valid / (total_valid + total_corrupt) * 100):.2f}%" if (total_valid + total_corrupt) > 0 else "0.0%"
    }

    # Xuất file JSON
    out_file = REPORTS_DIR / "vision_data_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    rep = run_full_verification()
    print("=" * 80)
    print("KẾT QUẢ KIỂM ĐỊNH TÍNH TOÀN VẸN VÀ PHÂN BỐ DỮ LIỆU THỊ GIÁC (TUẦN 2)")
    print(f"Người kiểm định: {rep['verified_by']}")
    print("=" * 80)
    for sub, data in rep["subsystems"].items():
        print(f"\n[+] Phân hệ: {sub.upper()}")
        print(f"    Tổng số ảnh:     {data['total_files']}")
        print(f"    Ảnh hợp lệ:      {data['valid_files']}")
        print(f"    Ảnh lỗi/hỏng:    {data['corrupt_files']}")
        print(f"    Kích thước TB:   {data['avg_width']} x {data['avg_height']} px")
        print(f"    Kênh màu:        {data['channel_distribution']}")
        print(f"    Dung lượng TB:   {data['avg_size_kb']} KB")
    print("-" * 80)
    print(f"Tổng kết: {rep['overall_summary']}")
    print(f"Báo cáo chi tiết đã được xuất ra: {REPORTS_DIR / 'vision_data_summary.json'}")
    print("=" * 80)
