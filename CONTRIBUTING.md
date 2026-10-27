# HƯỚNG DẪN ĐÓNG GÓP & QUY CHUẨN KỸ THUẬT DỰ ÁN
## ĐỒ ÁN MÔN HỌC: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)
### Đề tài: Khung mô hình học máy đa phương thức nhận diện trạng thái rủi ro an toàn cho tài xế xe hai bánh
**Trưởng nhóm / Phụ trách quy chuẩn:** Trần Hoàng Hôn (MSSV: 26410046)

---

Tài liệu này quy định các chuẩn mực kỹ thuật, quy trình quản lý mã nguồn Git, tiêu chuẩn viết code và quy tắc phối hợp dành cho 5 thành viên nhóm thực hiện đồ án môn **Máy học nâng cao** (Trường ĐH Công nghệ Thông tin – ĐHQG-HCM, Giảng viên: ThS./TS. Đặng Việt Dũng).

Mọi thành viên trong nhóm (*Trần Hoàng Hôn, Nguyễn Duy Khang, Phạm Thành Trung, Vũ Văn Duy, Hoàng Cao Sơn*) có trách nhiệm tuân thủ nghiêm ngặt các quy định dưới đây.

---

## 1. Mô hình Phân nhánh Git (Git Branching Strategy)

Nhóm áp dụng mô hình phân nhánh **GitFlow thu gọn**, tối ưu cho nhóm nghiên cứu AI/Data Science:

```text
  main ──────●─────────────────────────────────────────●───────────► (Bản nộp cuối kỳ / Releases)
              \                                       /
  develop ─────●──────●──────────────●───────────────●─────────────► (Nhánh tích hợp chính)
                \    /                \             /
  feature/* ─────●──● (Khang/Trung)    ●───●───────● (Duy/Sơn)
```

### Quy định các nhánh:
1. **`main` (Nhánh chính thức / Production):**
   * Chỉ chứa mã nguồn ổn định, đã kiểm thử đạt chuẩn (Macro-F1 $\ge 90\%$) và tài liệu báo cáo chính thức.
   * **Tuyệt đối không commit trực tiếp vào `main`.** Chỉ merge từ `develop` qua Pull Request được Trưởng nhóm phê duyệt.
2. **`develop` (Nhánh phát triển / Staging):**
   * Là nhánh làm việc trung tâm để tích hợp các tính năng mới của các thành viên.
   * Mọi Pull Request tính năng đều phải trỏ vào `develop`.
3. **`feature/<ten-tinh-nang>` (Nhánh tính năng cá nhân):**
   * Nhánh làm việc tạm thời khi phát triển một module mới.
   * Đặt tên theo cú pháp:
     * `feature/vision-mobilenetv3` (Khang)
     * `feature/hrv-ppg-features` (Trung)
     * `feature/focal-loss-tuning` (Duy)
     * `feature/gradcam-shap-xai` (Sơn)
     * `feature/cross-attention-fusion` (Hôn)
4. **`fix/<ten-loi>` (Nhánh sửa lỗi):**
   * Dùng để sửa lỗi nhanh (ví dụ: `fix/data-loader-leakage`, `fix/nan-loss`).

---

## 2. Quy chuẩn Thông điệp Commit (Conventional Commits)

Mỗi commit phải giải thích ngắn gọn, rõ ràng nội dung thay đổi bằng tiếng Anh hoặc tiếng Việt chuẩn theo quy tắc **Conventional Commits**:

```text
<loại-commit>(<phạm-vi-thay-đổi>): <mô tả ngắn bằng động từ thì hiện tại>
```

### Các loại commit chuẩn (`type`):
* `feat`: Thêm tính năng mới hoặc mô hình mới (ví dụ: `feat(models): implement CrossAttentionFusion layer`).
* `fix`: Sửa lỗi trong code hoặc pipeline (ví dụ: `fix(features): correct RMSSD denominator index in hrv.py`).
* `docs`: Cập nhật tài liệu, README, báo cáo Word/PDF (ví dụ: `docs(plan): add week 2 action items for lead`).
* `test`: Thêm hoặc sửa các bài kiểm thử unit test (ví dụ: `test(losses): add test cases for CostSensitiveBayesLoss`).
* `refactor`: Tái cấu trúc mã nguồn không làm thay đổi chức năng (ví dụ: `refactor(data): simplify batch tensor stacking`).
* `perf`: Cải thiện hiệu năng xử lý, tốc độ suy luận hoặc tối ưu RAM (ví dụ: `perf(vision): enable mixed precision fp16 inference`).
* `chore`: Cập nhật dependencies, cấu hình công cụ, file `.gitignore` (ví dụ: `chore(deps): add albumentations to requirements`).

---

## 3. Quy chuẩn Viết Code & Đảm bảo Chất lượng (Code Quality Standards)

1. **Chuẩn phong cách Python (PEP 8):**
   * Sử dụng thụt đầu dòng 4 khoảng trắng (spaces), không dùng tab.
   * Giới hạn độ dài dòng: tối đa 100 ký tự.
   * Đặt tên biến/hàm theo `snake_case` (ví dụ: `compute_hrv_features`).
   * Đặt tên lớp theo `CamelCase` (ví dụ: `MultiClassFocalLoss`).
   * Đặt tên hằng số theo `UPPER_SNAKE_CASE` (ví dụ: `DEFAULT_BASELINE_HR`).
2. **Gợi ý kiểu dữ liệu (Type Annotations):**
   * Mọi hàm và phương thức bắt buộc phải có Type Annotations cho tham số và giá trị trả về:
     ```python
     from typing import Dict, List, Optional
     import torch

     def predict_step(
         image: torch.Tensor,
         bio_sequence: torch.Tensor,
         sos_state: int
     ) -> Dict[str, float]:
         ...
     ```
3. **Docstring chuẩn Google Style:**
   * Mỗi module, class và hàm công khai phải có docstring giải thích mục đích, tham số (*Args*), kết quả (*Returns*), và ngoại lệ (*Raises*):
     ```python
     def compute_hrv_features(hr_sequence: List[float], baseline_hr: float = 80.0) -> Dict[str, float]:
         """Trích xuất 8 đặc trưng thống kê miền thời gian từ chuỗi nhịp tim.

         Args:
             hr_sequence: Danh sách các giá trị nhịp tim (BPM) theo chuỗi thời gian.
             baseline_hr: Nhịp tim nền khi lái xe tỉnh táo (mặc định 80 BPM).

         Returns:
             Dict chứa các chỉ số: mean_hr, std_hr (SDNN), rmssd, delta_hr_pct, pnn50_approx.
         """
     ```
4. **Quy tắc Quản lý Dữ liệu Lớn & Trọng số Mô hình:**
   * **NGHIÊM CẤM:** Không commit các file dữ liệu thô (`.mp4`, `.avi`, `.pkl`, `.csv` kích thước $> 50$MB) hoặc file trọng số mô hình (`.pth`, `.onnx` kích thước $> 100$MB) trực tiếp vào Git.
   * Mọi dữ liệu lớn phải được lưu trên Google Drive nhóm hoặc Kaggle Datasets; thư mục cục bộ phải tuân thủ `.gitignore` và chỉ đẩy file rỗng `.gitkeep` lên remote.

---

## 4. Quy trình Tạo Pull Request & Kiểm thử Chéo (PR Workflow)

Trước khi mã nguồn được tích hợp vào nhánh `develop`, thành viên phải thực hiện đủ 5 bước sau:

1. **Bước 1: Cập nhật nhánh cục bộ:**
   ```powershell
   git checkout develop
   git pull origin develop
   git checkout feature/<ten-tinh-nang>
   git rebase develop
   ```
2. **Bước 2: Chạy kiểm thử tự động toàn diện:**
   ```powershell
   python -m pytest tests/ -v
   ```
   *Yêu cầu:* Toàn bộ các bài kiểm thử phải pass 100%. Không có lỗi cú pháp hoặc cảnh báo nghiêm trọng.
3. **Bước 3: Đẩy nhánh lên GitHub:**
   ```powershell
   git push origin feature/<ten-tinh-nang>
   ```
4. **Bước 4: Mở Pull Request trên GitHub:**
   * Điền đầy đủ thông tin theo mẫu PR Template (`.github/PULL_REQUEST_TEMPLATE.md`).
   * Gắn nhãn (*Labels*) tương ứng: `vision`, `bio-signals`, `modeling`, `data`, `docs`.
   * Gán người Review (*Reviewers*): Chọn ít nhất **01 thành viên liên quan** và **Trưởng nhóm (Trần Hoàng Hôn)**.
5. **Bước 5: Thảo luận & Hợp nhất (Merge):**
   * Sau khi nhận được ít nhất 1 phê duyệt (Approval) và giải quyết xong các góp ý, Trưởng nhóm sẽ merge PR vào `develop` theo phương thức *Squash and merge* hoặc *Rebase and merge* để giữ lịch sử commit gọn gàng.

---

## 5. Hướng dẫn Thiết lập Môi trường Phát triển Cục bộ

### Cách 1: Sử dụng Python Virtual Environment (venv)
```powershell
# 1. Di chuyển vào thư mục đồ án
cd "D:\Trí tuệ nhân tạo\HK3\Máy học nâng cao\Đồ án"

# 2. Tạo môi trường ảo venv
python -m venv .venv

# 3. Kích hoạt môi trường (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 4. Cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install -r requirements.txt
```

### Cách 2: Sử dụng Conda
```powershell
conda create -n driver-safety-ml python=3.10 -y
conda activate driver-safety-ml
pip install -r requirements.txt
```

---

*Tài liệu được soạn thảo và ban hành bởi Trưởng nhóm Trần Hoàng Hôn phục vụ công tác điều phối đồ án môn Máy học nâng cao.*
