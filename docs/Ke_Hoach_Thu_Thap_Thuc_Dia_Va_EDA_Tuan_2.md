# KẾ HOẠCH THU THẬP DỮ LIỆU THỰC ĐỊA TP.HCM & BÁO CÁO PHÂN TÍCH KHÁM PHÁ (EDA) – TUẦN 2
## HỌC PHẦN: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)
### Đề tài: Khung mô hình học máy đa phương thức nhận diện trạng thái rủi ro an toàn cho tài xế xe hai bánh
**Người thực hiện:** Hoàng Cao Sơn (MSSV: 25730061) – Data Engineering & Explainable AI (XAI)  
**Giảng viên phụ trách:** ThS./TS. Đặng Việt Dũng  
**Thời gian báo cáo:** Tuần 2 (27/10/2026 – 01/11/2026)

---

> [!IMPORTANT]
> **Ghi chú Minh bạch Nguồn gốc Dữ liệu (Data Provenance Notice):**
> - Thư mục dữ liệu thực địa thật `data/raw/local_samples/` hiện có **0 ảnh chụp thực tế ngoài phố** (chưa thực hiện phiên quay hiện trường).
> - Để hoàn thiện và kiểm thử tự động toàn bộ luồng Data Engineering ngay trong Tuần 2, module `src/data/local_samples.py` đã sinh **50 mẫu thử nghiệm tổng hợp có cấu trúc (`is_synthetic=True`)** trong thư mục riêng `data/raw/local_samples_synthetic/` kèm bảng kê `manifest.csv` đầy đủ 13 trường metadata.
> - Toàn bộ biểu đồ trong Notebook `notebooks/01_eda_and_data_preprocessing.ipynb` và `reports/figures/week2_eda_*.png` đều gắn nhãn `[SYNTHETIC / PILOT DRY-RUN]` rõ ràng để không bao giờ bị nhầm lẫn với số liệu thực nghiệm ngoài đường phố TP.HCM (sẽ thu thập và đánh giá OOD ở giai đoạn tiếp theo).

---

## 1. Tổng quan Nhiệm vụ Tuần 2 & Mức độ Hoàn thành

| Đầu việc theo Kế hoạch Tuần 2 | Sản phẩm bàn giao | Trạng thái |
| :--- | :--- | :---: |
| 1. Lập kế hoạch quay video & thu thập dữ liệu thực nghiệm mũ bảo hiểm tại TP.HCM | Quy trình an toàn giao thông, đạo đức dữ liệu, cấu hình gá lắp camera & đồng bộ Mi Band (Mục 2) | ✅ Hoàn thành |
| 2. Khởi tạo Notebook EDA đầu tiên trực quan hóa phân bố mẫu đa nguồn | `notebooks/01_eda_and_data_preprocessing.ipynb` (đã thực thi 100%) + 3 biểu đồ `reports/figures/week2_eda_*.png` | ✅ Hoàn thành |
| 3. Chuẩn bị 50 ảnh mẫu nón bảo hiểm & kiểm định tự động pipeline dữ liệu địa phương | `src/data/local_samples.py`, `reports/local_samples_pilot_summary.json`, `tests/test_local_samples_and_eda.py` | ✅ Hoàn thành *(trên 50 mẫu synthetic pilot)* |

---

## 2. Kế hoạch Quay Video Thực nghiệm Ngoài Phố & Bãi Tập Khép kín tại TP.HCM

Nhằm thu thập tập kiểm định thực tế địa phương (*Local Out-of-Distribution Validation Set*, mục tiêu $1.200$ khung hình đồng bộ nhịp tim phục vụ kiểm thử thực địa Tuần 7) đảm bảo **tuyệt đối an toàn tính mạng, tuân thủ Luật Giao thông Đường bộ và quyền riêng tư**, nhóm thiết lập quy trình 2 giao thức tách biệt:

### 2.1. Phân tách Giao thức An toàn (Safety-First Dual Protocol)

| Giao thức | Lớp dữ liệu thu thập | Địa điểm thực hiện tại TP.HCM | Quy tắc an toàn bắt buộc |
| :--- | :--- | :--- | :--- |
| **Giao thức A (Lưu thông thực tế)** | **Lớp 0 – Normal Driving** (lái thẳng, liếc gương chiếu hậu, cúi nhìn đồng hồ $< 1.2\text{s}$) | Tuyến Làng Đại học ĐHQG-HCM (TP. Thủ Đức), Xa lộ Võ Nguyên Giáp, đường dưới tán cây Quận Bình Thạnh | Tài xế hoàn toàn tỉnh táo, tốc độ $\le 40\text{ km/h}$; camera gắn cố định vào chân gương chiếu hậu trái, không thao tác tay khi xe chạy. |
| **Giao thức B (Diễn tập tĩnh khép kín)** | **Lớp 1 – Drowsiness** (gục mũ $\theta \ge 25^\circ \ge 1.5\text{s}$) & **Lớp 2 – Assault** (chồm siết cổ / giơ vật mô hình) | Bãi đỗ xe khép kín trong khuôn viên Trường ĐH CNTT – ĐHQG-HCM (xe dựng chân chống giữa, **tắt máy**) | **Nghiêm cấm** diễn tập ngủ gật hoặc khống chế khi xe đang lăn bánh; **nghiêm cấm** dùng dao/hung khí kim loại thật (chỉ dùng mô hình bìa cứng/cao su mềm). |

### 2.2. Cấu hình Thiết bị & Đồng bộ Thời gian Đa phương thức
1. **Nhánh Thị giác:** Action Camera góc rộng $120^\circ$ ($1080\text{p} @ 30\text{ FPS}$) gắn tại gốc chân gương chiếu hậu bên trái, hướng chéo $35^\circ$ về phía sau để bao quát đồng thời **mũ bảo hiểm tài xế** và **vùng vai/tay người ngồi sau**.
2. **Nhánh Sinh trắc học:** Thiết bị đeo cổ tay **Xiaomi Smart Band** phát nhịp tim thời gian thực qua giao thức BLE (*Heart Rate Service UUID `0x180D`*) về điện thoại ghi log CSV 1 Hz.
3. **Đồng bộ đồng hồ (Clock Sync):** Điện thoại ghi log BLE và camera đều đồng bộ giờ chuẩn NTP trước mỗi phiên quay; đầu mỗi video hiển thị bảng mã phiên (`SESSION_ID` + Unix Timestamp) trong 2 giây để căn chỉnh chính xác từng khung hình với dòng thời gian nhịp tim 1 Hz.
4. **Bảo mật & Ẩn danh hóa (Privacy & De-identification):** Toàn bộ tình nguyện viên (10 đối tượng `VN_S01`..`VN_S10`) ký phiếu đồng thuận tham gia nghiên cứu học thuật; mọi biển số xe và khuôn mặt người đi đường ngẫu nhiên lọt vào hậu cảnh được làm mờ Gaussian trước khi đưa vào `data/processed/`.

### 2.3. Ma trận Phân tầng Mục tiêu Tập Thực nghiệm TP.HCM ($1.200$ mẫu, 10 đối tượng)
* **5 chủng loại nón bảo hiểm phổ biến tại Việt Nam** (mỗi loại $240$ mẫu):
  1. `grab_green`: Nón bảo hiểm xanh lá xe ôm công nghệ Grab (kèm khẩu trang vải/y tế).
  2. `be_yellow`: Nón bảo hiểm vàng hãng Be (kèm kính râm chống chói).
  3. `shopeefood_orange`: Nón bảo hiểm cam giao hàng công nghệ.
  4. `three_quarter_visor`: Nón $3/4$ có kính chắn gió tối màu che kín toàn mặt.
  5. `half_helmet_mask`: Nón nửa đầu phổ thông kết hợp kính cận/râm và khẩu trang kín.
* **4 bối cảnh chiếu sáng đặc trưng đô thị TP.HCM:** Nắng gắt ban ngày (`daylight_sunny`), bóng râm hàng cây cắt ngang mặt đường (`tree_shadow`), chiều chạng vạng dưới đèn cao áp (`dusk_streetlight`), và ngược sáng đèn pha ban đêm (`night_glare`).

---

## 3. Kiểm định Bộ 50 Mẫu Thử nghiệm Pilot & Kết quả Phân tích EDA Tuần 2

### 3.1. Kết quả xác thực Schema & Chất lượng 50 Mẫu Pilot (`reports/local_samples_pilot_summary.json`)
Chạy công cụ kiểm định tự động `python src/data/local_samples.py --generate-pilot --num-samples 50`:
* **Tính toàn vẹn tệp ảnh:** **50 / 50 ảnh hợp lệ (100.0%)**, chuẩn 3 kênh RGB $160 \times 160\text{ px}$, 0 tệp hỏng, 0 tệp thiếu.
* **Đầy đủ 13 cột Metadata chuẩn (`manifest.csv`):** `sample_id`, `subject_id` (10 tài xế giả lập `VN_SYN_S01..S10`), `class_label`, `class_name`, `helmet_type`, `lighting_condition`, `occlusion_type`, `head_pitch_deg_est`, `has_rear_passenger`, `is_synthetic` (`True`), `file_name`, `width`, `height`.
* **Tỷ lệ phân tầng chuẩn xác theo thiết kế bài toán:**
  * Lớp `0: Normal`: **35 mẫu (70.0%)** – góc nghiêng đầu ước lượng trung bình $10.49^\circ$ ($< 20^\circ$).
  * Lớp `1: Drowsiness`: **10 mẫu (20.0%)** – góc nghiêng gục đầu trung bình $34.34^\circ$ ($\ge 25^\circ$).
  * Lớp `2: Assault`: **5 mẫu (10.0%)** – có hành khách phía sau kèm cử chỉ vươn tay/vật mô hình.
* **Cân bằng chủng loại nón:** Đúng **10 mẫu / loại** cho cả 5 kiểu nón bảo hiểm Việt Nam.
* **Độ sáng trung bình theo môi trường (0–255):** `daylight_sunny` ($159.86$) $>$ `tree_shadow` ($116.25$) $>$ `dusk_streetlight` ($93.44$) $>$ `night_glare` ($67.09$).

### 3.2. Các Biểu đồ Xuất bản từ Notebook `01_eda_and_data_preprocessing.ipynb`
Notebook đã được thực thi hoàn chỉnh từ đầu đến cuối bằng `nbclient` (0 lỗi thực thi) và xuất 3 đồ thị chất lượng cao vào `reports/figures/`:
1. `reports/figures/week2_eda_class_distribution.png`: Đối chiếu phân bố mất cân bằng mục tiêu ($4200 : 1200 : 600$), phân bố chồng theo 5 kiểu nón của 50 mẫu pilot TP.HCM ($35 : 10 : 5$) và phân bố số giây hợp lệ trên WESAD 1 Hz theo nhãn proxy.
2. `reports/figures/week2_eda_local_helmet_grid.png`: Lưới trực quan $3 \text{ lớp} \times 5 \text{ kiểu nón bảo hiểm}$ kèm thông số góc gục đầu và điều kiện ánh sáng.
3. `reports/figures/week2_eda_bio_hr_profile.png`: Đồ thị chuỗi nhịp tim 1 Hz của đối tượng `S2` làm nổi bật vùng cờ mất tín hiệu (`is_dropout`) và cờ nhiễu chuyển động (`is_motion`), cùng biểu đồ hộp (boxplot) phân phối nhịp tim qua 15 đối tượng synthetic.

---

## 4. Kiểm thử Tự động & Bàn giao sang Tuần 3

* Bộ kiểm thử mới `tests/test_local_samples_and_eda.py` (**8/8 test cases PASSED**) kiểm tra: tỷ lệ phân tầng $70:20:10$, tính tái lập theo seed, phát hiện ảnh hỏng/thiếu, báo cáo trung thực khi thư mục ảnh thật trống (`real_field_data_collected == False`), bắt ngoại lệ tham số biên và kiểm tra toàn bộ các ô code trong `notebooks/01_eda_and_data_preprocessing.ipynb` đã được thực thi sạch sẽ không có lỗi.
* **Điểm phối hợp sang Tuần 3:**
  * Mở rộng `notebooks/01_eda_and_data_preprocessing.ipynb` để trực quan hóa ảnh trước/sau `Albumentations` (nhận từ Nguyễn Duy Khang) và phổ sóng PPG trước/sau lọc Butterworth (nhận từ Phạm Thành Trung).
  * Kiểm chứng không rò rỉ `subject_id` trên file `splits_5fold.json` (nhận từ Trần Hoàng Hôn) cùng `MultimodalDataset` (nhận từ Vũ Văn Duy).

---
*Báo cáo do Hoàng Cao Sơn (MSSV: 25730061 – Data Engineering & Explainable AI) lập và kiểm định – Tuần 2.*
