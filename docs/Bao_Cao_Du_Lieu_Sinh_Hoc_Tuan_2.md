# BÁO CÁO DỮ LIỆU SINH TRẮC HỌC (WESAD) – TUẦN 2
## HỌC PHẦN: MÁY HỌC NÂNG CAO
**Người thực hiện:** Phạm Thành Trung (MSSV: 26410141) – Bio-Signal & Time-Series Specialist
**Giảng viên:** ThS./TS. Đặng Việt Dũng
**Thời gian:** Tuần 2 (27/10/2026 – 01/11/2026)

---

> [!IMPORTANT]
> **Trạng thái dữ liệu:** Bộ WESAD **thật** (vài GB) chưa được tải về máy; thư mục `data/raw/bio/WESAD/` hiện **trống**. Toàn bộ số liệu trong báo cáo này đến từ **dữ liệu tổng hợp (synthetic)** có cấu trúc giống hệt WESAD, dùng để phát triển và kiểm thử pipeline. Các con số đó **chứng minh mã chạy đúng, không chứng minh hiệu năng trên người thật**. Khi có dữ liệu thật, chỉ cần chạy lại `python src/data/parse_wesad.py --source real`.

---

## 1. Nhiệm vụ Tuần 2 và mức độ hoàn thành

| Đầu việc (theo kế hoạch) | Kết quả | Trạng thái |
| :--- | :--- | :---: |
| Tải bộ dữ liệu WESAD | Viết bộ kiểm tra/hướng dẫn tải; **chưa tải dữ liệu thật** (cần tải thủ công) | ⚠️ Một phần |
| Script đọc `.pkl` của 15 đối tượng, trích PPG và đồng bộ nhãn | `src/data/parse_wesad.py` – hoàn chỉnh, đã kiểm thử | ✅ |
| Phát hiện đoạn mất tín hiệu / nhiễu | Cờ `is_dropout`, `is_motion`, `is_valid` theo từng giây | ✅ |
| Xuất CSV nhịp tim theo từng đối tượng | `data/raw/bio/hr_csv*/S*.csv` (không commit lên Git) | ✅ (trên dữ liệu synthetic) |

## 2. Về bộ dữ liệu WESAD

* **Nguồn:** Schmidt et al., *Introducing WESAD*, ACM ICMI 2018, pp. 400–408.
* **Đối tượng:** 15 người dùng được (S2–S11, S13–S17; **không có S1 và S12**).
* **Tín hiệu cổ tay (Empatica E4)** dùng cho dự án: BVP 64 Hz, ACC 32 Hz, EDA 4 Hz, TEMP 4 Hz. Nhãn lưu ở **700 Hz** (theo đồng hồ thiết bị đeo ngực).
* **Mã nhãn:** 0 = chuyển tiếp, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5–7 = bỏ qua.
* **Lưu ý quan trọng:** file `.pkl` **không chứa sẵn nhịp tim**. Nhịp tim phải được **suy ra từ BVP** bằng phát hiện đỉnh xung – đây là phần xử lý chính của `parse_wesad.py`.

## 3. Quy trình xử lý (pipeline)

```
S*.pkl → BVP(64Hz) → lọc Butterworth 0.5–4 Hz → phát hiện đỉnh → IBI → HR (BPM) → nội suy 1 Hz
       → ACC(32Hz) → |ACC| → độ lệch chuẩn / giây → cờ nhiễu chuyển động
       → label(700Hz) → hạ về 1 Hz (giây lẫn nhãn = chuyển tiếp)
       → ghép: hr_bpm, label, is_dropout, is_motion, is_valid  → CSV
```

Quyết định thiết kế đáng chú ý:
* **Không điền bừa vào chỗ trống:** giây nào không có IBI hợp lệ trong vòng 2,5 s thì `hr_bpm = NaN`, thay vì nội suy qua đoạn mất tín hiệu.
* **IBI ngoài khoảng 40–200 BPM bị loại.**
* **Vùng đệm 1 s** quanh đoạn nhiễu cũng bị coi là nghi ngờ (bộ lọc dội ở mép đoạn mất tín hiệu).
* Ngưỡng mất tín hiệu (`std BVP < 1e-3`) và ngưỡng chuyển động (`std |ACC| > 10`) là **giá trị ban đầu**, cần hiệu chỉnh lại trên dữ liệu thật.

## 4. Gán nhãn sang các lớp của đồ án (giả định PROXY)

WESAD ghi nhận stress/giải trí/thiền, **không** ghi nhận ngủ gật hay bị tấn công. Nhóm dùng ánh xạ thay thế sau và cần nêu rõ đây là **giả định mô hình hóa**:

| Nhãn WESAD | Lớp đồ án | Lý do |
| :--- | :---: | :--- |
| baseline, amusement | 0 – Normal | Trạng thái nền, nhịp tim bình thường |
| meditation | 1 – Drowsy *(proxy)* | Nhịp tim thấp, thư giãn – chỉ **gần giống** mẫu nhịp tim khi ngủ gật |
| stress | 2 – Assault *(proxy)* | Nhịp tim cao do kích thích giao cảm – chỉ **gần giống** phản ứng hoảng sợ |
| transition / 5–7 | bỏ (−1) | Không dùng để huấn luyện |

> [!WARNING]
> Nhịp tim lúc thiền không đồng nghĩa với nhịp tim lúc buồn ngủ khi đang lái xe, và stress tâm lý trong phòng thí nghiệm khác với hoảng sợ khi bị tấn công. Hạn chế này nên được nêu trong phần thảo luận của báo cáo cuối kỳ; bộ dữ liệu thực tế tại TP.HCM (Tuần 7) là cách duy nhất để kiểm chứng.

## 5. Kết quả kiểm định trên dữ liệu SYNTHETIC

Mỗi đối tượng tổng hợp có 620 giây, gồm 4 giai đoạn (baseline, stress, amusement, meditation), cộng 2 đoạn mất tín hiệu (6 s) và 2 đoạn nhiễu chuyển động (8 s) được chèn có chủ đích. Vì **biết trước nhịp tim thật**, có thể đo sai số của bộ trích xuất (kiểm tra trên 4 đối tượng S2, S5, S10, S17):

| Chỉ số | Kết quả |
| :--- | :---: |
| Sai số tuyệt đối trung bình (MAE) của HR | **0,55 BPM** |
| Phân vị 95 của sai số | 1,37 BPM |
| Sai số lớn nhất | 2,63 BPM |
| Đoạn mất tín hiệu được gắn cờ | **48 / 48 giây** |
| Đoạn nhiễu chuyển động được gắn cờ | **64 / 64 giây** |
| Tỷ lệ giây hợp lệ (15 đối tượng) | 94,19 % (8.760 / 9.300 giây) |

**Cách đọc các con số này một cách thận trọng:**
* Tín hiệu tổng hợp **sạch hơn dữ liệu thật rất nhiều** (không có dạng sóng bất thường, thay đổi biên độ, ngón tay lỏng, v.v.). MAE 0,55 BPM là cận **lạc quan**; trên WESAD thật sai số sẽ lớn hơn.
* `valid_rate` bằng nhau ở mọi đối tượng (0,9419) vì mỗi người được chèn cùng số đoạn nhiễu – đây là đặc điểm của bộ sinh, không phải kết quả đo.
* Quan hệ nhịp tim **stress > baseline > meditation** (ví dụ S2: 100,9 > 73,2 > 64,7 BPM) là **tham số được cài sẵn** trong bộ sinh dữ liệu, **không phải phát hiện khoa học**.

## 6. Kiểm thử tự động

`tests/test_bio_data.py` – **17 test, đạt 17/17**; toàn bộ bộ test của dự án: **22/22**.
Gồm: danh sách 15 đối tượng (không S1, S12), cấu trúc pickle, tính tái lập theo seed, lọc băng thông, chuyển đỉnh → HR (kể cả đoạn trống và quá ít đỉnh), hạ mẫu nhãn 700 → 1 Hz, ánh xạ nhãn, độ chính xác HR so với giá trị thật, gắn cờ đúng đoạn nhiễu, và ghi file đầu ra.
Mọi test ghi vào thư mục tạm, không đụng tới `data/raw`.

## 7. Việc còn lại / rủi ro

1. **Tải WESAD thật** và chạy lại pipeline; hiệu chỉnh ngưỡng nhiễu theo dữ liệu thật. *(Việc cần người thực hiện – không thể tự động.)*
2. Báo cáo lại MAE/tỷ lệ hợp lệ **trên dữ liệu thật**; con số synthetic chỉ để tham khảo nội bộ.
3. Chuyển giao cho Tuần 3: cung cấp bảng 1 Hz (`hr_bpm`, `project_label`, `is_valid`) cho bước lọc Butterworth/cửa sổ trượt T = 30 s và cho Vũ Văn Duy tích hợp `MultimodalDataset`.
4. WESAD chỉ có 15 người → bắt buộc dùng chia theo đối tượng (Subject-Independent) để tránh rò rỉ dữ liệu.

---
*Báo cáo do Phạm Thành Trung (Bio-Signal & Time-Series Specialist) lập – Tuần 2.*
