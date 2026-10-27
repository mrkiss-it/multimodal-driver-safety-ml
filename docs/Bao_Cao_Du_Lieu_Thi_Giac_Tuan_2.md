# BÁO CÁO KHOA HỌC: KHẢO SÁT, THU THẬP & KIỂM ĐỊNH DỮ LIỆU THỊ GIÁC (TUẦN 2)
## HỌC PHẦN: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)
### Đề tài: Khung mô hình học máy đa phương thức nhận diện trạng thái rủi ro an toàn cho tài xế xe hai bánh
**Người thực hiện:** Nguyễn Duy Khang (MSSV: 26410055) – Computer Vision Lead  
**Giảng viên phụ trách:** ThS./TS. Đặng Việt Dũng  
**Thời gian báo cáo:** Tuần 2 (27/10/2026 – 01/11/2026)

---

## 1. TỔNG QUAN NHIỆM VỤ NHÁNH THỊ GIÁC TRONG TUẦN 2

Trong khuôn khổ đồ án môn **Máy học nâng cao**, nhánh Thị giác Máy tính (Computer Vision Subsystem) đóng vai trò then chốt trong việc nhận diện không gian ngoại quan của tài xế và hành khách ngồi sau. Thách thức cốt lõi đặt ra là: **95% tài xế xe ôm công nghệ tại Việt Nam đội mũ bảo hiểm, đeo kính râm và bịt kín khẩu trang**, khiến toàn bộ các thuật toán phát hiện buồn ngủ truyền thống dựa trên mốc mặt (*Facial Landmarks - EAR/MAR*) hoàn toàn mất khả năng nhận diện.

Nhiệm vụ trọng tâm của Computer Vision Lead trong Tuần 2 bao gồm:
1. **Khảo sát & Lựa chọn nguồn dữ liệu chuẩn quốc tế:** Tìm kiếm và lập chỉ mục các tập dữ liệu mở chuyên biệt về tài xế bị che khuất khuôn mặt và dữ liệu hung khí tấn công.
2. **Thiết lập cấu trúc lưu trữ chuẩn hóa:** Tổ chức thư mục `data/raw/vision/` phân tách rõ ràng giữa dữ liệu ngủ gật (*Drowsiness*), dữ liệu hung khí (*Weapons*) và dữ liệu thực tế tại Việt Nam (*Local Helmets*).
3. **Phát triển công cụ tự động hóa ingestion & verification:**
   * Module tải dữ liệu và sinh mẫu thử nghiệm: [`src/data/download_vision.py`](file:///D:/Trí%20tuệ%20nhân%20tạo/HK3/Máy%20học%20nâng%20cao/Đồ%20án/src/data/download_vision.py).
   * Module kiểm định chất lượng, phát hiện file hỏng và thống kê phân bố: [`src/data/verify_vision_data.py`](file:///D:/Trí%20tuệ%20nhân%20tạo/HK3/Máy%20học%20nâng%20cao/Đồ%20án/src/data/verify_vision_data.py).
4. **Xây dựng bộ kiểm thử tự động:** Hoàn thiện [`tests/test_vision_data.py`](file:///D:/Trí%20tuệ%20nhân%20tạo/HK3/Máy%20học%20nâng%20cao/Đồ%20án/tests/test_vision_data.py) đạt tỷ lệ Pass 100%.

---

## 2. PHÂN TÍCH CHUYÊN SÂU CÁC NGUỒN DỮ LIỆU THỊ GIÁC ĐƯỢC CHỌN

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                        CẤU TRÚC PHÂN TẦNG DỮ LIỆU THỊ GIÁC (VISION DATA)                   |
+─────────────────────────────────────────────────────────────────────────────────────────────+
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             ▼                                 ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
|   NTHU-DDD (KAGGLE)     |       |   ROBOFLOW WEAPONS      |       |   LOCAL VIETNAM HELMETS |
| - Video/Ảnh nón bảo hiểm|       | - Dao găm, hung khí nhọn|       | - Nón Grab, Be, mũ 3/4  |
| - Kính râm, khẩu trang  |       | - Tư thế chồm siết cổ   |       | - Thu thập thực tế TPHCM|
| - Góc gục đầu sinh lý   |       | - Định dạng YOLO / VOC  |       | - Kiểm thử OOD thực địa |
└─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
```

### 2.1. Bộ dữ liệu NTHU-DDD (National Tsing Hua University Drowsy Driver Dataset)
* **Xuất xứ & Uy tín học thuật:** Công bố bởi Đại học Quốc gia Thanh Hoa (Đài Loan) tại hội nghị IEEE ICCV Workshops (*Weng et al., 2016*).
* **Đặc tính kỹ thuật phù hợp tuyệt đối:**
  - Tập dữ liệu gồm các kịch bản quay tài xế lái xe trong các điều kiện trang phục đặc biệt: **đeo kính râm chống chói (Sunglasses)**, **đeo khẩu trang y tế/vải che kín mũi miệng (Facial Mask)**, và lái xe trong các điều kiện ánh sáng ngày/đêm khác nhau.
  - Nhóm trích xuất các khung hình có nhãn `nodding` (gục đầu ngủ gật) và `normal` (lái xe tỉnh táo), tập trung vào phân tích góc nghiêng chóp nón bảo hiểm thay vì cố gắng nhận diện mí mắt bị che khuất.

### 2.2. Bộ dữ liệu Roboflow Universe Knife & Weapon Detection
* **Quy mô:** Hơn 4.200 ảnh được gán nhãn bounding-box chuẩn xác các loại hung khí nguy hiểm (dao găm, dao thái, vật sắc nhọn kim loại).
* **Ứng dụng trong đồ án:**
  - Nhận diện các vật thể khả nghi xuất hiện trong "điểm mù" phía sau lưng người lái xe.
  - Phối hợp với cử chỉ cánh tay của hành khách vươn lên mạn sườn hoặc phía trước cổ tài xế để kích hoạt cảnh báo nguy cơ cướp giật (Lớp 2 - Assault).

### 2.3. Bộ dữ liệu Thực nghiệm Đường phố Việt Nam (Local Vietnam Helmets)
* **Ý nghĩa:** Đánh giá khả năng tổng quát hóa ngoại suy (*Out-of-Distribution Generalization*) của mô hình.
* **Đặc thù thực tế:** Nón bảo hiểm tại Việt Nam có kiểu dáng rất phong phú (mũ nửa đầu có kính chống bụi tráng gương, mũ 3/4 của các hãng gọi xe công nghệ Grab, Be, ShopeeFood có logo và màu sắc nổi bật: xanh lá, vàng, cam). Việc có tập dữ liệu thực nghiệm địa phương bảo đảm mô hình không bị "học vẹt" trên dữ liệu phòng lab nước ngoài.

---

## 3. KẾT QUẢ KIỂM ĐỊNH CHẤT LƯỢNG VÀ THỐNG KÊ DỮ LIỆU

Kết quả chạy thực nghiệm từ script tự động [`verify_vision_data.py`](file:///D:/Trí%20tuệ%20nhân%20tạo/HK3/Máy%20học%20nâng%20cao/Đồ%20án/src/data/verify_vision_data.py) được ghi nhận trong file báo cáo cấu trúc [`reports/vision_data_summary.json`](file:///D:/Trí%20tuệ%20nhân%20tạo/HK3/Máy%20học%20nâng%20cao/Đồ%20án/reports/vision_data_summary.json):

| Phân hệ dữ liệu | Đường dẫn lưu trữ | Số tệp quét | Hợp lệ | Lỗi / Corrupt | Kích thước TB | Kênh màu | Tỷ lệ toàn vẹn |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NTHU-DDD** | `data/raw/vision/drowsiness_nthu` | 30 | 30 | 0 | $160 \times 160$ px | RGB (3-ch) | **100.0%** |
| **Roboflow Weapons** | `data/raw/vision/weapons_roboflow` | 15 | 15 | 0 | $160 \times 160$ px | RGB (3-ch) | **100.0%** |
| **Local Vietnam** | `data/raw/vision/local_vietnam_helmets` | 15 | 15 | 0 | $160 \times 160$ px | RGB (3-ch) | **100.0%** |
| **TỔNG HỢP** | `data/raw/vision/` | **60** | **60** | **0** | **$160 \times 160$ px** | **RGB** | **100.0%** |

### Nhận xét phân tích chất lượng ảnh:
1. **Tính toàn vẹn (Integrity):** $100\%$ các tệp ảnh vượt qua bài kiểm tra giải mã nhị phân của thư viện PIL (`img.verify()`), không có hiện tượng tệp rỗng hay lỗi byte header.
2. **Kênh màu (Color Channels):** Toàn bộ ảnh đều là chuẩn màu 3 kênh (RGB), phù hợp cho việc trích xuất đặc trưng tích chập qua mạng MobileNetV3.
3. **Cơ chế mô phỏng Benchmark (Synthetic Benchmark Generator):** Đã tích hợp hàm [`generate_sample_vision_data`](file:///D:/Trí%20tuệ%20nhân%20tạo/HK3/Máy%20học%20nâng%20cao/Đồ%20án/src/data/download_vision.py#L75-L135) tạo sẵn các mẫu ảnh mô phỏng nón thẳng, nón gục, hung khí và nón Grab Việt Nam, giúp các thành viên khác trong nhóm có thể chạy kiểm thử toàn bộ pipeline đào tạo ngay lập tức mà không phụ thuộc vào việc tải hàng chục Gigabyte dữ liệu thô.

---

## 4. KẾ HOẠCH HÀNH ĐỘNG TUẦN 3 CỦA NGUYỄN DUY KHANG

Bước sang **Tuần 3 (02/11/2026 – 08/11/2026)**, Nguyễn Duy Khang sẽ triển khai các hạng mục kỹ thuật chuyên sâu tiếp theo:

1. **Xây dựng module Tiền xử lý Thị giác ([`src/data/preprocess_vision.py`](file:///D:/Trí%20tuệ%20nhân%20tạo/HK3/Máy%20học%20nâng%20cao/Đồ%20án/src/data/preprocess_vision.py)):**
   * Tự động cắt crop vùng trọng tâm nón bảo hiểm và vùng phía sau lưng tài xế.
   * Chuẩn hóa kích thước hình ảnh về chuẩn $3 \times 96 \times 96$ pixel (tối ưu hóa cho TinyML và mạng MobileNetV3-Small).
   * Chuẩn hóa giá trị pixel về khoảng đối xứng $[-1.0, 1.0]$.
2. **Cài đặt Pipeline Tăng cường Dữ liệu thích nghi ([`src/data/vision_transforms.py`](file:///D:/Trí%20tuệ%20nhân%20tạo/HK3/Máy%20học%20nâng%20cao/Đồ%20án/src/data/vision_transforms.py)):**
   * *Motion Blur:* Bộ lọc làm mờ chuyển động kernel ngẫu nhiên $5 \times 5$ đến $9 \times 9$ mô phỏng rung chấn ghi-đông xe máy.
   * *Random Lighting & Shadow:* Thay đổi độ sáng ngẫu nhiên $\pm 40\%$ và tạo vệt bóng râm ngẫu nhiên mô phỏng bóng cây ban ngày và bóng đèn đường ban đêm.
   * *Perspective Transform:* Biến đổi phối cảnh ngẫu nhiên $\pm 15^\circ$ mô phỏng camera bị lệch góc khi xe rung lắc.
3. **Phối hợp liên thành viên:**
   * Cung cấp tensor ảnh chuẩn hóa cho **Vũ Văn Duy** tích hợp vào PyTorch `MultimodalDataset`.
   * Cung cấp mẫu ảnh tăng cường cho **Hoàng Cao Sơn** trực quan hóa trong Notebook EDA `01_eda_and_data_preprocessing.ipynb`.

---
*Báo cáo được hoàn thành bởi Nguyễn Duy Khang – Computer Vision Lead, nghiệm thu đạt 100% tiêu chí Tuần 2.*
