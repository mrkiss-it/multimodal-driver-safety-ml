# ĐỒ ÁN MÔN HỌC: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)
## Đề tài: Khung mô hình học máy đa phương thức nhận diện trạng thái rủi ro an toàn cho tài xế xe hai bánh (Multimodal Safety Risk Recognition for Two-Wheel Drivers)

---

### 1. Thông tin Đồ án & Nhóm tác giả
* **Cơ sở đào tạo:** Trường Đại học Công nghệ Thông tin – ĐHQG-HCM (UIT - VNU-HCM)
* **Môn học:** Máy học nâng cao
* **Giảng viên phụ trách:** ThS./TS. Đặng Việt Dũng
* **Học kỳ:** Học kỳ 3 (Năm học 2026 – 2027)
* **Nhóm sinh viên thực hiện (Nhóm 5 thành viên):**

| STT | Thành viên | MSSV | Vai trò chính | Phân công chuyên môn |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Trần Hoàng Hôn** | **26410046** | **Trưởng nhóm / ML Architect** | Thiết kế khung kiến trúc học máy tổng thể, mô hình hóa toán học, chiến lược dung hợp đa phương thức (Multimodal Fusion). |
| **2** | **Nguyễn Duy Khang** | **26410055** | **Computer Vision Lead** | Nhánh thị giác: Tiền xử lý dữ liệu ảnh, thiết kế MobileNetV3 trích xuất góc nghiêng nón bảo hiểm và nhận diện hung khí. |
| **3** | **Phạm Thành Trung** | **26410141** | **Bio-Signal Specialist** | Nhánh chuỗi thời gian: Tiền xử lý PPG/HR, trích xuất đặc trưng miền thời gian/tần số (HRV: SDNN, RMSSD, LF/HF). |
| **4** | **Vũ Văn Duy** | **26410031** | **ML Modeling & Evaluation** | Thiết kế hàm mất mát (Focal Loss, Class-Balanced Loss), quy trình kiểm định Stratified 5-Fold, ma trận nhầm lẫn và chi phí Bayes. |
| **5** | **Hoàng Cao Sơn** | **25730061** | **Data Engineering & XAI** | Xây dựng pipeline dữ liệu & Augmentation, triển khai Explainable AI (Grad-CAM và SHAP), thực nghiệm đường phố tại TP.HCM. |

---

### 2. Hồ sơ Tài liệu Đồ án (Project Documentation)

| STT | Văn kiện học thuật | Định dạng Markdown | Định dạng Word (.docx) | Mô tả tóm tắt |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Bản Thuyết minh Bài toán Khoa học** | [Xem file .md](Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.md) | [Tải file .docx](Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.docx) | Định nghĩa toán học hình thức 10 chương, hàm mất mát Focal Loss, Ma trận chi phí Bayes, 3 cơ chế dung hợp và phương pháp luận thẩm định. |
| **2** | **Kế hoạch & Lộ trình Thực hiện (8 Tuần)** | [Xem file .md](Ke_Hoach_Thuc_Hien_Do_An.md) | [Tải file .docx](Ke_Hoach_Thuc_Hien_Do_An.docx) | Lộ trình 5 giai đoạn từ 27/10 đến 20/12/2026, ma trận phân công trách nhiệm RACI chi tiết cho 5 thành viên, tiêu chuẩn nghiệm thu khoa học. |
| **3** | **Kế hoạch Step-by-Step Từng Tuần & Thành viên** | [Xem file .md](Ke_Hoach_Chi_Tiet_Tung_Tuan_Va_Thanh_Vien.md) | [Tải file .docx](Ke_Hoach_Chi_Tiet_Tung_Tuan_Va_Thanh_Vien.docx) | Kế hoạch hành động cụ thể từng bước từ Tuần 2 đến Tuần 9, deliverables, file phụ trách và tiêu chí DoD cho từng bạn (Hôn, Khang, Trung, Duy, Sơn). |
| **4** | **Biên bản Họp Khởi động Đồ án (Tuần 2)** | [Xem file .md](docs/Bien_Ban_Hop_Khoi_Dong_Tuan_2.md) | [Tải file .docx](docs/Bien_Ban_Hop_Khoi_Dong_Tuan_2.docx) | Biên bản họp chính thức chốt đề tài, thống nhất cơ cấu nhóm 5 thành viên và biểu quyết thông qua lộ trình thực hiện. |
| **5** | **Quy chuẩn Đóng góp & Git Workflow** | [Xem file .md](CONTRIBUTING.md) | — | Hướng dẫn phân nhánh GitFlow, quy chuẩn Conventional Commits, code standards PEP 8, và quy trình Pull Request. |
| **6** | **Báo cáo Dữ liệu Thị giác (Tuần 2)** | [Xem file .md](docs/Bao_Cao_Du_Lieu_Thi_Giac_Tuan_2.md) | [Tải file .docx](docs/Bao_Cao_Du_Lieu_Thi_Giac_Tuan_2.docx) | Khảo sát NTHU-DDD, Roboflow Weapons, kết quả kiểm định chất lượng tệp ảnh và công cụ tự động hóa ingestion. |

---

### 3. Cấu trúc thư mục Đồ án (Project Directory Structure)

```text
D:\Trí tuệ nhân tạo\HK3\Máy học nâng cao\Đồ án\
├── .github/                                            # Cấu hình GitHub templates
│   ├── ISSUE_TEMPLATE/                                 # Templates tạo task tuần & báo lỗi
│   │   ├── task_weekly.md
│   │   └── bug_report.md
│   └── PULL_REQUEST_TEMPLATE.md                        # Template kiểm thử chéo khi tạo PR
├── CONTRIBUTING.md                                     # Quy chuẩn đóng góp GitFlow & Code style
├── Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.md      # Bản thuyết minh khoa học & định nghĩa bài toán (Markdown)
├── Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.docx    # Văn bản chuẩn học thuật Word (in ấn / nộp giảng viên)
├── Ke_Hoach_Thuc_Hien_Do_An.md                         # Kế hoạch chi tiết 8 tuần & ma trận RACI (Markdown)
├── Ke_Hoach_Thuc_Hien_Do_An.docx                       # Kế hoạch chi tiết định dạng Word (.docx)
├── Ke_Hoach_Chi_Tiet_Tung_Tuan_Va_Thanh_Vien.md        # Kế hoạch chi tiết theo tuần & từng thành viên (Markdown)
├── Ke_Hoach_Chi_Tiet_Tung_Tuan_Va_Thanh_Vien.docx      # Kế hoạch chi tiết theo tuần & từng thành viên (.docx)
├── README.md                                           # Hướng dẫn tổng quan đồ án và quy trình thực nghiệm
├── docs/                                               # Thư mục lưu trữ tài liệu báo cáo & script sinh văn bản
│   ├── Bien_Ban_Hop_Khoi_Dong_Tuan_2.md                # Biên bản họp khởi động tuần 2
│   ├── Bien_Ban_Hop_Khoi_Dong_Tuan_2.docx              # Bản Word biên bản họp
│   ├── Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.md
│   ├── Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.docx
│   ├── Ke_Hoach_Thuc_Hien_Do_An.md
│   ├── Ke_Hoach_Thuc_Hien_Do_An.docx
│   ├── Ke_Hoach_Chi_Tiet_Tung_Tuan_Va_Thanh_Vien.md
│   ├── Ke_Hoach_Chi_Tiet_Tung_Tuan_Va_Thanh_Vien.docx
│   ├── generate_docx.py                                # Script Python sinh file Word Thuyết minh bài toán
│   ├── generate_plan_docx.py                           # Script Python sinh file Word Kế hoạch đồ án
│   ├── generate_weekly_plan_docx.py                    # Script Python sinh file Word Kế hoạch tuần & thành viên
│   └── generate_meeting_docx.py                        # Script Python sinh file Word Biên bản họp
├── data/                                               # Dữ liệu phục vụ huấn luyện và kiểm thử
│   ├── raw/                                          # Dữ liệu gốc chưa qua xử lý (video, file ppg thô)
│   ├── processed/                                    # Dữ liệu đã chuẩn hóa, cắt crop nón bảo hiểm, vector hóa
│   └── external/                                     # Metadata và liên kết tải các bộ dữ liệu mở (NTHU-DDD, WESAD, Roboflow)
├── notebooks/                                        # Jupyter Notebooks nghiên cứu & trực quan hóa
│   ├── 01_eda_and_data_preprocessing.ipynb          # Phân tích dữ liệu khám phá (EDA) và cân bằng lớp
│   ├── 02_baseline_models.ipynb                      # Huấn luyện mô hình cơ sở (LightGBM, XGBoost, Single-Modal CNN)
│   ├── 03_multimodal_fusion_training.ipynb           # Huấn luyện mô hình đề xuất (Cross-Attention Multimodal Network)
│   └── 04_evaluation_and_xai.ipynb                   # Đánh giá Stratified K-Fold, PR-AUC, Grad-CAM & SHAP
├── models/                                           # Lưu trữ trọng số mô hình đã huấn luyện
│   ├── checkpoints/                                  # Checkpoints trong quá trình train
│   └── saved_models/                                 # Mô hình tối ưu hoàn chỉnh (.pth, .onnx)
├── src/                                              # Mã nguồn module hóa của dự án
│   ├── data/                                         # Data loaders, dataset classes, augmentation pipelines
│   ├── features/                                     # Trích xuất đặc trưng HRV, spatial feature transforms
│   ├── models/                                       # Định nghĩa các kiến trúc mạng (Vision, Bio, Fusion Head)
│   ├── training/                                     # Trainer, Focal Loss, Class-Balanced Loss, learning rate schedulers
│   └── evaluation/                                   # K-Fold validator, PR-AUC, Confusion Matrix, Cost calculator
└── reports/                                          # Kết quả thực nghiệm và biểu đồ
    └── figures/                                      # Biểu đồ ROC, PR curve, Grad-CAM heatmaps, SHAP summary plots
```

---

### 4. Tóm tắt Định nghĩa Bài toán Hình thức (Formal Problem Formulation)

* **Không gian đầu vào:** $\mathbf{x}_t = (\mathbf{I}_t, \mathbf{H}_t, s_t) \in \mathcal{X}$
  * $\mathbf{I}_t \in \mathbb{R}^{3 \times 96 \times 96}$: Khung hình góc rộng ghi nhận tư thế nón bảo hiểm và người ngồi sau.
  * $\mathbf{H}_t \in \mathbb{R}^{T \times D}$: Chuỗi thời gian nhịp tim từ cảm biến PPG cùng các chỉ số dẫn xuất HRV (SDNN, RMSSD, $\Delta hr_t$).
  * $s_t \in \{0, 1\}$: Tín hiệu công tắc khẩn cấp SOS ngầm bố trí tại cùm tay lái xe máy.
* **Không gian nhãn mục tiêu:** $\mathcal{Y} = \{0: \text{Normal}, 1: \text{Drowsiness}, 2: \text{Assault/Danger}\}$
* **Hàm mục tiêu tối ưu hóa:** Tối thiểu hóa rủi ro kỳ vọng kết hợp hàm mất mát Focal Loss và phạt chi phí sai lầm bất đối xứng:
  $$\min_\Theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \mathcal{L}_{\text{FL}}(f_\Theta(\mathbf{x}), y) + \mu \cdot \mathcal{R}_{\text{cost}}(f_\Theta(\mathbf{x}), y) \right] + \lambda \|\Theta\|_2^2$$
  với mức phạt cực nặng cho lỗi bỏ sót hiểm họa ($C_{10} = 50, C_{20} = 200$).

---

### 5. Lộ trình 5 Giai đoạn Thực hiện (8 Tuần)

```text
Giai đoạn 1 (Tuần 2-3: 27/10 - 08/11/2026) : Chuẩn bị Dữ liệu đa nguồn & Pipeline Tăng cường (Data Augmentation)
Giai đoạn 2 (Tuần 4  : 09/11 - 15/11/2026) : Baseline Models (LightGBM/XGBoost) & Trích xuất Đặc trưng HRV
Giai đoạn 3 (Tuần 5-6: 16/11 - 29/11/2026) : Phát triển Khung Deep Multimodal & Tối ưu Focal Loss / Cost Matrix
Giai đoạn 4 (Tuần 7  : 30/11 - 06/12/2026) : Thẩm định Stratified 5-Fold, Thực nghiệm TP.HCM & XAI (Grad-CAM/SHAP)
Giai đoạn 5 (Tuần 8-9: 07/12 - 20/12/2026) : Hoàn thiện Quyển Báo cáo khoa học >= 40 trang & Bảo vệ Đồ án
```

---

### 6. Nguồn Dữ liệu & Quy mô Mẫu (Datasets)

1. **Ngủ gật qua Mũ & Kính râm:** Tập dữ liệu **NTHU-DDD** (National Tsing Hua University) và **UTA-RLDD** chuyên biệt về tài xế bịt kín khẩu trang/kính râm.
2. **Hung khí & Hành vi Tấn công:** Tập **Roboflow Universe Weapon Detection** (>4.200 ảnh dao, hung khí và tư thế siết cổ phía sau).
3. **Chuỗi thời gian Sinh trắc học:** Tập **WESAD** (Wearable Stress and Affect Detection - University of Freiburg) ghi nhận nhịp tim lúc thư giãn, buồn ngủ và hoảng loạn sợ hãi.
4. **Tập Thực nghiệm Đường phố Việt Nam:** 1.200 ảnh nón bảo hiểm và log nhịp tim thực tế từ **Xiaomi Smart Band** thu thập trực tiếp tại TP.HCM.

---

### 7. Chuẩn mực Thẩm định Khoa học (Evaluation Standards)
* **Phương pháp chia tập:** Stratified Subject-Independent Group K-Fold Cross-Validation ($K=5$). Không rò rỉ dữ liệu cùng một cá nhân giữa tập Train và Test.
* **Chỉ số bắt buộc:** Recall từng lớp (Recall Drowsy $\ge 95\%$, Recall Assault $\ge 98\%$), Precision Drowsy $\ge 90\%$, Macro-F1 Score, PR-AUC, và Ma trận nhầm lẫn chuẩn hóa.
* **Tính giải thích (XAI):** Trực quan hóa bản đồ nhiệt Grad-CAM trên nón bảo hiểm/hung khí và biểu đồ giá trị Shapley (SHAP) trên các biến số sinh trắc học.
