# CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
### Độc lập – Tự do – Hạnh phúc
---

# BIÊN BẢN HỌP KHỞI ĐỘNG ĐỒ ÁN MÔN HỌC (TUẦN 2)
## HỌC PHẦN: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)
### Đề tài: Khung mô hình học máy đa phương thức nhận diện trạng thái rủi ro an toàn cho tài xế xe hai bánh dựa trên thị giác máy tính và chuỗi thời gian sinh trắc học

---

### I. THÔNG TIN CHUNG VỀ CUỘC HỌP
* **Thời gian bắt đầu:** 20h00, Thứ Ba ngày 27/10/2026.
* **Thời gian kết thúc:** 21h45 cùng ngày.
* **Hình thức tổ chức:** Họp trực tuyến qua Microsoft Teams / Google Meet.
* **Chủ tọa cuộc họp:** Trần Hoàng Hôn (Trưởng nhóm – MSSV: 26410046).
* **Thư ký ghi biên bản:** Hoàng Cao Sơn (Thành viên – MSSV: 25730061).
* **Thành phần tham dự:** Có mặt đầy đủ 05/05 thành viên nhóm.

| STT | Họ và Tên | Mã số sinh viên | Vai trò trong nhóm | Trạng thái tham dự |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Trần Hoàng Hôn** | **26410046** | Trưởng nhóm / ML Architect | Có mặt |
| 2 | **Nguyễn Duy Khang** | **26410055** | Thành viên / Computer Vision Lead | Có mặt |
| 3 | **Phạm Thành Trung** | **26410141** | Thành viên / Bio-Signal Specialist | Có mặt |
| 4 | **Vũ Văn Duy** | **26410031** | Thành viên / ML Evaluation Lead | Có mặt |
| 5 | **Hoàng Cao Sơn** | **25730061** | Thành viên / Data Engineering & XAI | Có mặt |

---

### II. NỘI DUNG CUỘC HỌP & NGHỊ QUYẾT

#### 1. Thống nhất Tên đề tài và Định hướng Học thuật
* **Nội dung thảo luận:** Trưởng nhóm Trần Hoàng Hôn trình bày định hướng đề tài giải quyết vấn đề kép cho tài xế xe ôm công nghệ tại Việt Nam (ngủ gật tích tắc và nguy cơ cướp giật ban đêm). Đề tài hoàn toàn tránh các hướng làm sáo rỗng (như nhận diện hoa, phân loại biển số xe) và đáp ứng đúng tiêu chí đánh giá khắt khe của Thầy Đặng Việt Dũng.
* **Nghị quyết chốt:**
  * **Tên tiếng Việt:** *Khung mô hình học máy đa phương thức nhận diện trạng thái rủi ro an toàn cho tài xế xe hai bánh dựa trên thị giác máy tính và chuỗi thời gian sinh trắc học*.
  * **Tên tiếng Anh:** *A Multimodal Machine Learning Framework for Two-Wheel Driver Safety Risk State Recognition Integrating Computer Vision and Physiological Bio-Signal Time-Series*.
  * **Mục tiêu điểm số kỳ vọng:** $9.5 - 10.0$ điểm (đạt chuẩn học thuật báo cáo và có khả năng phát triển thành bài báo nghiên cứu).

#### 2. Phân công Vai trò Chuyên môn & Ma trận RACI 5 Thành viên
Cuộc họp đã tiến hành biểu quyết phân công công việc chuyên môn dựa trên thế mạnh của từng thành viên:
1. **Trần Hoàng Hôn (Trưởng nhóm / ML Architect):**
   * Quản lý tiến độ Git, review Pull Requests, thiết kế kiến trúc toán học đa phương thức (Cross-Modal Attention), viết báo cáo tổng hợp và điều phối bảo vệ cuối kỳ.
2. **Nguyễn Duy Khang (Computer Vision Lead):**
   * Phụ trách toàn diện nhánh thị giác máy tính: Tiền xử lý dữ liệu ảnh nón bảo hiểm và hung khí (NTHU-DDD, Roboflow Weapon), Data Augmentation chống rung lắc, huấn luyện MobileNetV3-Small và Vision Baseline.
3. **Phạm Thành Trung (Bio-Signal Specialist):**
   * Phụ trách toàn diện nhánh tín hiệu sinh học: Tiền xử lý tín hiệu PPG chuỗi thời gian (WESAD, Xiaomi Smart Band), lọc thông dải Butterworth, lập trình module trích xuất đặc trưng HRV trong `src/features/hrv.py`, huấn luyện nhánh BiLSTM.
4. **Vũ Văn Duy (ML Modeling & Evaluation Lead):**
   * Phụ trách nghiên cứu hàm mất mát tối ưu cho dữ liệu mất cân bằng cực độ: Focal Loss, Class-Balanced Loss, Cost-Sensitive Bayes Risk Matrix; thiết lập kiểm định Stratified Subject-Independent 5-Fold Cross-Validation, tính toán PR-AUC, Confusion Matrix.
5. **Hoàng Cao Sơn (Data Engineering & XAI):**
   * Xây dựng pipeline dữ liệu tự động, thu thập và gắn nhãn tập kiểm thử thực tế ngoài phố tại TP.HCM, cài đặt Grad-CAM trên nhánh ảnh và SHAP trên chuỗi nhịp tim, dựng video clip demo sản phẩm.

#### 3. Thông qua Kế hoạch Thực hiện 8 Tuần (Tuần 2 đến Tuần 9)
Nhóm đã xem xét và biểu quyết thông qua lộ trình 5 giai đoạn:
* **Giai đoạn 1 (Tuần 2-3: 27/10 – 08/11/2026):** Dữ liệu đa nguồn & Pipeline Tăng cường (Data Augmentation).
* **Giai đoạn 2 (Tuần 4: 09/11 – 15/11/2026):** Baseline Models (LightGBM/XGBoost) & Trích xuất Đặc trưng HRV.
* **Giai đoạn 3 (Tuần 5-6: 16/11 – 29/11/2026):** Phát triển Khung Deep Multimodal & Tối ưu Focal Loss / Cost Matrix.
* **Giai đoạn 4 (Tuần 7: 30/11 – 06/12/2026):** Thẩm định Stratified 5-Fold, Thực nghiệm TP.HCM & XAI (Grad-CAM/SHAP).
* **Giai đoạn 5 (Tuần 8-9: 07/12 – 20/12/2026):** Hoàn thiện Quyển Báo cáo khoa học $\ge 40$ trang & Bảo vệ Đồ án.

#### 4. Quy định Kỷ luật Git & Quy chuẩn Đóng góp Code
* Toàn bộ mã nguồn được lưu trữ tại GitHub: [`mrkiss-it/multimodal-driver-safety-ml`](https://github.com/mrkiss-it/multimodal-driver-safety-ml).
* Tuân thủ quy chuẩn `CONTRIBUTING.md`:
  - Không commit trực tiếp vào nhánh `main`.
  - Mọi tính năng phát triển trên nhánh `feature/<ten-tinh-nang>` xuất phát từ `develop`.
  - Tạo Pull Request phải có ít nhất 01 thành viên review và Trưởng nhóm duyệt merge.
  - Tuân thủ quy tắc Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`).
  - Nghiêm cấm commit file dữ liệu thô nặng $>50$MB vào Git.

#### 5. Phân công Công việc Ngay trong Tuần 2 (27/10 – 01/11/2026)
* **Hôn:** Hoàn thiện `CONTRIBUTING.md`, PR/Issue Templates, nhánh `develop` và Biên bản họp.
* **Khang:** Tải bộ dữ liệu NTHU-DDD và Roboflow Weapon về máy, kiểm tra số lượng ảnh.
* **Trung:** Tải bộ dữ liệu WESAD, viết script đọc file `.pkl` trích xuất chuỗi nhịp tim PPG.
* **Duy:** Tạo file môi trường `requirements.txt`, viết `tests/test_env.py` kiểm thử CUDA GPU.
* **Sơn:** Lên kế hoạch thu thập mẫu ảnh nón bảo hiểm ngoài đường phố TP.HCM, khởi tạo notebook EDA.

---

### III. KẾT LUẬN & BIỂU QUYẾT

Toàn thể 05 thành viên nhóm tham gia cuộc họp đã nghe, thảo luận dân chủ và nhất trí biểu quyết thông qua 100% các nội dung trong biên bản này. Biên bản được lập thành văn bản lưu vào thư mục `docs/` của đồ án và công khai trên GitHub repository của nhóm.

Cuộc họp kết thúc vào lúc 21h45 cùng ngày.

---

#### CHỮ KÝ XÁC NHẬN CỦA CÁC THÀNH VIÊN NHÓM

| STT | Họ và Tên | Vai trò | Chữ ký xác nhận |
| :---: | :--- | :--- | :---: |
| 1 | **Trần Hoàng Hôn** | Trưởng nhóm (Chủ tọa) | *(Đã ký duyệt)* |
| 2 | **Nguyễn Duy Khang** | Thành viên (Vision Lead) | *(Đã đồng thuận)* |
| 3 | **Phạm Thành Trung** | Thành viên (Bio Specialist) | *(Đã đồng thuận)* |
| 4 | **Vũ Văn Duy** | Thành viên (ML Evaluation) | *(Đã đồng thuận)* |
| 5 | **Hoàng Cao Sơn** | Thành viên (Thư ký) | *(Đã ký biên bản)* |
