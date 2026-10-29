# BÁO CÁO KỸ THUẬT: CHUẨN HÓA MÔI TRƯỜNG HỌC MÁY & RÀ SOÁT TOÁN HỌC HÀM MẤT MÁT (TUẦN 2)
## HỌC PHẦN: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)
### Đề tài: Khung mô hình học máy đa phương thức nhận diện trạng thái rủi ro an toàn cho tài xế xe hai bánh
**Người thực hiện:** Vũ Văn Duy (MSSV: 26410031) – ML Modeling & Evaluation Lead  
**Giảng viên phụ trách:** ThS./TS. Đặng Việt Dũng  
**Thời gian báo cáo:** Tuần 2 (27/10/2026 – 01/11/2026)

---

> [!IMPORTANT]
> **Ghi chú trung thực về phần cứng kiểm định (Hardware Transparency):** Máy trạm phát triển cục bộ hiện tại sử dụng vi xử lý 16 luồng logic (AMD64, Windows 10/11) đi kèm đồ họa tích hợp **Intel(R) Iris(R) Xe Graphics**, **không có GPU rời NVIDIA CUDA** (`nvidia-smi` không tồn tại, `torch.cuda.is_available() == False`). Do đó:
> - Toàn bộ 16/16 thư viện trong môi trường ảo `.venv` (`torch 2.7.1+cpu`, `torchvision 0.22.1+cpu`, `lightgbm 4.7.0`, `xgboost 2.1.3`, `albumentations 2.0.8`, `shap 0.49.1`, ...) và toàn bộ phép kiểm chứng đạo hàm số (`torch.autograd.gradcheck` ở độ chính xác kép `float64`) được thực thi và vượt qua **100% trên CPU**.
> - Bài kiểm thử `test_compute_device_resolution_and_autograd` kiểm chứng cơ chế tự động chọn thiết bị (`cuda` nếu có, hạ cấp an toàn về `cpu` nếu không có) đạt **PASSED**. Bài kiểm thử kernel vật lý `test_cuda_gpu_kernel_when_available` được gắn `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` để tự động kích hoạt khi chạy trên máy trạm có GPU NVIDIA của nhóm ở giai đoạn huấn luyện 5-Fold (Tuần 6).

---

## 1. Tổng quan Nhiệm vụ Tuần 2 & Mức độ Hoàn thành

| Đầu việc theo Kế hoạch Tuần 2 | Sản phẩm mã nguồn / Báo cáo | Trạng thái |
| :--- | :--- | :---: |
| 1. Thiết lập cấu hình môi trường Python chuẩn hóa (`pip` & `conda`) | `requirements.txt`, `environment.yml`, môi trường ảo `.venv` (16/16 gói) | ✅ Hoàn thành |
| 2. Xây dựng tiện ích kiểm tra phần cứng, cố định seed tái lập & xuất báo cáo JSON | `src/utils/env_check.py`, `reports/env_and_loss_verification.json` | ✅ Hoàn thành |
| 3. Viết bộ kiểm thử tự động cài đặt thư viện, smoke-test backbone & chẩn đoán GPU/CPU | `tests/test_env.py` (21 passed, 1 skipped do không có GPU NVIDIA) | ✅ Hoàn thành |
| 4. Rà soát toán học, sửa lỗi tiềm ẩn và bổ sung quy tắc quyết định Bayes tối ưu | `src/models/losses.py`, `tests/test_losses.py` (15/15 passed, gồm `gradcheck`) | ✅ Hoàn thành |

---

## 2. Thiết kế Môi trường Chuẩn hóa & Phòng ngừa Xung đột Thư viện

### 2.1. Danh mục 16 thư viện đã cài đặt và kiểm định thực tế trong `.venv`
Kết quả trích xuất tự động từ `python src/utils/env_check.py` (lưu tại `reports/env_and_loss_verification.json`):

| Nhóm chức năng | Thư viện (`import`) | Phiên bản thực tế | Vai trò trong Đồ án |
| :--- | :--- | :---: | :--- |
| **Tính toán khoa học & Tín hiệu** | `numpy` / `scipy` / `pandas` | `1.26.4` / `1.15.2` / `2.2.3` | Lọc thông dải Butterworth PPG, trích xuất HRV, xử lý bảng |
| **Học sâu Đa phương thức** | `torch` / `torchvision` | `2.7.1+cpu` / `0.22.1+cpu` | Backbone MobileNetV3-Small, 1D-CNN + BiLSTM, Cross-Attention, Custom Losses |
| **Học máy cổ điển & GBDT Baseline** | `sklearn` / `lightgbm` / `xgboost` | `1.6.0` / `4.7.0` / `2.1.3` | Chia tập `StratifiedGroupKFold`, mô hình đối chuẩn Tabular Tuần 4 |
| **Thị giác & Tăng cường ảnh** | `PIL` / `cv2` / `albumentations` | `11.0.0` / `4.11.0` / `2.0.8` | Đọc ảnh, Motion Blur, Random Lighting/Shadow chống rung lắc xe máy |
| **Trực quan hóa, XAI & Báo cáo** | `matplotlib` / `seaborn` / `shap` / `docx` | `3.9.3` / `0.13.2` / `0.49.1` / `1.2.0` | Đồ thị PR-AUC, Confusion Matrix, giải thích SHAP trên HRV, xuất báo cáo Word |
| **Đảm bảo chất lượng** | `pytest` | `9.1.1` | Kiểm thử đơn vị tự động toàn dự án |

### 2.2. Phát hiện kỹ thuật quan trọng về xung đột ABI khi cài đặt
Trong quá trình kiểm tra giải quyết phụ thuộc (`pip install --dry-run`), nhóm phát hiện:
1. Gói `albumentations>=2.0` mặc định kéo `opencv-python-headless>=5.0`, từ đó ép nâng cấp `numpy` lên nhánh `2.2.x` và ghi đè không gian tên `cv2` nếu máy đã cài `opencv-python` `4.x`.
2. Để bảo vệ tính tương thích nhị phân (ABI) đồng thời cho `scipy`, `numba`/`shap` và `albumentations` trên máy của cả 5 thành viên, `requirements.txt` và `environment.yml` đã khóa chặn trên an toàn: `numpy>=1.26.0,<2.1.0` và `opencv-python-headless>=4.9.0,<5.0.0`, kèm hướng dẫn phân tách rõ `--extra-index-url` cho máy CPU và máy CUDA 12.4.

---

## 3. Rà soát Toán học & Chuẩn hóa Module Hàm Mất Mát (`src/models/losses.py`)

Đối chiếu mã nguồn khởi tạo ban đầu của `src/models/losses.py` với Chương 4 của Bản thuyết minh toán học (`Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.md`), quá trình rà soát đã phát hiện và khắc phục **05 vấn đề toán học và kỹ thuật**:

### 3.1. Quản lý Buffer chuẩn PyTorch `nn.Module` và tương thích Mixed Precision
* **Vấn đề cũ:** Trọng số lớp `self.alpha` và ma trận chi phí `self.C` được gán như thuộc tính tensor thường (`torch.float32`), không tự động chuyển kiểu dữ liệu khi chạy kiểm định đạo hàm `float64` hoặc huấn luyện Mixed Precision (`AMP float16` ở Tuần 5), cũng không nằm trong `state_dict()`.
* **Khắc phục:** Đăng ký qua `self.register_buffer(...)` và ép kiểu động `.to(device=inputs.device, dtype=inputs.dtype)` trong `forward()`.

### 3.2. Độ ổn định số học của hệ số điều biến $(1 - p_t)^\gamma$ ở vùng bão hòa logit
* **Công thức Chương 4.1:**
  $$\mathcal{L}_{\text{FL}}(\mathbf{p}_t, y) = -\alpha_y (1 - p_{t, y})^\gamma \log(p_{t, y})$$
* **Vấn đề cũ:** Tính `prob = torch.exp(log_prob)` rồi `(1.0 - target_prob) ** gamma` gặp hai rủi ro:
  1. Hiện tượng triệt tiêu chữ số có nghĩa (*catastrophic cancellation*) khi $\log(p_t) \to 0^-$.
  2. Khi $p_t = 1.0$ trong `float32` (ví dụ logit `[100, -100, -100]`) và $0 < \gamma < 1$, đạo hàm $\frac{d}{dx}(x^\gamma) = \gamma x^{\gamma - 1}$ tại $x = 1 - p_t = 0$ tiến tới $\infty$, sinh ra `NaN` trong lan truyền ngược (`backward()`).
* **Khắc phục:** Sử dụng `-torch.expm1(target_log_prob)` để tính chính xác $1 - e^{\log p_t} = 1 - p_t$, tách nhánh $\gamma = 0$ (khớp tuyệt đối với `nn.CrossEntropyLoss`, sai số tuyệt đối $0.0$), và kẹp ngưỡng dưới $\epsilon = 10^{-12}$ khi $0 < \gamma < 1$.

### 3.3. Ngữ nghĩa phép lấy trung bình batch (`reduction='mean'`) khi có trọng số $\alpha_c$
* Trong `nn.CrossEntropyLoss(weight=alpha, reduction='mean')` của PyTorch, tổng loss được chia cho $\sum_{i=1}^B \alpha_{y_i}$. Hệ quả là nếu một mini-batch nhỏ chỉ chứa các mẫu thuộc cùng một lớp thiểu số $c$, hệ số $\alpha_c$ ở tử số và mẫu số **tự triệt tiêu hoàn toàn** ($\frac{\alpha_c \sum \ell_i}{B \alpha_c} = \frac{1}{B}\sum \ell_i$).
* Cài đặt `MultiClassFocalLoss` của nhóm chia cho kích thước batch $B$ theo đúng bài báo gốc của *Lin et al. (ICCV 2017)*, bảo toàn tỷ lệ khuếch đại gradient $\alpha_c$ cho lớp thiểu số trong mọi cấu hình batch.

### 3.4. Trọng số Số mẫu Hiệu dụng (`ClassBalancedFocalLoss`) trên phân bố 70 : 20 : 10
* Với quy mô mục tiêu của đồ án $N = [4200, 1200, 600]$ (70% Normal : 20% Drowsiness : 10% Assault) và $\beta = 0.999$, thể tích mẫu hiệu dụng $E_{n_c} = \frac{1 - \beta^{n_c}}{1 - \beta}$ sinh ra bộ trọng số chuẩn hóa ($\sum_{c=0}^2 w_c = 3$):
  $$\mathbf{w}_{0.999} = [0.653361,\; 0.920739,\; 1.425900] \quad \left(\frac{w_2}{w_0} \approx 2.182\right)$$
* Khi tăng $\beta \to 0.9999$, tỷ số trọng số tiến dần về tỷ lệ nghịch tần suất mẫu ($w_2 / w_0 \approx 5.83$). Siêu tham số $\beta \in \{0.999, 0.9999\}$ và $\gamma \in \{1.5, 2.0, 3.0\}$ sẽ được đưa vào Ablation Study ở Tuần 6. Đã bổ sung kiểm tra biên chặt chẽ (`beta in [0, 1)`, `n_c > 0`, giới hạn $\beta = 0 \implies \mathbf{w} = [1, 1, 1]$).

### 3.5. Sửa lỗi `CostSensitiveBayesLoss`, phân tích bão hòa Softmax & Hàm mục tiêu kết hợp `CombinedFocalCostLoss`
* **Sửa lỗi `reduction`:** Bản nháp ban đầu của `CostSensitiveBayesLoss` bỏ sót nhánh `reduction='none'` (trả nhầm về `loss.sum()`). Đã chuẩn hóa cả 3 chế độ `'mean'`, `'sum'`, `'none'`.
* **Phát hiện hiện tượng triệt tiêu Gradient của rủi ro tuyến tính theo Softmax:**
  Xét hàm rủi ro kỳ vọng khả vi:
  $$\mathcal{R}_{\text{cost}}(\mathbf{p}, y) = \sum_{c=0}^2 p_c C_{y, c}, \quad \mathbf{p} = \text{Softmax}(\mathbf{z})$$
  Vì Jacobian của Softmax là $\frac{\partial p_c}{\partial z_k} = p_c(\delta_{ck} - p_k)$, nếu một mẫu **Tấn công ($y = 2$)** bị mô hình dự đoán sai hoàn toàn về lớp Bình thường với logit bão hòa $\mathbf{z} = [20, -20, -20]$ ($p_0 \approx 1, p_2 \approx 4.2 \times 10^{-18}$), chuẩn gradient của $\mathcal{R}_{\text{cost}}$ sụp đổ về $\|\nabla_{\mathbf{z}} \mathcal{R}_{\text{cost}}\|_2 < 10^{-10}$ — mô hình không thể tự sửa sai!
* **Giải pháp (`CombinedFocalCostLoss`):** Hiện thực hóa đúng hàm mục tiêu tổng hợp đã đề xuất tại Chương 4:
  $$\mathcal{L}_{\text{total}}(\mathbf{z}, y) = \mathcal{L}_{\text{FL}}(\mathbf{z}, y) + \mu \cdot \mathcal{R}_{\text{cost}}(\mathbf{z}, y)$$
  Thành phần logarit $-\log(p_y)$ trong $\mathcal{L}_{\text{FL}}$ khử đạo hàm mũ của Softmax, duy trì lực kéo gradient mạnh cỡ $O(1)$ ($\|\nabla_{\mathbf{z}} \mathcal{L}_{\text{total}}\|_2 > 1.0$) ngay cả khi logit bị bão hòa sai, trong khi $\mu \cdot \mathcal{R}_{\text{cost}}$ tinh chỉnh biên quyết định bất đối xứng ở vùng tranh chấp xác suất.

### 3.6. Bổ sung Quy tắc Quyết định Bayes Tối ưu ở Pha Suy luận (`bayes_optimal_predict`)
Hiện thực hóa Công thức (4.3) với ma trận chi phí bất đối xứng $\mathbf{C}$ ($C_{10} = 50, C_{20} = 200$):
$$\hat{y}^* = \arg\min_{k \in \{0, 1, 2\}} \sum_{j=0}^2 C_{j, k} \cdot P(y = j \mid \mathbf{x}) = \arg\min_{k} (\mathbf{p}\mathbf{C})_k$$

Kết quả kiểm chứng định lượng từ `reports/env_and_loss_verification.json`:

| Kịch bản phân phối hậu nghiệm $\mathbf{p} = [P(0), P(1), P(2)]$ | Chi phí kỳ vọng $(\mathbf{p}\mathbf{C}) = [R(0), R(1), R(2)]$ | Dự đoán `argmax` ngây thơ | Dự đoán Bayes tối ưu $\hat{y}^*$ | Ý nghĩa an toàn thực tiễn |
| :--- | :---: | :---: | :---: | :--- |
| `[0.850, 0.050, 0.100]` *(10% nguy cơ Tấn công)* | `[22.500, 2.850, 1.950]` | `0` (Normal) | **`2` (Assault)** | Ưu tiên cảnh báo khẩn cấp vì bỏ sót Tấn công bị phạt $C_{20}=200$ |
| `[0.900, 0.090, 0.010]` *(9% nguy cơ Ngủ gật)* | `[6.500, 1.100, 2.250]` | `0` (Normal) | **`1` (Drowsiness)** | Nhắc nhở tài xế sớm trước khi rơi vào giấc ngủ ngắn (micro-sleep) |
| `[0.995, 0.004, 0.001]` *(Lái xe tỉnh táo an toàn)* | `[0.400, 1.015, 2.010]` | `0` (Normal) | **`0` (Normal)** | Không gây báo động giả khi xác suất rủi ro thực sự thấp |

---

## 4. Kết quả Kiểm thử Tự động (`pytest`)

Chạy toàn bộ bộ kiểm thử tự động của dự án trong môi trường chuẩn `.venv`:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
* `tests/test_env.py`: **21 passed, 1 skipped** (kiểm tra Python $\ge 3.10$, đầy đủ file cấu hình, import 16 thư viện, forward smoke-test `MobileNetV3-Small` trên ảnh $3 \times 96 \times 96$, pipeline `Albumentations`, `LightGBM + SHAP TreeExplainer`, tính tái lập `set_global_seed`, lan truyền ngược autograd; chỉ skip test kernel vật lý CUDA trên máy Intel Iris Xe).
* `tests/test_losses.py`: **15 passed** (đối chiếu $\gamma = 0$ với `CrossEntropyLoss`, kiểm chứng $\alpha$-scaling trên batch đơn lớp, chống tràn số/NaN ở logit $\pm 100$, kiểm chứng đạo hàm giải tích bằng `torch.autograd.gradcheck` ở `float64` cho cả 4 lớp hàm mất mát, kiểm chứng quy tắc quyết định Bayes và bắt ngoại lệ đầu vào).
* Các test hiện hữu của nhóm (`tests/test_vision_data.py`: 5 passed, `tests/test_bio_data.py`: 17 passed) giữ nguyên trạng thái hoạt động ổn định.
* **Tổng cộng toàn dự án:** **58 passed, 1 skipped, 0 failed**.

---

## 5. Bàn giao & Kế hoạch Chuẩn bị cho Tuần 3

1. **Hướng dẫn đồng bộ cho 5 thành viên:** Các thành viên sau khi `git pull origin develop` chỉ cần khởi tạo `.venv` theo `requirements.txt` (hoặc `conda env create -f environment.yml`) và chạy `pytest tests/test_env.py` để xác nhận môi trường đạt chuẩn trước khi code Tuần 3.
2. **Sẵn sàng cho nhiệm vụ Tuần 3 của Vũ Văn Duy:**
   * Xây dựng `src/data/multimodal_dataset.py` (`Dataset` & `DataLoader` PyTorch) ghép nối đồng bộ tensor ảnh $3 \times 96 \times 96$ từ nhánh của Nguyễn Duy Khang và cửa sổ chuỗi nhịp tim $T = 30\text{s}$ từ nhánh của Phạm Thành Trung theo giao diện dữ liệu do Trần Hoàng Hôn quy định (`splits_5fold.json`).
   * Xây dựng `tests/test_dataset.py` kiểm định kích thước batch, kiểu tensor và tỷ lệ phân tầng 70% : 20% : 10% qua từng Fold.

---
*Báo cáo do Vũ Văn Duy (MSSV: 26410031 – ML Modeling & Evaluation Lead) thực hiện và kiểm định – Tuần 2.*
