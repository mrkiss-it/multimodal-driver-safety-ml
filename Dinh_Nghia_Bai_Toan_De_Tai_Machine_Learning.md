# BÁO CÁO XÁC ĐỊNH BÀI TOÁN & BẢN THUYẾT MINH KHOA HỌC ĐỀ TÀI
## HỌC PHẦN: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)

---

### **TÊN ĐỀ TÀI KHOA HỌC:**
* **Tiếng Việt:** **Khung mô hình học máy đa phương thức nhận diện trạng thái rủi ro an toàn cho tài xế xe hai bánh dựa trên thị giác máy tính và chuỗi thời gian sinh trắc học**
* **Tiếng Anh:** **A Multimodal Machine Learning Framework for Two-Wheel Driver Safety Risk State Recognition Integrating Computer Vision and Physiological Bio-Signal Time-Series**

---

### **THÔNG TIN HỌC PHẦN & NHÓM TÁC GIẢ**
* **Cơ sở đào tạo:** Trường Đại học Công nghệ Thông tin – Đại học Quốc gia TP. Hồ Chí Minh (UIT - VNU-HCM)
* **Học phần:** Máy học nâng cao
* **Giảng viên phụ trách:** ThS./TS. Đặng Việt Dũng
* **Học kỳ:** Học kỳ 3 (Năm học 2026 – 2027)
* **Nhóm sinh viên thực hiện (Nhóm 5 thành viên):**

| STT | Họ và Tên | Mã số sinh viên | Vai trò trong dự án | Trách nhiệm chuyên môn học máy |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Trần Hoàng Hôn** | **26410046** | **Trưởng nhóm / ML System Architect** | Thiết kế khung kiến trúc học máy tổng thể, mô hình hóa toán học bài toán phân loại đa trạng thái, xây dựng chiến lược dung hợp đa phương thức (Multimodal Fusion). |
| **2** | **Nguyễn Duy Khang** | **26410055** | **Computer Vision Engineer** | Phụ trách nhánh thị giác máy tính: Tiền xử lý dữ liệu ảnh, thiết kế kiến trúc trích xuất đặc trưng không gian (Spatial Feature Extraction) cho tư thế nón bảo hiểm và nhận diện hung khí. |
| **3** | **Phạm Thành Trung** | **26410141** | **Time-Series & Bio-Signal Specialist** | Phụ trách nhánh chuỗi thời gian sinh lý học: Tiền xử lý tín hiệu PPG/HR, kỹ thuật trích xuất đặc trưng miền thời gian/miền tần số (HRV Features), lọc nhiễu chuyển động (Motion Artifacts). |
| **4** | **Vũ Văn Duy** | **26410031** | **Machine Learning & Evaluation Lead** | Thiết kế hàm mất mát chuyên biệt (Focal Loss, Class-Balanced Loss), thiết lập quy trình kiểm định chéo Stratified Group K-Fold, phân tích ma trận nhầm lẫn và đánh giá chi phí sai lầm bất đối xứng. |
| **5** | **Hoàng Cao Sơn** | **25730061** | **Data Engineering & Explainable AI (XAI)** | Xây dựng pipeline dữ liệu (Data Ingestion & Augmentation), tích hợp các kỹ thuật giải thích mô hình (Grad-CAM, SHAP), thực nghiệm đối chứng mô hình trên tập dữ liệu thực tế tại Việt Nam. |

* **Thời gian hoàn thành thuyết minh bài toán:** 27/10/2026

---

## TÓM TẮT ĐỀ TÀI (ABSTRACT)
Nghiên cứu này đề xuất một khung học máy đa phương thức (*Multimodal Machine Learning Framework*) nhằm nhận diện sớm và chính xác các trạng thái rủi ro an toàn đe dọa tính mạng tài xế xe máy công nghệ tại các đô thị đang phát triển, bao gồm: **hiểm họa tai nạn do ngủ gật tức thì (*micro-sleep/drowsiness*)** và **hiểm họa an ninh do bị khách ngồi sau tấn công/cướp đoạt tài sản (*rear passenger assault/hostile threat*)**.

Trong bối cảnh thực tiễn tại Việt Nam, các phương pháp giám sát trạng thái tài xế (Driver Monitoring System - DMS) truyền thống dựa trên điểm mốc khuôn mặt (*Facial Landmarks*) như tỷ lệ mở mắt (EAR), cử động miệng (MAR) hay hướng nhìn (*Gaze Direction*) hoàn toàn bị tê liệt do thói quen đặc thù của người lái xe máy: trùm khẩu trang kín mặt, đeo kính râm chống bụi/nắng và đội mũ bảo hiểm che khuất phần lớn diện mạo. Để vượt qua rào cản này, nghiên cứu chuyển dịch trọng tâm bài toán học máy sang hai phương thức dữ liệu mới:
1. **Dữ liệu thị giác phi khuôn mặt (*Non-facial Visual Representation*):** Mô hình hóa tư thế nghiêng gục của mũ bảo hiểm (*Helmet Pose Estimation*) và phát hiện cử chỉ vươn tay/hung khí của người ngồi sau thông qua mạng nơ-ron tích chập (CNN).
2. **Dữ liệu sinh trắc học dạng chuỗi thời gian (*Physiological Time-Series Data*):** Khai thác động học biến thiên nhịp tim (Heart Rate Variability - HRV) trích xuất từ cảm biến quang phổ thể tích (PPG) trên thiết bị đeo tay thương mại, phản ánh trực tiếp trạng thái ức chế thần kinh khi lơ mơ ngủ gật (nhịp tim sụt giảm sâu) hoặc phản xạ sợ hãi tột độ (*Fight-or-Flight*) khi bị uy hiếp (nhịp tim tăng vọt).

Bài toán được phát biểu hình thức dưới dạng **Phân loại chuỗi đa phương thức có điều kiện bất đối xứng chi phí và mất cân bằng dữ liệu cực độ (*Extreme Class-Imbalanced Cost-Sensitive Multimodal Classification*)**. Nhóm nghiên cứu đề xuất và so sánh ba chiến lược dung hợp (Early, Late và Cross-Modal Attention Intermediate Fusion) kết hợp hàm mất mát trọng số tập trung (Focal Loss) và hàm mất mát cân bằng lớp (Class-Balanced Loss). Quy trình thẩm định khoa học được thiết lập chặt chẽ bằng chiến lược **Stratified Subject-Independent K-Fold Cross-Validation** trên các tập dữ liệu chuẩn quốc tế (NTHU-DDD, WESAD, Roboflow Weapon) cùng tập dữ liệu thực nghiệm đường phố tại Việt Nam, kết hợp phương pháp giải thích mô hình (Grad-CAM và SHAP) nhằm đảm bảo tính minh bạch, độ tin cậy và khả năng triển khai thực tế.

**Từ khóa (Keywords):** *Advanced Machine Learning, Multimodal Sensor Fusion, Driver Drowsiness Detection, Violence & Assault Recognition, Extreme Class Imbalance, Focal Loss, Helmet Pose Estimation, Heart Rate Variability (HRV), Explainable AI (Grad-CAM/SHAP).*

---

## CHƯƠNG 1: BỐI CẢNH THỰC TIỄN & TÍNH CẤP THIẾT CỦA BÀI TOÁN HỌC MÁY

### 1.1. Bối cảnh thực tiễn và hiểm họa kép tại Việt Nam
Tại các đô thị lớn ở Việt Nam như TP. Hồ Chí Minh và Hà Nội, sự bùng nổ của các nền tảng kinh tế số (Grab, Be, Gojek, ShopeeFood, Xanh SM...) đã tạo nên lực lượng lao động hàng trăm nghìn tài xế xe gắn máy hai bánh. Đây là lực lượng nòng cốt bảo đảm chuỗi cung ứng giao vận nhưng lại phải làm việc trong môi trường đặc biệt đơn độc và dễ bị tổn thương nhất trong hệ thống giao thông đô thị.

Hai rủi ro lớn nhất đe dọa sinh mạng tài xế bao gồm:
1. **Rủi ro an ninh nghiêm trọng (Bị tấn công bạo lực & cướp xe ban đêm):**
   * Đối tượng tội phạm thường đặt các cuốc xe đêm muộn tới các khu vực vùng ven, nghĩa trang, bãi đất trống hoặc hẻm cụt vắng vẻ. Khi di chuyển, đối tượng ngồi ở vị trí yên sau bất ngờ sử dụng dây dù/dây thừng siết cổ từ phía sau, kề dao vào mạn sườn khống chế hoặc dùng hung khí đánh đập tài xế để cướp phương tiện và tài sản.
   * *Đặc thù chí mạng:* Tài xế bị tấn công hoàn toàn bất ngờ từ "điểm mù" phía sau lưng. Trong trạng thái hoảng loạn và bị khống chế cơ học, việc mở khóa điện thoại thông minh để gọi trợ giúp là bất khả thi và có thể kích động kẻ cướp gây án tàn độc hơn.
2. **Rủi ro tai nạn do ngủ gật tích tắc (*Micro-sleep* do lao động kiệt sức):**
   * Áp lực mưu sinh khiến tài xế làm việc trung bình 10 – 14 tiếng mỗi ngày ngoài đường phố. Căng thẳng thần kinh kéo dài dẫn đến các cơn ngủ gật tích tắc kéo dài từ 1 đến 3 giây (*Micro-sleep episodes*).
   * Trên phương tiện xe máy di chuyển ở tốc độ $30 - 50$ km/h, chỉ 1.5 giây mất nhận thức đồng nghĩa phương tiện đã lao đi mất kiểm soát từ $12 - 20$ mét, dẫn đến các vụ va chạm đối đầu hoặc tông vào dải phân cách gây tử vong tại chỗ.

```
+───────────────────────────────────────────────────────────────────────────────+
|                  HIỂM HỌA KÉP ĐỐI VỚI TÀI XẾ XE HAI BÁNH                      |
+───────────────────────────────────────┬───────────────────────────────────────+
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
┌───────────────────────────────────────┐ ┌─────────────────────────────────────┐
|      HIỂM HỌA AN NINH (CƯỚP GIẬT)     | |    HIỂM HỌA TAI NẠN (NGỦ GẬT)       |
| - Bị khống chế bất ngờ từ phía sau    | | - Áp lực ca làm việc kéo dài 10-14h |
| - Đối tượng siết cổ, kề dao, hung khí | | - Hiện tượng Micro-sleep (1-3s)     |
| - Hoàn toàn tê liệt khả năng kêu cứu  | | - Mất lái, va chạm tốc độ cao       |
└───────────────────┬───────────────────┘ └──────────────────┬──────────────────┘
                    │                                        │
                    └───────────────────┬────────────────────┘
                                        ▼
+───────────────────────────────────────────────────────────────────────────────+
|        YÊU CẦU CẤP THIẾT: KHUNG HỌC MÁY ĐA PHƯƠNG THỨC THỜI GIAN THỰC          |
|    - Nhận diện trạng thái Ngủ gật (Drowsiness) và Bị tấn công (Assault)       |
|    - Vượt qua rào cản trang phục "kín mít" và môi trường rung lắc ngoài trời   |
+───────────────────────────────────────────────────────────────────────────────+
```

### 1.2. Thách thức khoa học đặc thù dưới góc độ Machine Learning
Dưới lăng kính Khoa học Dữ liệu và Học máy Nâng cao, bài toán này mang tính thách thức học thuật sâu sắc hơn rất nhiều so với các bài toán phân loại thông thường:

* **Thách thức 1: Rào cản che khuất khuôn mặt nghiêm ngặt (*Severe Occlusion Problem*):**
  Hơn 95% tài xế xe ôm công nghệ tại Việt Nam trùm kín khẩu trang y tế/vải chống bụi, đeo kính râm hoặc hạ kính chắn gió tráng gương, và đội mũ bảo hiểm. Các mô hình dựa trên trích xuất Facial Landmarks (Dlib 68 points, MediaPipe 468 points) để tính EAR (*Eye Aspect Ratio*) hay MAR (*Mouth Aspect Ratio*) hoàn toàn mất điểm neo dữ liệu (*Landmark Collapse*), dẫn đến độ chính xác suy giảm về mức ngẫu nhiên.
* **Thách thức 2: Sự mất cân bằng dữ liệu cực độ (*Extreme Class Imbalance*):**
  Trong quá trình vận hành phương tiện tự nhiên, thời lượng lái xe bình thường (*Normal Driving*) chiếm trên $98\% - 99\%$ tổng thời gian quan sát. Trạng thái ngủ gật chiếm chưa tới $1\%$, và trạng thái bị tấn công là sự kiện cực kỳ hiếm gặp ($< 0.1\%$). Nếu áp dụng các hàm mất mát tiêu chuẩn như Cross-Entropy thông thường, mô hình sẽ bị thiên lệch cực độ (*Majority Class Bias*), dự đoán tất cả là "Bình thường" và đạt độ chính xác ảo (Accuracy 99%) nhưng hoàn toàn thất bại trong việc cứu tính mạng con người.
* **Thách thức 3: Dữ liệu đa phương thức không đồng nhất & bất đối xứng về tốc độ lấy mẫu (*Heterogeneous Asynchronous Modalities*):**
  Hệ thống xử lý đồng thời hai nguồn dữ liệu có bản chất vật lý và cấu trúc tensor hoàn toàn khác nhau:
  - Luồng dữ liệu thị giác $\mathbf{X}_{\text{vis}} \in \mathbb{R}^{B \times C \times H \times W}$ với tần số lấy mẫu cao ($10 - 15$ FPS), chứa đựng thông tin không gian cục bộ giàu ngữ nghĩa nhưng chịu ảnh hưởng lớn bởi điều kiện quang học (ngược sáng, bóng tối).
  - Luồng tín hiệu sinh trắc học $\mathbf{X}_{\text{bio}} \in \mathbb{R}^{B \times T \times D}$ với tần số lấy mẫu thấp ($1$ Hz từ cảm biến PPG), chứa đựng động học sinh lý toàn cục nhưng có độ trễ quán tính sinh học nhất định.
  - Tín hiệu kích hoạt rời rạc $s \in \{0, 1\}$ từ nút nhấn cơ học khẩn cấp.
  Việc dung hợp các nguồn tin này sao cho khai thác được tính bổ trợ tương hỗ (*Complementary Information*) mà không bị áp đảo bởi phương thức có chiều không gian lớn là một câu hỏi nghiên cứu mở.
* **Thách thức 4: Nhiễu môi trường động (*Environmental Noise & Motion Artifacts*):**
  Mặt đường gồ ghề, ổ gà và rung chấn từ khối động cơ xe máy tạo ra hiện tượng nhòe chuyển động (*Motion Blur*) trên cảm biến quang học máy ảnh, đồng thời sinh ra các xung giả (*Motion Artifacts - MA*) làm méo dạng sóng mạch quang thể tích PPG của vòng đeo tay thông minh.

---

## CHƯƠNG 2: TỔNG QUAN HỌC THUẬT & KHOẢNG TRỐNG NGHIÊN CỨU (LITERATURE REVIEW)

### 2.1. Phân tích các hướng tiếp cận hiện hữu

#### 2.1.1. Hệ thống phát hiện buồn ngủ tài xế (Driver Drowsiness Detection - DDD)
Các công trình nghiên cứu trên thế giới chủ yếu chia làm ba nhóm tiếp cận:
* **Phương pháp thị giác khuôn mặt:** Các công trình tiêu biểu của *Soukupová & Čech (2016)* giới thiệu chỉ số EAR dựa trên 6 điểm mốc mí mắt; *Abbas (2020)* kết hợp CNN và RNN để theo dõi chuỗi chớp mắt. Tuy nhiên, tất cả các công trình này đều tiền giả định khuôn mặt tài xế lộ rõ hoàn toàn trong buồng lái xe hơi khép kín có điều hòa nhiệt độ, hoàn toàn không tương thích với điều kiện xe máy ngoài trời tại Đông Nam Á.
* **Phương pháp động lực học xe:** *Sayed & Eskandarian (2001)*, *McDonald et al. (2014)* phân tích góc đánh vô-lăng và độ lệch làn đường. Phương pháp này đòi hỏi hệ thống bus CAN trên xe hơi và không thể áp dụng cho xe hai bánh vốn có động học điều khiển linh hoạt và không có cảm biến gắn sẵn.
* **Phương pháp sinh lý học y tế:** *Lal & Craig (2001)* chứng minh điện não đồ (EEG) và điện tâm đồ (ECG) là "tiêu chuẩn vàng" để xác định ngủ gật. Tuy nhiên, việc gắn điện cực ướt lên trán hoặc ngực tài xế xe máy khi đang giao hàng hay chở khách là hoàn toàn phi thực tế. Gần đây, việc ứng dụng cảm biến PPG quang học trên Smartwatch (*Schmidt et al., 2018 - WESAD Dataset*) mở ra cơ hội lớn nhờ tính tiện dụng và khả năng trích xuất nhịp tim liên tục qua BLE.

#### 2.1.2. Hệ thống phát hiện hành vi bạo lực & hung khí (Violence and Weapon Detection)
* Các kiến trúc như 3D ResNet (*Carreira & Zisserman, 2017* - I3D) hay SlowFast (*Feichtenhofer et al., 2019*) đạt độ chính xác cao trong bài toán Action Recognition nhưng có số lượng tham số khổng lồ ($>30$ triệu tham số), đòi hỏi GPU máy chủ mạnh để xử lý và không thể phản hồi tức thì.
* Trong phát hiện vũ khí, các công trình của *Pérez-Hernández et al. (2020)* áp dụng YOLOv5/v8 để phát hiện dao và súng trong các khung hình giám sát an ninh công cộng.

### 2.2. Khoảng trống học thuật xác định (Identified Research Gaps)
Qua rà soát y văn, nhóm tác giả chỉ ra ba khoảng trống học thuật cốt lõi mà đề tài tập trung giải quyết:

| STT | Khoảng trống học thuật (Research Gap) | Giới hạn của các công trình hiện nay | Đóng góp mới của đề tài môn Máy học |
| :---: | :--- | :--- | :--- |
| **GAP 1** | **Mô hình hóa tư thế Mũ bảo hiểm thay thế khuôn mặt bị che khuất** | 100% các công trình DMS hiện hữu dựa trên Face Landmarks, thất bại khi tài xế đeo kính râm + khẩu trang. | Đề xuất trích xuất đặc trưng tư thế mũ bảo hiểm (*Helmet Pose Representation*) và góc gục trọng tâm đầu thông qua mạng nơ-ron học sâu. |
| **GAP 2** | **Cơ chế dung hợp Đa phương thức Thị giác - Sinh trắc học chuỗi thời gian** | Các nghiên cứu thị giác hoặc sinh trắc học thường đứng độc lập, dẫn đến tỷ lệ báo động giả (*False Alarm*) rất cao khi tài xế cúi đầu xem gương/đồng hồ. | Xây dựng mô hình Dung hợp Đa phương thức (Multimodal Fusion) kết hợp góc nghiêng mũ với xu hướng sụt giảm nhịp tim sinh học, triệt tiêu báo động giả. |
| **GAP 3** | **Chiến lược học máy nhạy cảm chi phí cho hiểm họa an ninh kép** | Các mô hình phân loại thường tối ưu Accuracy hoặc Cross-Entropy đồng đều giữa các lớp, không phân định chi phí hiểm họa. | Thiết lập hàm mất mát nhạy cảm chi phí (*Cost-Sensitive Loss*) và Focal Loss, phạt nặng sai số bỏ sót nguy hiểm (*False Negative* cho lớp Nguy hiểm). |

---

## CHƯƠNG 3: ĐỊNH NGHĨA HÌNH THỨC BÀI TOÁN HỌC MÁY (FORMAL MATHEMATICAL PROBLEM FORMULATION)

### 3.1. Mô hình toán học tổng quát
Xét bài toán học có giám sát đa phương thức (*Multimodal Supervised Learning*). Tại mỗi bước thời gian $t$, hệ thống nhận một bộ dữ liệu quan sát đầu vào $\mathbf{x}_t \in \mathcal{X}$ và cần dự đoán nhãn trạng thái $\hat{y}_t \in \mathcal{Y}$.

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                FORMAL MATHEMATICAL FORMULATION                              |
|                                                                                             |
|   1. Input Space:                                                                           |
|      X_t = ( I_t, H_t, s_t ) in X                                                           |
|        - Visual Tensor:      I_t in R^{C x H x W}   (Grayscale/RGB image, H=W=96 or 160)     |
|        - Temporal Sequence:  H_t = [hr_{t-k}, ..., hr_t]^T in R^{k+1}  (Heart Rate PPG)     |
|        - Discrete Signal:    s_t in {0, 1}          (Handlebar tactile SOS trigger)         |
|                                                                                             |
|   2. Target State Space:                                                                    |
|      Y = { 0: Normal, 1: Drowsiness, 2: Assault / Extreme Danger }                          |
|                                                                                             |
|   3. Hypothesis Class:                                                                      |
|      f_Theta: X -> Delta^2                                                                  |
|      hat{y}_t = argmax_{c in Y} f_Theta(X_t)_c                                              |
|                                                                                             |
|   4. Objective:                                                                             |
|      min_Theta E_{(x, y) ~ D} [ L_task(f_Theta(x), y) ] + lambda * Omega(Theta)             |
+─────────────────────────────────────────────────────────────────────────────────────────────+
```

### 3.2. Không gian đầu vào đa phương thức (Input Representation Space $\mathcal{X}$)
Không gian đặc trưng đầu vào tại thời điểm $t$ là tích Descartes của ba không gian con:
$$\mathcal{X} = \mathcal{X}_{\text{vis}} \times \mathcal{X}_{\text{bio}} \times \mathcal{X}_{\text{disc}}$$

1. **Không gian Thị giác Không gian ($\mathcal{X}_{\text{vis}} \subset \mathbb{R}^{C \times H \times W}$):**
   * $\mathbf{I}_t$: Khung hình số thu nhận từ camera góc rộng ($120^\circ$) quan sát đồng thời tư thế mũ bảo hiểm tài xế và vùng chuyển động của hành khách ngồi sau.
   * Kích thước chuẩn hóa: $C \in \{1, 3\}$, chiều cao $H = 96$, chiều rộng $W = 96$ (hoặc $160 \times 160$).
   * Không gian giá trị pixel: $I_{c, i, j} \in [-1.0, 1.0]$.
2. **Không gian Chuỗi thời gian Sinh trắc học ($\mathcal{X}_{\text{bio}} \subset \mathbb{R}^{T \times D}$):**
   * Chuỗi quan sát nhịp tim sinh học trong cửa sổ trượt quá khứ độ dài $T$ chu kỳ:
     $$\mathbf{H}_t = \left[ hr_{t-T+1}, hr_{t-T+2}, \dots, hr_{t-1}, hr_t \right]^\top \in \mathbb{R}^T$$
   * Vector đặc trưng dẫn xuất miền thời gian và tần số của biến thiên nhịp tim (HRV Features) $\mathbf{v}_{\text{hrv}} \in \mathbb{R}^D$:
     - Độ lệch chuẩn các khoảng cách nhịp đập liền kề (SDNN - *Standard Deviation of NN intervals*):
       $$\text{SDNN} = \sqrt{\frac{1}{T-1} \sum_{i=1}^T (hr_i - \overline{hr})^2}$$
     - Căn bậc hai trung bình bình phương các hiệu số liên tiếp (RMSSD):
       $$\text{RMSSD} = \sqrt{\frac{1}{T-1} \sum_{i=1}^{T-1} (hr_{i+1} - hr_i)^2}$$
     - Tỷ lệ sụt giảm nhịp tim tương đối so với mức cơ sở khi lái xe tỉnh táo ($\overline{hr}_{\text{base}}$):
       $$\Delta hr_t = \frac{hr_t - \overline{hr}_{\text{base}}}{\overline{hr}_{\text{base}}} \times 100\%$$
3. **Không gian Tín hiệu Rời rạc Khẩn cấp ($\mathcal{X}_{\text{disc}} \in \{0, 1\}$):**
   * $s_t$: Tín hiệu ngắt logic từ công tắc xúc giác mini giấu kín dưới ngón tay cái tại cùm tay lái:
     $$s_t = \begin{cases} 1 & \text{khi tài xế chủ động kích hoạt SOS ngầm} \\ 0 & \text{trạng thái thả tự do bình thường} \end{cases}$$

### 3.3. Không gian nhãn mục tiêu (Target Label Space $\mathcal{Y}$)
Hệ thống giải quyết bài toán phân loại đa lớp ba trạng thái ($K = 3$):
$$\mathcal{Y} = \{0, 1, 2\}$$

* **Lớp 0 ($y = 0$ - Normal Driving):** Trạng thái lái xe an toàn, tỉnh táo. Tư thế mũ bảo hiểm giữ thẳng hoặc dao động góc nhỏ ($\theta < 20^\circ$), nhịp tim sinh lý ổn định trong dải bình thường khi điều khiển phương tiện ($75 \le hr_t \le 95$ BPM), người ngồi sau ngồi ngay ngắn.
* **Lớp 1 ($y = 1$ - Drowsiness / Micro-sleep):** Trạng thái ngủ gật hoặc mất tỉnh táo nghiêm trọng. Mũ bảo hiểm cúi gục về phía trước hoặc ngả nghiêng sang bên vượt ngưỡng an toàn ($\theta \ge 25^\circ$) duy trì liên tục qua cửa sổ thời gian trễ sinh lý ($\Delta t \ge 1.5$s), đồng thời hệ thần kinh phó giao cảm chiếm ưu thế khiến nhịp tim sinh lý suy giảm ($\Delta hr_t \le -15\%$).
* **Lớp 2 ($y = 2$ - Assault / Extreme Hazard):** Trạng thái tài xế bị đe dọa an ninh, bị khống chế hoặc tấn công bạo lực. Ghi nhận cử chỉ hung hãn từ phía sau (tay siết cổ, giơ vật sắc nhọn) hoặc nhịp tim tài xế tăng đột biến do phản xạ giao cảm hoảng sợ (*Fight-or-Flight*, $hr_t > 130$ BPM, $\Delta hr_t \ge +50\%$) hoặc tài xế chủ động ấn nút SOS ($s_t = 1$).

### 3.4. Hàm suy luận & Các chiến lược Dung hợp Đa phương thức (Multimodal Fusion Strategies)
Mô hình học máy tìm hàm ánh xạ $f_\Theta(\mathbf{x}_t)$ chiếu từ không gian đa phương thức vào phân phối xác suất trên 3 lớp:
$$\mathbf{p}_t = f_\Theta(\mathbf{x}_t) = \left[ P(y=0|\mathbf{x}_t), P(y=1|\mathbf{x}_t), P(y=2|\mathbf{x}_t) \right]^\top \in \Delta^2$$

Đồ án triển khai và thực nghiệm so sánh đối chứng ba cấu trúc dung hợp kinh điển trong học máy hiện đại:

```
        CÁC CHIẾN LƯỢC DUNG HỢP ĐA PHƯƠNG THỨC (MULTIMODAL FUSION ARCHITECTURES)

1. Early Fusion (Feature Concatenation):
   I_t ──► [CNN Backbone] ─────► z_vis ─┐
   H_t ──► [1D-CNN / MLP] ─────► z_bio ─┼──► [Concatenate] ──► [Joint MLP Classifier] ──► y_hat
   s_t ────────────────────────► s_t   ─┘

2. Late Fusion (Decision-Level Soft Voting / Bayesian Fusion):
   I_t ──► [Vision Net] ───────► P_vis(y) ─┐
   H_t ──► [Bio-Signal Net] ───► P_bio(y) ─┼──► [Weighted Fusion / Logic Gates] ────────► y_hat
   s_t ────────────────────────► Trigger ──┘

3. Intermediate Fusion (Cross-Modal Attention Transformer):
   I_t ──► [Visual Feature Map]  ────── (Keys & Values K_v, V_v) ┐
                                                                 ├──► [Cross-Attention] ──► y_hat
   H_t ──► [Bio-Temporal Embeddings] ── (Queries Q_b) ───────────┘
```

#### Chiến lược 1: Dung hợp sớm (Early Fusion - Feature-level Fusion)
Trích xuất vector nhúng đặc trưng thị giác $\mathbf{z}_{\text{vis}} = \phi_{\text{vis}}(\mathbf{I}_t) \in \mathbb{R}^{d_1}$ và vector đặc trưng sinh trắc học $\mathbf{z}_{\text{bio}} = \phi_{\text{bio}}(\mathbf{H}_t) \in \mathbb{R}^{d_2}$. Vector đặc trưng hợp nhất được tạo thành bằng phép nối chuỗi:
$$\mathbf{z}_{\text{joint}} = \left[ \mathbf{z}_{\text{vis}} \;\|\; \mathbf{z}_{\text{bio}} \;\|\; s_t \right] \in \mathbb{R}^{d_1 + d_2 + 1}$$
Dự đoán cuối cùng được tính qua mạng nơ-ron truyền thẳng phân loại:
$$\mathbf{p}_t = \text{Softmax}\left( \mathbf{W}_f \mathbf{z}_{\text{joint}} + \mathbf{b}_f \right)$$

#### Chiến lược 2: Dung hợp muộn (Late Fusion - Decision-level Fusion)
Mỗi phương thức được huấn luyện bằng một mô hình chuyên biệt độc lập để sinh ra phân phối xác suất riêng:
$$\mathbf{p}_{\text{vis}} = \text{Softmax}(g_{\text{vis}}(\mathbf{I}_t)), \quad \mathbf{p}_{\text{bio}} = \text{Softmax}(g_{\text{bio}}(\mathbf{H}_t))$$
Quyết định cuối cùng được tổng hợp bằng cơ chế trung bình có trọng số thích nghi kết hợp luật ghi đè ưu tiên an ninh (*Priority Override Rule*):
$$\mathbf{p}_t = \begin{cases}
[0, 0, 1]^\top & \text{nếu } s_t = 1 \\
w_{\text{vis}} \mathbf{p}_{\text{vis}} + (1 - w_{\text{vis}}) \mathbf{p}_{\text{bio}} & \text{nếu } s_t = 0
\end{cases}$$

#### Chiến lược 3: Dung hợp tầng trung gian dựa trên Cơ chế Chú ý Chéo (Cross-Modal Attention Fusion)
Sử dụng các đặc trưng chuỗi thời gian sinh lý học đóng vai trò làm truy vấn (*Query* $\mathbf{Q}_{\text{bio}}$) để rà quét và trích chọn các vùng đặc trưng không gian quan trọng trên bản đồ thị giác (*Key* $\mathbf{K}_{\text{vis}}$, *Value* $\mathbf{V}_{\text{vis}}$):
$$\mathbf{Z}_{\text{cross}} = \text{Attention}(\mathbf{Q}_{\text{bio}}, \mathbf{K}_{\text{vis}}, \mathbf{V}_{\text{vis}}) = \text{Softmax}\left( \frac{\mathbf{Q}_{\text{bio}} \mathbf{K}_{\text{vis}}^\top}{\sqrt{d_k}} \right) \mathbf{V}_{\text{vis}}$$
Cơ chế này cho phép khi nhịp tim tài xế thay đổi bất thường, mô hình thị giác sẽ tự động "dồn sự chú ý" vào các vùng không gian có khả năng chứa hung khí hoặc vị trí nón bảo hiểm bị gục.

---

## CHƯƠNG 4: HÀM MỤC TIÊU TỐI ƯU HÓA & XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU CỰC ĐỘ

### 4.1. Hàm mất mát Focal Loss cho bài toán phân loại đa lớp (Multi-Class Focal Loss)
Hàm mất mát Cross-Entropy tiêu chuẩn xem trọng số của mọi mẫu dữ liệu là như nhau. Trong bài toán của nhóm, các mẫu "Bình thường" áp đảo số lượng và rất dễ học, khiến tổng Gradient bị chi phối hoàn toàn bởi các mẫu dễ (*easy negative examples*). Để khắc phục, nhóm áp dụng **Focal Loss** (*Lin et al., 2017*):
$$\mathcal{L}_{\text{FL}}(\mathbf{p}_t, y) = -\sum_{c=0}^2 \alpha_c (1 - p_{t, c})^\gamma \log(p_{t, c}) \cdot \mathbb{I}(y = c)$$

Trong đó:
* $p_{t, c}$: Xác suất mô hình dự đoán mẫu $t$ thuộc về lớp $c$.
* $(1 - p_{t, c})^\gamma$: Hệ số điều biến (*Modulating Factor*). Khi một mẫu dễ phân loại có $p_{t, c} \to 1$, hệ số này triệt tiêu về 0, làm giảm trọng số gradient của mẫu đó. Ngược lại, các mẫu khó phân loại hoặc mẫu thiểu số có $p_{t, c} \to 0$ sẽ được giữ nguyên trọng số học tập.
* $\gamma \ge 0$: Tham số tập trung (*Focusing Parameter*), được tinh chỉnh thực nghiệm trong dải $\gamma \in [1.5, 3.0]$.
* $\alpha_c$: Trọng số cân bằng tần suất lớp (*Class-balancing weight*), thỏa mãn $\sum_{c=0}^2 \alpha_c = 1$.

### 4.2. Hàm mất mát cân bằng lớp dựa trên số mẫu hiệu dụng (Class-Balanced Loss - CB Loss)
Theo lý thuyết của *Cui et al. (CVPR 2019)*, khi số lượng mẫu của một lớp tăng lên, lượng thông tin gia tăng cận biên sẽ suy giảm do dữ liệu bị trùng lặp không gian. Nhóm nghiên cứu thiết lập hàm mất mát cân bằng dựa trên thể tích mẫu hiệu dụng $E_{n_c} = \frac{1 - \beta^{n_c}}{1 - \beta}$:
$$\mathcal{L}_{\text{CB}}(\mathbf{p}_t, y) = -\frac{1 - \beta}{1 - \beta^{n_y}} \left( 1 - p_{t, y} \right)^\gamma \log(p_{t, y})$$
Với $n_y$ là tổng số mẫu của lớp $y$ trong tập huấn luyện, và $\beta \in [0.9, 0.9999]$ là siêu tham số kiểm soát mức độ bão hòa thông tin.

### 4.3. Mô hình hóa Rủi ro Nhạy cảm Chi phí (Cost-Sensitive Risk Formulation)
Trong bài toán bảo vệ an toàn tính mạng, sai lầm của mô hình học máy mang tính bất đối xứng nghiêm trọng:
* **Sai lầm loại I (False Alarm / False Positive):** Mô hình dự đoán nhầm tài xế đang ngủ gật trong khi họ chỉ cúi đầu nhìn đồng hồ xe máy. Hậu quả: Cảnh báo rung hoặc tiếng bíp ngắn gây chút phiền toái.
* **Sai lầm loại II (Missed Detection / False Negative):** Tài xế ngủ gật thật hoặc đang bị kề dao siết cổ nhưng mô hình dự đoán là "Bình thường". Hậu quả: Tai nạn thảm khốc hoặc mất tính mạng.

Do đó, nhóm thiết lập **Ma trận Chi phí Sai lầm $\mathbf{C} \in \mathbb{R}^{3 \times 3}$** với quy ước $C_{i, j}$ là chi phí gánh chịu khi nhãn thực tế là $i$ nhưng mô hình dự đoán là $j$:
$$\mathbf{C} = \begin{pmatrix}
C_{00} & C_{01} & C_{02} \\
C_{10} & C_{11} & C_{12} \\
C_{20} & C_{21} & C_{22}
\end{pmatrix} = \begin{pmatrix}
0 & 1 & 2 \\
\mathbf{50} & 0 & 5 \\
\mathbf{200} & 20 & 0
\end{pmatrix}$$

Mục tiêu học máy trở thành bài toán tối thiểu hóa Rủi ro Kỳ vọng Bayes (*Bayes Expected Risk Minimization*):
$$\mathcal{R}_{\text{cost}}(f) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \sum_{c=0}^2 C_{y, c} \cdot \mathbb{I}(f(\mathbf{x}) = c) \right] \longrightarrow \min$$

Tại pha suy luận (*Inference Phase*), quyết định nhãn tối ưu được xác định theo quy tắc quyết định Bayes tối thiểu hóa rủi ro có điều kiện:
$$\hat{y}^* = \arg\min_{k \in \{0, 1, 2\}} \sum_{j=0}^2 C_{j, k} \cdot P(y = j \mid \mathbf{x})$$

---

## CHƯƠNG 5: KIẾN TRÚC TẬP DỮ LIỆU & QUY TRÌNH KỸ THUẬT DỮ LIỆU (DATA ENGINEERING)

### 5.1. Thành phần dữ liệu học máy đa nguồn

Nhóm xây dựng một tập dữ liệu gán nhãn đa phương thức kết hợp giữa dữ liệu chuẩn quốc tế (*Benchmark Datasets*) và dữ liệu thực tế thu thập tại Việt Nam (*In-the-wild Local Datasets*):

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                         CẤU TRÚC TẬP DỮ LIỆU HỌC MÁY ĐA PHƯƠNG THỨC                         |
+─────────────────────────────────────────────────────────────────────────────────────────────+
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             ▼                                 ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
|  NHÁNH NGỦ GẬT BỊT MẶT  |       |   NHÁNH HUNG KHÍ & CƯỚP |       |   NHÁNH SINH TRẮC HỌC   |
|   (NTHU-DDD / UTA-RLDD) |       |   (Roboflow Universe)   |       |   (WESAD / Smart Band)  |
| - Kính râm, khẩu trang  |       | - Dao, hung khí sắc nhọn|       | - Nhịp tim PPG (1 Hz)   |
| - Tư thế mũ bảo hiểm    |       | - Hành vi khống chế     |       | - Stress, Micro-sleep   |
| - Góc nghiêng gục đầu   |       | - 4.200 ảnh chú thích   |       | - 15 đối tượng y tế     |
└────────────┬────────────┘       └────────────┬────────────┘       └────────────┬────────────┘
             │                                 │                                 │
             └─────────────────────────────────┼─────────────────────────────────┘
                                               ▼
                              ┌─────────────────────────────────┐
                              |  TẬP THỰC NGHIỆM ĐƯỜNG PHỐ VN   |
                              | - 1.200 ảnh nón BH thực tế      |
                              | - Log nhịp tim Xiaomi Mi Band   |
                              | - Đo đạc rung lắc & ánh sáng    |
                              └─────────────────────────────────┘
```

1. **Bộ dữ liệu thị giác Ngủ gật qua Mũ & Khẩu trang (Drowsiness Visual Dataset):**
   * **NTHU-DDD (National Tsing Hua University Drowsy Driver Dataset):** Trích xuất các chuỗi video tài xế đeo kính râm, đeo khẩu trang trong các điều kiện ánh sáng ngày và đêm. Tập trung cắt và gán nhãn góc nghiêng nón bảo hiểm ($\theta < 20^\circ$: Tỉnh táo, $\theta \ge 25^\circ$: Ngủ gật).
   * Bổ sung tập dữ liệu **UTA-RLDD (University of Texas at Arlington Real-Life Drowsiness Dataset)** gồm các đoạn video đa dạng góc nhìn.
2. **Bộ dữ liệu thị giác Hung khí & Hành vi Tấn công (Threat & Weapon Detection Dataset):**
   * Khai thác tập dữ liệu mở **Knife & Weapon Detection Dataset** trên Roboflow Universe với hơn 4.200 ảnh gán nhãn bounding-box hung khí nguy hiểm (dao nhọn, mã tấu, vật kim loại vươn lên từ phía sau).
   * Gán nhãn bổ sung hành vi người ngồi sau vươn tay qua cổ người lái xe (*Choking / Rear Assault Gestures*).
3. **Bộ dữ liệu Chuỗi thời gian Sinh trắc học (Physiological Bio-Signal Dataset):**
   * **WESAD (Wearable Stress and Affect Detection - University of Freiburg):** Tập dữ liệu sinh lý học đa cảm biến (PPG, ECG, EDA, TEMP, ACC) ghi nhận từ 15 đối tượng trong các trạng thái: Thư giãn tĩnh (*Baseline*), Căng thẳng hoảng loạn (*Stress / Panic*) và Buồn ngủ lơ mơ. Trích xuất chuỗi nhịp tim tương thích với chuẩn đầu ra của Smartband.
4. **Bộ dữ liệu thực nghiệm đường phố Việt Nam (Local Validation Set):**
   * Nhóm tự ghi hình và thu thập 1.200 khung hình tài xế xe ôm công nghệ tại TP.HCM (đội mũ bảo hiểm các loại: nửa đầu, 3/4, nón grab có kính chống giọt bắn).
   * Ghi log thực tế chuỗi nhịp tim thời gian thực từ thiết bị đeo **Xiaomi Smart Band** qua kết nối BLE trong các điều kiện di chuyển ngoài đường phố.

### 5.2. Bảng tổng hợp quy mô và phân bố mẫu dữ liệu

| Phân lớp ($c$) | Tên trạng thái | Nguồn dữ liệu chính | Định dạng mẫu | Quy mô số mẫu ($N$) | Tỷ lệ phân bố |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **0** | **Bình thường (Normal)** | NTHU-DDD + WESAD Baseline + Thực nghiệm | Cặp `(Ảnh 96x96, HR-PPG)` | $4.200$ mẫu | **$70.0\%$** |
| **1** | **Ngủ gật (Drowsiness)** | NTHU-DDD Occluded + WESAD Drowsy + Thực nghiệm | Cặp `(Ảnh 96x96, HR-PPG)` | $1.200$ mẫu | **$20.0\%$** |
| **2** | **Tấn công (Assault)** | Roboflow Weapon + WESAD Panic + Diễn tập | Cặp `(Ảnh 96x96, HR-PPG)` | $600$ mẫu | **$10.0\%$** |
| **TỔNG** | — | — | — | **$6.000$ mẫu** | **$100.0\%$** |

### 5.3. Chiến lược Tiền xử lý & Tăng cường Dữ liệu thích nghi (Adaptive Data Augmentation)
Để mô hình học máy đạt khả năng tổng quát hóa cao (*High Generalization*) trên điều kiện đường sá khắc nghiệt tại Việt Nam, nhóm thiết kế pipeline tăng cường dữ liệu đặc thù:

* **Tăng cường miền thị giác (Visual Space Augmentations):**
  * *Motion Blur (Làm mờ chuyển động):* Áp dụng bộ lọc chuyển động ngẫu nhiên với kernel kích thước $5 \times 5$ đến $9 \times 9$ theo các góc nghiêng để mô phỏng độ rung giật của ghi-đông xe máy.
  * *Random Lighting & Shadow (Chỉnh sáng thích nghi):* Thay đổi độ sáng ngẫu nhiên ($\pm 40\%$) và tạo các dải bóng râm ngẫu nhiên cắt ngang mũ bảo hiểm để mô phỏng bóng cây râm mát và bóng đèn đường đêm muộn.
  * *Perspective Transform:* Phép biến đổi phối cảnh góc nhìn ngẫu nhiên $\pm 15^\circ$ mô phỏng việc camera gắn trên chân gương xe máy bị lệch góc nhẹ trong quá trình sử dụng.
* **Tăng cường miền chuỗi thời gian sinh học (Bio-Temporal Augmentations):**
  * *Additive Gaussian Noise:* Cộng nhiễu ngẫu nhiên $\epsilon \sim \mathcal{N}(0, \sigma^2)$ vào chuỗi nhịp tim để mô phỏng sai số rung lắc cơ học của cảm biến quang học trên cổ tay tài xế.
  * *Time-Warping & Jittering:* Kéo giãn hoặc nén ngẫu nhiên trục thời gian theo hệ số $\alpha \in [0.85, 1.15]$ để mô phỏng sự khác biệt về tốc độ phản xạ sinh lý giữa các cá nhân khác nhau.

---

## CHƯƠNG 6: KHUNG KIẾN TRÚC MÔ HÌNH HỌC MÁY ĐỀ XUẤT (PROPOSED ML ARCHITECTURES)

Nhóm nghiên cứu thiết kế hai tầng mô hình: Mô hình đối chuẩn (*Baseline Benchmark*) và Khung mô hình học sâu đa phương thức đề xuất (*Proposed Multimodal Deep Learning Model*).

### 6.1. Mô hình đối chuẩn (Baselines)
1. **LightGBM / XGBoost trên Đặc trưng Trích xuất Thủ công (Hand-crafted Feature Baseline):**
   * *Đặc trưng ảnh:* Trích xuất Histogram of Oriented Gradients (HOG) trên vùng nón bảo hiểm và Local Binary Patterns (LBP).
   * *Đặc trưng nhịp tim:* 8 chỉ số thống kê HRV ($\overline{hr}, \sigma_{hr}, \text{SDNN}, \text{RMSSD}, \Delta hr, \max, \min, \text{Skewness}$).
   * Mô hình: Cây quyết định tăng cường gradient (*Gradient Boosted Decision Trees*) với cấu hình `max_depth=6`, `learning_rate=0.05`.
2. **Mô hình Thị giác Đơn lẻ (Vision-Only ResNet-18 / MobileNetV3):**
   * Chỉ sử dụng nhánh ảnh để dự đoán nhãn trạng thái mà không có sự tham gia của nhịp tim, dùng làm đối chứng nhằm đo lường giá trị gia tăng của việc dung hợp đa phương thức.

### 6.2. Khung mô hình Đa phương thức đề xuất (Proposed Cross-Attention Multimodal Network)
Kiến trúc tổng thể gồm 3 phân hệ liên hoàn:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────────+
|                       KIẾN TRÚC MÔ HÌNH HỌC SÂU ĐA PHƯƠNG THỨC ĐỀ XUẤT                               |
+───────────────────────────────────────────────────────────────────────────────────────────────────────+

  [NHÁNH THỊ GIÁC]
   Ảnh đầu vào I_t (3x96x96)
          │
          ▼
   [MobileNetV3 Backbone] ──► Feature Map F_vis in R^{C' x H' x W'} ──► Linear Projection ──► Keys/Values (K_v, V_v)
                                                                                                    │
  [NHÁNH SINH TRẮC HỌC]                                                                             │
   Chuỗi nhịp tim H_t (1xT)                                                                         │
          │                                                                                         │
          ▼                                                                                         │
   [1D-CNN + BiLSTM]      ──► Bio Embedding z_bio in R^d ─────────────► Linear Projection ──► Queries (Q_b)
                                                                                                    │
                                                                                                    ▼
                                                                                   [CROSS-MODAL ATTENTION BLOCK]
                                                                                   Attn = Softmax(Q_b * K_v^T / sqrt(d)) * V_v
                                                                                                    │
                                                                                                    ▼
                                                                                   [FUSED CONTEXT VECTOR z_fused]
                                                                                                    │
                                                   Nút bấm SOS s_t ─────────────────────────────────┤
                                                                                                    ▼
                                                                                        [CLASSIFICATION HEAD MLP]
                                                                                                    │
                                                                                                    ▼
                                                                                     Output Probabilities: [p_0, p_1, p_2]
```

* **Trích xuất đặc trưng thị giác (Visual Backbone):** Sử dụng phiên bản rút gọn của MobileNetV3 kết hợp các khối Inverted Residual và Squeeze-and-Excitation (SE), tối ưu hóa việc học các cạnh nghiêng của nón bảo hiểm và các hình thái hung khí.
* **Trích xuất chuỗi thời gian (Temporal Bio-Signal Head):** Mạng tích chập 1 chiều (1D-CNN) để lọc nhiễu tần số cao, kết hợp mạng nơ-ron hồi quy hai chiều (Bidirectional LSTM) gồm 32 hidden units để bắt giữ xu hướng tăng vọt hay suy giảm dài hạn của nhịp tim.
* **Khối Dung hợp Tương tác Chú ý (Cross-Modal Fusion Block):** Tích hợp thông tin chéo giúp mạng nhận thức được: "Khi nhịp tim có biểu hiện bất thường, hãy tập trung soi kỹ vùng không gian nghi vấn trên khung hình".

---

## CHƯƠNG 7: PHƯƠNG PHÁP THẨM ĐỊNH KHOA HỌC & HỆ CHỈ SỐ ĐÁNH GIÁ (EVALUATION PROTOCOL)

### 7.1. Chiến lược phân chia dữ liệu độc lập đối tượng (Stratified Subject-Independent K-Fold Cross-Validation)
Một trong những cạm bẫy lớn nhất trong các bài toán học máy sinh trắc học và DMS là **Hiện tượng rò rỉ dữ liệu (*Data Leakage*)** khi các khung hình hoặc chuỗi nhịp tim của cùng một người xuất hiện ở cả tập Train và tập Test. Khi đó, mô hình thực chất chỉ "học vẹt" đặc điểm nhận dạng khuôn mặt hay nhịp tim cơ sở của cá nhân đó thay vì học được quy luật tổng quát của hiện tượng ngủ gật hay bị tấn công.

Để đảm bảo tính khách quan và chuẩn mực học thuật cao nhất, đề tài áp dụng phương pháp:
* **Stratified Subject-Independent Group K-Fold Cross-Validation ($K = 5$):** Toàn bộ dữ liệu của một đối tượng thử nghiệm chỉ được nằm hoàn toàn trong tập Train hoặc hoàn toàn trong tập Validation/Test tại mỗi Fold.
* **Tỷ lệ phân chia:** $80\%$ dữ liệu (ứng với đối tượng tập huấn luyện) dùng để huấn luyện mô hình qua 5 Folds, và $20\%$ độc lập hoàn toàn dùng làm tập kiểm thử cuối cùng (*Held-out Test Set*).

### 7.2. Hệ chỉ số đo lường hiệu năng chuyên sâu (Comprehensive Evaluation Metrics)
Do dữ liệu mang tính mất cân bằng lớn, chỉ số *Accuracy* thông thường hoàn toàn vô nghĩa. Nhóm thiết lập hệ chỉ số toàn diện sau:

1. **Độ nhạy từng lớp (Recall / True Positive Rate per Class):**
   $$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
   *Yêu cầu đặc biệt:* $\text{Recall}_{\text{Drowsy}} \ge 95\%$ và $\text{Recall}_{\text{Assault}} \ge 98\%$ nhằm triệt tiêu tối đa nguy cơ bỏ sót tai nạn và cướp bóc.
2. **Độ chuẩn xác từng lớp (Precision per Class):**
   $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$
   *Yêu cầu đặc biệt:* $\text{Precision}_{\text{Drowsy}} \ge 90\%$ để không làm phiền tài xế khi họ đang tập trung lái xe.
3. **Chỉ số Macro-F1 Score & Weighted-F1 Score:**
   $$\text{Macro-F1} = \frac{1}{3}\sum_{c=0}^2 \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
   Chỉ số Macro-F1 đối xử bình đẳng giữa lớp thiểu số (Tấn công) và lớp đa số (Bình thường), là thước đo chuẩn xác nhất cho bài toán mất cân bằng.
4. **Diện tích dưới đường cong Precision-Recall (PR-AUC):**
   Đánh giá sự cân bằng giữa Precision và Recall trên mọi ngưỡng cắt xác suất ($\tau \in [0, 1]$), đặc biệt hữu hiệu cho lớp dữ liệu cực hiếm ($y = 2$).
5. **Ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix):**
   Trực quan hóa tỷ lệ phần trăm phân loại đúng và sai lệch giữa các cặp trạng thái, làm rõ các trường hợp nhầm lẫn giữa cúi đầu và ngủ gật thực sự.

---

## CHƯƠNG 8: TÍNH MINH BẠCH & KHẢ NĂNG GIẢI THÍCH MÔ HÌNH (EXPLAINABLE AI - XAI)

Để đáp ứng tiêu chuẩn học thuật khắt khe của một đồ án Trí tuệ Nhân tạo nâng cao, mô hình không được phép là một "hộp đen" bí ẩn (*Black-box Model*). Nhóm tích hợp hai kỹ thuật XAI hàng đầu thế giới:

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                        CƠ CHẾ GIẢI THÍCH HỌC MÁY (EXPLAINABLE AI PIPELINE)                  |
+─────────────────────────────────────────────────────────────────────────────────────────────+
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             ▼                                                                   ▼
┌─────────────────────────────────────────┐         ┌─────────────────────────────────────────┐
|     GRAD-CAM (NHÁNH THỊ GIÁC MÁY TÍNH)  |         |     SHAP VALUES (NHÁNH SINH TRẮC HỌC)   |
| - Tính đạo hàm lớp dự đoán theo Feature |         | - Giá trị đóng góp Shapley của từng     |
| - Trực quan hóa bản đồ nhiệt (Heatmap)  |         |   chỉ số biến thiên nhịp tim HRV        |
| - Chứng minh mô hình soi đúng nón bảo   |         | - Phân định rõ ràng ảnh hưởng của SDNN, |
|   hiểm bị gục hoặc hung khí phía sau    |         |   RMSSD và tốc độ sụt giảm nhịp tim     |
└─────────────────────────────────────────┘         └─────────────────────────────────────────┘
```

### 8.1. Grad-CAM cho Nhánh Thị giác Máy tính (Gradient-weighted Class Activation Mapping)
* Tính toán đạo hàm của điểm số phân loại $y^c$ đối với các bản đồ kích hoạt đặc trưng $A^k$ của lớp tích chập cuối cùng:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
  Bản đồ nhiệt nhiệt lượng tương ứng:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
* **Ý nghĩa thực nghiệm:** Chứng minh rằng mô hình đưa ra quyết định "Ngủ gật" vì vùng đỉnh mũ bảo hiểm bị chúc xuống dưới chứ không phải do màu sắc áo khoác hay cảnh vật nền đường xá bên cạnh. Tương tự, khi kích hoạt nhãn "Tấn công", bản đồ nhiệt phải sáng rực tại vị trí tay cầm hung khí hoặc cánh tay người ngồi sau đang chồm lên phía trước.

### 8.2. Phân tích giá trị đóng góp Shapley (SHAP Values cho Tín hiệu Sinh trắc học)
* Sử dụng lý thuyết trò chơi hợp tác (*Shapley Values*) để định lượng mức độ đóng góp cận biên của từng thuộc tính sinh lý học ($\Delta hr_t$, SDNN, RMSSD) vào việc kéo xác suất dự đoán về phía nhãn "Ngủ gật" hay "Bình thường":
  $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$
* **Ý nghĩa thực nghiệm:** Khi tài xế cúi đầu nhìn đồng hồ xe trong $1$ giây, góc nghiêng mũ có thể vượt $25^\circ$, nhưng giá trị SHAP của nhánh nhịp tim sẽ mang giá trị âm kìm hãm lại vì nhịp tim vẫn duy trì mức tỉnh táo ($85$ BPM), ngăn chặn triệt để hiện tượng còi hú báo động giả.

---

## CHƯƠNG 9: KẾ HOẠCH NGHIÊN CỨU & PHÂN CÔNG NHIỆM VỤ CHI TIẾT (WORK PLAN)

### 9.1. Lộ trình thực hiện đồ án 8 tuần

```
Tuần 1-2 (27/10 - 08/11/2026) : Khảo sát y văn, định nghĩa bài toán hình thức, chuẩn bị dữ liệu NTHU-DDD & WESAD
Tuần 3-4 (09/11 - 22/11/2026) : Tiền xử lý dữ liệu, Data Augmentation, xây dựng Baseline Models (LightGBM & ResNet)
Tuần 5-6 (23/11 - 06/12/2026) : Xây dựng mô hình Dung hợp Đa phương thức (Early/Late/Cross-Attention), tinh chỉnh Focal Loss
Tuần 7   (07/12 - 13/12/2026) : Đánh giá Stratified K-Fold, phân tích Confusion Matrix, tích hợp XAI (Grad-CAM & SHAP)
Tuần 8   (14/12 - 20/12/2026) : Tổng hợp kết quả thực nghiệm, hoàn thiện báo cáo khoa học và chuẩn bị slide bảo vệ
```

### 9.2. Ma trận phân công trách nhiệm nhóm (RACI Matrix)
* **R (Responsible):** Người trực tiếp thực hiện chính.
* **A (Accountable):** Người chịu trách nhiệm phê duyệt và kết quả cuối cùng.
* **C (Consulted):** Người được tham vấn chuyên môn.
* **I (Informed):** Người được thông báo tiến độ.

| Hạng mục công việc chuyên môn | Trần Hoàng Hôn (Nhóm trưởng) | Nguyễn Duy Khang | Phạm Thành Trung | Vũ Văn Duy | Hoàng Cao Sơn |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mô hình hóa toán học & Thiết kế kiến trúc tổng thể** | **A / R** | C | C | C | I |
| **Thu thập & Tiền xử lý dữ liệu ảnh nón bảo hiểm** | I | **A / R** | I | C | R |
| **Xử lý chuỗi thời gian nhịp tim PPG & Tính toán HRV** | I | I | **A / R** | C | R |
| **Cài đặt Baseline Models (LightGBM / XGBoost)** | C | R | R | **A / R** | I |
| **Cài đặt Mạng học sâu Đa phương thức (Cross-Attention)** | **A / R** | R | R | C | I |
| **Tối ưu hóa Hàm mất mát Focal Loss & Cost Matrix** | C | I | I | **A / R** | I |
| **Thực hiện Thẩm định Stratified 5-Fold Cross Validation** | C | C | C | **A / R** | R |
| **Triển khai Explainable AI (Grad-CAM & SHAP)** | C | R | R | C | **A / R** |
| **Thu thập tập kiểm thử đường phố thực tế tại TP.HCM** | R | R | R | R | **A / R** |
| **Viết bản Thuyết minh bài toán & Báo cáo tổng kết** | **A / R** | R | R | R | R |

---

## CHƯƠNG 10: TÀI LIỆU THAM KHẢO HỌC THUẬT (ACADEMIC REFERENCES)

1. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). *Focal Loss for Dense Object Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 42(2), 318-327.
2. **Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S.** (2019). *Class-Balanced Loss Based on Effective Number of Samples*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9268-9277.
3. **Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K.** (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*. In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI), pp. 400-408.
4. **Weng, C. H., Lai, Y. H., & Lai, S. H.** (2016). *Driver Drowsiness Detection via a Smartphone Platform*. In IEEE International Conference on Computer Vision Workshops (ICCVW), pp. 1-8 (NTHU-DDD Dataset).
5. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.** (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618-626.
6. **Lundberg, S. M., & Lee, S. I.** (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 30), pp. 4765-4774 (SHAP).
7. **Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H.** (2019). *Searching for MobileNetV3*. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314-1324.
8. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I.** (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 30), pp. 5998-6008.
9. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.
10. **Elkan, C.** (2001). *The Foundations of Cost-Sensitive Learning*. In International Joint Conference on Artificial Intelligence (IJCAI), Vol. 17, No. 1, pp. 973-978.

---
*Tài liệu được biên soạn hoàn chỉnh phục vụ đánh giá học thuật và triển khai đồ án môn học Máy học nâng cao - Giảng viên ThS./TS. Đặng Việt Dũng.*
