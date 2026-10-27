## 📝 Mô tả Thay đổi (Description)
<!-- Tóm tắt ngắn gọn các thay đổi trong PR này -->
- **Nội dung thực hiện:** 
- **Liên quan đến Issue / Task:** Fixes # / Relates to #
- **Thành viên phụ trách:** @

## 🔬 Phân loại Thay đổi (Type of Change)
- [ ] 🚀 `feat`: Tính năng hoặc module mới
- [ ] 🐛 `fix`: Sửa lỗi bug hoặc pipeline
- [ ] 📚 `docs`: Cập nhật tài liệu, README, báo cáo Word
- [ ] 🧪 `test`: Thêm kiểm thử đơn vị hoặc test cases
- [ ] ⚡ `perf`: Tối ưu hóa hiệu năng, giảm latency / RAM
- [ ] 🔨 `refactor`: Tái cấu trúc mã nguồn

## 📊 Phân hệ Chuyên môn (Subsystem)
- [ ] 👁️ Computer Vision (MobileNetV3, Augmentation, NTHU-DDD, Roboflow)
- [ ] 💓 Bio-Signals & Time-Series (PPG, HRV, WESAD, BiLSTM)
- [ ] 🧠 ML Modeling & Fusion (Cross-Attention, Early/Late Fusion, Focal Loss)
- [ ] 📈 Evaluation & XAI (K-Fold, PR-AUC, Confusion Matrix, Grad-CAM, SHAP)
- [ ] 📄 Documentation & Báo cáo đồ án

## ✅ Bảng Kiểm tra Chất lượng (Quality Checklist)
- [ ] Mã nguồn tuân thủ quy chuẩn PEP 8 và đã bổ sung Type Annotations.
- [ ] Tất cả các hàm mới đều có docstring giải thích tham số và kiểu trả về.
- [ ] Đã chạy toàn bộ unit tests cục bộ (`pytest tests/`) và đều pass 100%.
- [ ] **Kiểm tra rò rỉ dữ liệu (No Data Leakage):** Không để dữ liệu của cùng một đối tượng xuất hiện ở cả tập Train và Validation.
- [ ] **Không commit file dữ liệu nặng (>50MB)** hoặc checkpoints trọng số chưa nén.
- [ ] Đã cập nhật tài liệu hoặc notebook tương ứng (nếu có thay đổi cấu trúc).

## 💬 Ghi chú Thêm cho Reviewer (Additional Notes)
<!-- Các lưu ý hoặc hướng dẫn kiểm thử cho người review -->
