import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def create_plan_document():
    doc = docx.Document()

    # Academic margins
    for section in doc.sections:
        section.top_margin = Inches(0.79)
        section.bottom_margin = Inches(0.79)
        section.left_margin = Inches(1.18)
        section.right_margin = Inches(0.79)

    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    style_normal.paragraph_format.line_spacing = 1.2
    style_normal.paragraph_format.space_after = Pt(6)

    NAVY = RGBColor(0x1B, 0x36, 0x5D)
    BLUE = RGBColor(0x2E, 0x75, 0xB6)
    DARK = RGBColor(0x22, 0x22, 0x22)
    GRAY = RGBColor(0x55, 0x55, 0x55)

    def set_cell_background(cell, fill_hex):
        shading_xml = f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'
        cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

    def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
            node = OxmlElement(f'w:{m}')
            node.set(qn('w:w'), str(val))
            node.set(qn('w:type'), 'dxa')
            tcMar.append(node)
        tcPr.append(tcMar)

    def set_table_borders(table, color="D3D3D3"):
        tblPr = table._tbl.tblPr
        borders_xml = f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="1B365D"/>
            <w:right w:val="none"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
        '''
        tblPr.append(parse_xml(borders_xml))

    # Header
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst1 = p_inst.add_run("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN – ĐHQG-HCM\n")
    r_inst1.font.size = Pt(11)
    r_inst1.font.bold = True
    r_inst2 = p_inst.add_run("HỌC PHẦN: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)\n")
    r_inst2.font.size = Pt(11)
    r_inst2.font.bold = True
    r_inst2.font.color.rgb = BLUE
    p_inst.paragraph_format.space_after = Pt(14)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t1 = p_title.add_run("KẾ HOẠCH TRIỂN KHAI VÀ LỘ TRÌNH THỰC HIỆN ĐỒ ÁN (8 TUẦN)\n")
    r_t1.font.size = Pt(14)
    r_t1.font.bold = True
    r_t1.font.color.rgb = NAVY
    
    r_t2 = p_title.add_run("KHUNG HỌC MÁY ĐA PHƯƠNG THỨC GIÁM SÁT AN TOÀN TÀI XẾ XE HAI BÁNH\n")
    r_t2.font.size = Pt(15)
    r_t2.font.bold = True
    r_t2.font.color.rgb = NAVY

    r_t3 = p_title.add_run("Lộ trình 5 giai đoạn, Ma trận RACI 5 thành viên & Tiêu chuẩn nghiệm thu khoa học")
    r_t3.font.size = Pt(11)
    r_t3.font.italic = True
    r_t3.font.color.rgb = GRAY
    p_title.paragraph_format.space_after = Pt(14)

    # Info table
    info_table = doc.add_table(rows=6, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("Giảng viên hướng dẫn:", "ThS./TS. Đặng Việt Dũng"),
        ("Thời gian thực hiện:", "27/10/2026 – 20/12/2026 (8 Tuần)"),
        ("Sĩ số & Cơ cấu nhóm:", "Nhóm 5 sinh viên (Chuyên ngành Trí tuệ Nhân tạo)"),
        ("Trưởng nhóm điều phối:", "Trần Hoàng Hôn – MSSV: 26410046"),
        ("Các thành viên chuyên trách:", "Nguyễn Duy Khang (26410055), Phạm Thành Trung (26410141),\nVũ Văn Duy (26410031), Hoàng Cao Sơn (25730061)"),
        ("Mục tiêu điểm số kỳ vọng:", "9.5 – 10.0 (Đạt chuẩn học thuật công bố/ứng dụng)")
    ]
    for i, (k, v) in enumerate(info_data):
        row = info_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.3)
        c1.width = Inches(4.5)
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(10)
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(10)
        set_cell_background(c0, "F2F5F9")
        set_cell_background(c1, "FAFAFA")
        set_cell_margins(c0, 50, 50, 100, 100)
        set_cell_margins(c1, 50, 50, 100, 100)
    set_table_borders(info_table, "B0C4DE")
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    def add_h1(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.bold = True
        r.font.size = Pt(13)
        r.font.color.rgb = NAVY
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)

    def add_h2(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.bold = True
        r.font.size = Pt(11.5)
        r.font.color.rgb = BLUE
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)

    def add_p(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.2
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        r1 = p.add_run(bold_prefix)
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r2 = p.add_run(text)
        r2.font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15

    def add_callout(text, bg_color="F0F4F8", border_color="2E75B6"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        cell.width = Inches(6.8)
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, 80, 80, 140, 140)
        tcPr = cell._tc.get_or_add_tcPr()
        borders_xml = f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
        '''
        tcPr.append(parse_xml(borders_xml))
        p = cell.paragraphs[0]
        r = p.add_run(text)
        r.font.size = Pt(10)
        r.font.italic = True
        r.font.color.rgb = DARK
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 1. Mục tiêu
    add_h1("1. MỤC TIÊU VÀ BỐI CẢNH TRIỂN KHAI")
    add_p(
        "Kế hoạch hành động 8 tuần này nhằm bảo đảm nhóm 5 thành viên triển khai đồ án học máy nâng cao "
        "đúng tiến độ, giải quyết bài toán an toàn kép cho tài xế xe hai bánh: Nhận diện ngủ gật tức thì (Drowsiness) "
        "và phòng chống tấn công/cướp giật (Assault) trong điều kiện tài xế bịt kín khẩu trang, đeo kính râm."
    )
    add_bullet("Không gian phân loại mục tiêu: ", "Lớp 0 (Normal Driving), Lớp 1 (Drowsiness - đầu gục >=25°, nhịp tim giảm), Lớp 2 (Assault - người sau uy hiếp, nhịp tim tăng vọt, nút SOS).")
    add_bullet("Trọng tâm học máy: ", "Xử lý dữ liệu mất cân bằng cực độ (Focal Loss, CB-Loss), Dung hợp đa phương thức (Cross-Modal Attention), Thẩm định Stratified 5-Fold không rò rỉ dữ liệu, và Giải thích mô hình (Grad-CAM & SHAP).")

    # 2. Lộ trình 5 giai đoạn
    add_h1("2. LỘ TRÌNH 5 GIAI ĐOẠN CHI TIẾT (8 TUẦN)")

    phases = [
        ("Giai đoạn 1: Chuẩn bị Dữ liệu & Pipeline Tăng cường (Tuần 2-3: 27/10 – 08/11/2026)",
         [("Thu thập bộ dữ liệu đa nguồn: ", "Tải NTHU-DDD (tài xế đeo kính/khẩu trang), Roboflow Weapon (>4.200 ảnh hung khí), WESAD (tín hiệu nhịp tim PPG)."),
          ("Phân tích khám phá (EDA): ", "Xây dựng notebook 01_eda_and_data_preprocessing.ipynb, phân tích phân bố nhãn 70:20:10."),
          ("Chiến lược Data Augmentation: ", "Cài đặt Motion Blur (5x5, 9x9), ánh sáng ngẫu nhiên ±40%, bóng râm, cộng nhiễu Gaussian và Time-warping trên chuỗi nhịp tim."),
          ("Chia tập độc lập đối tượng: ", "Áp dụng StratifiedGroupKFold (K=5) với groups=subject_id, tách biệt 20% Held-out Test Set.")]),
        
        ("Giai đoạn 2: Baseline Models & Đặc trưng HRV (Tuần 4: 09/11 – 15/11/2026)",
         [("Trích xuất đặc trưng HRV: ", "Cài đặt src/features/hrv.py tính 8 chỉ số: mean HR, SDNN, RMSSD, pNN50, Delta hr."),
          ("Trích xuất đặc trưng ảnh thủ công: ", "HOG trên nón bảo hiểm và LBP."),
          ("Huấn luyện mô hình cơ sở: ", "LightGBM và XGBoost trên dữ liệu bảng trích xuất; MobileNetV3-Small trên ảnh thuần (Vision-only)."),
          ("Thiết lập cột mốc Benchmark: ", "Đo đạc F1-Score, Recall làm mốc so sánh với mô hình đa phương thức.")]),

        ("Giai đoạn 3: Khung Deep Multimodal & Custom Loss (Tuần 5-6: 16/11 – 29/11/2026)",
         [("Xây dựng 3 cơ chế Dung hợp: ", "Early Fusion (Concatenation), Late Fusion (Soft Voting + Priority Rule), và Cross-Modal Attention (Bio Query -> Visual Key/Value)."),
          ("Tích hợp Hàm mất mát tùy biến: ", "MultiClassFocalLoss, ClassBalancedFocalLoss và CostSensitiveBayesLoss (phạt nặng FN lớp nguy hiểm với chi phí C_20 = 200)."),
          ("Huấn luyện End-to-End: ", "AdamW, Cosine Annealing Scheduler, huấn luyện đồng thời qua 5 Folds và lưu checkpoints tốt nhất.")]),

        ("Giai đoạn 4: Thẩm định Chuyên sâu, Thực nghiệm & XAI (Tuần 7: 30/11 – 06/12/2026)",
         [("Hệ chỉ số nâng cao: ", "Vẽ đường cong PR-AUC cho từng lớp, tính toán Ma trận nhầm lẫn chuẩn hóa và tổng chi phí rủi ro Bayes."),
          ("Giải thích mô hình (XAI): ", "Grad-CAM trên MobileNetV3 (chứng minh soi đúng nón bảo hiểm và hung khí), SHAP values trên chuỗi nhịp tim (giải thích triệt tiêu báo động giả)."),
          ("Kiểm thử hiện trường: ", "Đánh giá khả năng tổng quát hóa trên tập thực nghiệm 1.200 ảnh nón bảo hiểm và log nhịp tim ngoài đường TP.HCM.")]),

        ("Giai đoạn 5: Hoàn thiện Báo cáo Khoa học & Bảo vệ (Tuần 8-9: 07/12 – 20/12/2026)",
         [("Quyển báo cáo khoa học >= 40 trang: ", "Hoàn thiện bản Word + PDF theo cấu trúc chuẩn UIT, trích dẫn tài liệu tham khảo IEEE/CVPR."),
          ("Sản phẩm số đi kèm: ", "Mã nguồn sạch có tài liệu tái lập kết quả, 4 Jupyter Notebooks thực nghiệm hoàn chỉnh."),
          ("Slide & Video Demo: ", "Slide thuyết trình bảo vệ cuối kỳ và clip quay thực tế mô hình nhận diện ngoài đường phố."),
          ("Bảo vệ trước Giảng viên: ", "Thuyết trình và phản biện đồ án trước ThS./TS. Đặng Việt Dũng.")])
    ]

    for p_title, tasks in phases:
        add_h2(p_title)
        for b_prefix, b_text in tasks:
            add_bullet(b_prefix, b_text)

    # 3. Ma trận RACI
    add_h1("3. MA TRẬN PHÂN CÔNG TRÁCH NHIỆM NHÓM (RACI MATRIX)")
    raci_table = doc.add_table(rows=7, cols=6)
    raci_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    r_headers = ["Hạng mục công việc", "Hôn", "Khang", "Trung", "Duy", "Sơn"]
    for j, h in enumerate(r_headers):
        cell = raci_table.cell(0, j)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        set_cell_background(cell, "1B365D")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 60, 60, 60, 60)

    raci_data = [
        ("Thiết kế kiến trúc học máy & Dung hợp Attention", "A/R", "C", "C", "C", "I"),
        ("Xử lý dữ liệu ảnh nón BH & Backbone MobileNetV3", "I", "A/R", "I", "C", "R"),
        ("Xử lý tín hiệu PPG & Trích xuất đặc trưng HRV", "I", "I", "A/R", "C", "R"),
        ("Tối ưu hóa Focal Loss, CB-Loss & Đánh giá 5-Fold", "C", "I", "I", "A/R", "I"),
        ("Triển khai XAI (Grad-CAM & SHAP) & Test TP.HCM", "C", "R", "R", "C", "A/R"),
        ("Tổng hợp Báo cáo khoa học >= 40 trang & Slide", "A/R", "R", "R", "R", "R")
    ]
    for i, rdata in enumerate(raci_data):
        row = raci_table.rows[i+1]
        for j, val in enumerate(rdata):
            cell = row.cells[j]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j == 0:
                set_cell_background(cell, "F2F5F9")
            elif "A/R" in val:
                r.font.bold = True
                set_cell_background(cell, "E3F2FD")
                r.font.color.rgb = NAVY
            else:
                set_cell_background(cell, "FAFAFA")
            set_cell_margins(cell, 45, 45, 60, 60)
    set_table_borders(raci_table, "B0C4DE")
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 4. Tiêu chuẩn nghiệm thu
    add_h1("4. TIÊU CHUẨN NGHIỆM THU KHOA HỌC (VERIFICATION CRITERIA)")
    add_bullet("Recall Drowsiness >= 95%: ", "Bắt buộc nhằm triệt tiêu tối đa rủi ro bỏ sót tai nạn do ngủ gật.")
    add_bullet("Recall Assault >= 98%: ", "Bắt buộc nhằm bảo đảm an toàn tính mạng khi có hung khí hoặc hành vi siết cổ.")
    add_bullet("Precision Drowsiness >= 90%: ", "Loại bỏ báo động giả khi tài xế cúi đầu quan sát bảng đồng hồ xe máy.")
    add_bullet("Macro-F1 Score >= 90%: ", "Thước đo trung bình công bằng trên bài toán mất cân bằng dữ liệu lớn.")
    add_bullet("Minh chứng XAI trực quan: ", "Grad-CAM soi sáng chính xác nón bảo hiểm cúi gục và hung khí; SHAP giải thích được vai trò kìm hãm báo động giả của nhịp tim.")
    add_bullet("Quyển báo cáo đồ án: ", "Báo cáo Word + PDF đầy đủ >= 40 trang theo đúng chuẩn cấu trúc môn học của UIT.")

    docs_output = r"D:\Trí tuệ nhân tạo\HK3\Máy học nâng cao\Đồ án\docs\Ke_Hoach_Thuc_Hien_Do_An.docx"
    root_output = r"D:\Trí tuệ nhân tạo\HK3\Máy học nâng cao\Đồ án\Ke_Hoach_Thuc_Hien_Do_An.docx"
    doc.save(docs_output)
    doc.save(root_output)
    print("Plan docx created successfully.")

if __name__ == "__main__":
    create_plan_document()
