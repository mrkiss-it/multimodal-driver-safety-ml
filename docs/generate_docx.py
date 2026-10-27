import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn
import os

def create_document():
    doc = docx.Document()

    # Page Margins: Standard Academic (Top 2cm, Bottom 2cm, Left 3cm, Right 2cm)
    for section in doc.sections:
        section.top_margin = Inches(0.79)
        section.bottom_margin = Inches(0.79)
        section.left_margin = Inches(1.18)
        section.right_margin = Inches(0.79)

    # Base Styles
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

    # --- TITLE / HEADER ---
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst1 = p_inst.add_run("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN – ĐHQG-HCM\n")
    r_inst1.font.size = Pt(11)
    r_inst1.font.bold = True
    r_inst2 = p_inst.add_run("HỌC PHẦN: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)\n")
    r_inst2.font.size = Pt(11)
    r_inst2.font.bold = True
    r_inst2.font.color.rgb = BLUE
    p_inst.paragraph_format.space_after = Pt(16)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t1 = p_title.add_run("BÁO CÁO XÁC ĐỊNH BÀI TOÁN & BẢN THUYẾT MINH KHOA HỌC ĐỀ TÀI\n")
    r_t1.font.size = Pt(13)
    r_t1.font.bold = True
    r_t1.font.color.rgb = NAVY
    
    r_t2 = p_title.add_run("KHUNG MÔ HÌNH HỌC MÁY ĐA PHƯƠNG THỨC NHẬN DIỆN TRẠNG THÁI RỦI RO AN TOÀN CHO TÀI XẾ XE HAI BÁNH DỰA TRÊN THỊ GIÁC MÁY TÍNH VÀ CHUỖI THỜI GIAN SINH TRẮC HỌC\n")
    r_t2.font.size = Pt(15)
    r_t2.font.bold = True
    r_t2.font.color.rgb = NAVY

    r_t3 = p_title.add_run("A Multimodal Machine Learning Framework for Two-Wheel Driver Safety Risk State Recognition Integrating Computer Vision and Physiological Bio-Signal Time-Series")
    r_t3.font.size = Pt(11)
    r_t3.font.italic = True
    r_t3.font.color.rgb = GRAY
    p_title.paragraph_format.space_after = Pt(14)

    # --- INFO BOX TABLE ---
    info_table = doc.add_table(rows=6, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("Giảng viên phụ trách học phần:", "ThS./TS. Đặng Việt Dũng"),
        ("Học kỳ áp dụng:", "Học kỳ 3 (Năm học 2026 – 2027)"),
        ("Nhóm sinh viên thực hiện:", "Nhóm 5 thành viên (Môn Máy học nâng cao)"),
        ("Trưởng nhóm / Kiến trúc học máy:", "Trần Hoàng Hôn – MSSV: 26410046"),
        ("Các thành viên chuyên trách:", "Nguyễn Duy Khang (26410055) – Computer Vision Lead\nPhạm Thành Trung (26410141) – Bio-Signal & Time-Series\nVũ Văn Duy (26410031) – ML Modeling & Evaluation Lead\nHoàng Cao Sơn (25730061) – Data Engineering & XAI"),
        ("Thời gian hoàn thành thuyết minh:", "27/10/2026")
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

    # Helper functions
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

    # --- ABSTRACT ---
    p_abs = doc.add_paragraph()
    r_abs_t = p_abs.add_run("TÓM TẮT ĐỀ TÀI (ABSTRACT)\n")
    r_abs_t.font.bold = True
    r_abs_t.font.size = Pt(11.5)
    r_abs_t.font.color.rgb = NAVY
    r_abs = p_abs.add_run(
        "Nghiên cứu này đề xuất một khung học máy đa phương thức (Multimodal Machine Learning Framework) nhằm nhận diện sớm và chính xác các trạng thái rủi ro an toàn đe dọa tính mạng tài xế xe máy công nghệ tại các đô thị Việt Nam, bao gồm: hiểm họa tai nạn do ngủ gật tức thì (micro-sleep/drowsiness) và hiểm họa an ninh do bị khách ngồi sau tấn công/cướp giật (rear passenger assault/hostile threat).\n"
        "Trong bối cảnh thực tiễn tại Việt Nam, các phương pháp giám sát trạng thái tài xế (DMS) truyền thống dựa trên điểm mốc khuôn mặt (Facial Landmarks) như tỷ lệ mở mắt (EAR), cử động miệng (MAR) hay hướng nhìn (Gaze Direction) hoàn toàn bị tê liệt do thói quen đặc thù của người lái xe máy: trùm khẩu trang kín mặt, đeo kính râm chống bụi/nắng và đội mũ bảo hiểm che khuất phần lớn diện mạo. Để vượt qua rào cản này, nghiên cứu chuyển dịch trọng tâm bài toán học máy sang hai phương thức dữ liệu mới: (1) Dữ liệu thị giác phi khuôn mặt thông qua ước lượng tư thế nón bảo hiểm (Helmet Pose Estimation) và nhận diện cử chỉ hung khí; (2) Dữ liệu sinh trắc học dạng chuỗi thời gian khai thác động học biến thiên nhịp tim (Heart Rate Variability - HRV) trích xuất từ cảm biến quang phổ thể tích (PPG) trên vòng đeo tay thông minh.\n"
        "Bài toán được phát biểu hình thức dưới dạng Phân loại chuỗi đa phương thức có điều kiện bất đối xứng chi phí và mất cân bằng dữ liệu cực độ (Extreme Class-Imbalanced Cost-Sensitive Multimodal Classification). Nhóm đề xuất và so sánh ba chiến lược dung hợp (Early, Late và Cross-Modal Attention Intermediate Fusion) kết hợp hàm mất mát trọng số tập trung (Focal Loss) và hàm mất mát cân bằng lớp (Class-Balanced Loss). Quy trình thẩm định khoa học được thiết lập chặt chẽ bằng chiến lược Stratified Subject-Independent K-Fold Cross-Validation trên các tập dữ liệu chuẩn quốc tế (NTHU-DDD, WESAD, Roboflow Weapon) cùng tập dữ liệu thực nghiệm đường phố tại Việt Nam, kết hợp phương pháp giải thích mô hình (Grad-CAM và SHAP) nhằm đảm bảo tính minh bạch và độ tin cậy khoa học cao nhất."
    )
    r_abs.font.size = Pt(10)
    p_abs.paragraph_format.space_after = Pt(12)

    # --- CHƯƠNG 1 ---
    add_h1("CHƯƠNG 1: BỐI CẢNH THỰC TIỄN & TÍNH CẤP THIẾT CỦA BÀI TOÁN HỌC MÁY")
    add_h2("1.1. Bối cảnh thực tiễn và hiểm họa kép tại Việt Nam")
    add_p("Tại các đại đô thị Việt Nam như TP. Hồ Chí Minh và Hà Nội, sự bùng nổ của các nền tảng kinh tế số (Grab, Be, Gojek, ShopeeFood, Xanh SM...) đã hình thành lực lượng lao động hàng trăm nghìn tài xế xe gắn máy hai bánh. Mặc dù là mắt xích huyết mạch trong giao vận đô thị, lực lượng lao động này đang phải hoạt động trong môi trường làm việc đơn độc, thiếu sự bảo hộ trực tiếp và hàng ngày đối mặt với hai nguy cơ chí tử:")
    add_bullet("Hiểm họa an ninh nghiêm trọng (Cướp giật & Tấn công bạo lực): ", "Tài xế thường xuyên nhận những cuốc xe đêm muộn đưa đón khách vào các khu vực vắng người hoặc vùng ven heo hút. Đối tượng tội phạm thường ngồi ở yên sau, bất ngờ dùng dây siết cổ, kề dao khống chế hoặc dùng hung khí tấn công từ sau lưng để cướp tài sản. Tài xế hoàn toàn rơi vào thế bị động, việc mở điện thoại cầu cứu là bất khả thi.")
    add_bullet("Hiểm họa tai nạn giao thông do ngủ gật tức thì (Micro-sleep): ", "Áp lực mưu sinh buộc tài xế phải làm việc liên tục từ 10 đến 14 tiếng mỗi ngày ngoài đường phố. Cường độ làm việc kéo dài dẫn đến tình trạng kiệt sức thần kinh, gây ra hiện tượng ngủ gật tích tắc (micro-sleep kéo dài từ 1 đến 3 giây) khi đang điều khiển xe máy ở tốc độ cao, gây ra các vụ tai nạn đối đầu thảm khốc.")

    add_h2("1.2. Thách thức khoa học đặc thù dưới góc độ Machine Learning")
    add_bullet("Rào cản che khuất khuôn mặt nghiêm ngặt (Severe Facial Occlusion): ", "Có tới 95% tài xế xe máy tại Việt Nam sử dụng mũ bảo hiểm, đeo kính râm chống nắng/chống bụi và trùm khẩu trang che kín toàn bộ nửa dưới mặt. Điều này khiến các kỹ thuật thị giác máy tính truyền thống dựa trên Facial Landmarks (EAR, MAR, PERCLOS) hoàn toàn mất khả năng nhận diện.")
    add_bullet("Sự mất cân bằng dữ liệu cực độ (Extreme Class Imbalance): ", "Trong quá trình vận hành tự nhiên, trạng thái lái xe bình thường chiếm >98% tổng thời gian. Trạng thái ngủ gật chiếm <1-2%, và trạng thái bị tấn công là sự kiện cực hiếm (<0.1%). Các hàm mất mát tiêu chuẩn như Cross-Entropy sẽ dẫn đến mô hình bị thiên lệch cực độ, dự đoán tất cả là bình thường.")
    add_bullet("Dữ liệu đa phương thức không đồng nhất & bất đối xứng tần số lấy mẫu (Heterogeneous Asynchronous Modalities): ", "Hệ thống phải xử lý đồng thời luồng ảnh thị giác 2D (10-15 FPS), luồng chuỗi thời gian nhịp tim 1D (1 Hz) và tín hiệu rời rạc logic (nút bấm SOS khẩn cấp), đòi hỏi cơ chế dung hợp đặc trưng thông minh.")
    add_bullet("Nhiễu môi trường động (Motion Blur & Motion Artifacts): ", "Độ rung lắc mạnh của xe máy gây nhòe mờ khung hình camera, đồng thời làm biến dạng dạng sóng đo mạch quang thể tích PPG của vòng đeo tay.")

    # --- CHƯƠNG 2 ---
    add_h1("CHƯƠNG 2: TỔNG QUAN HỌC THUẬT & KHOẢNG TRỐNG NGHIÊN CỨU")
    add_h2("2.1. Phân tích các hướng tiếp cận hiện hữu")
    add_p("Trong cộng đồng khoa học Trí tuệ Nhân tạo, bài toán giám sát trạng thái tài xế (Driver Monitoring System - DMS) và phát hiện hành vi bạo lực đã có nhiều công trình nổi bật nhưng bộc lộ nhiều điểm hạn chế khi chuyển dịch sang phương tiện xe hai bánh:")
    add_bullet("Hệ thống phát hiện ngủ gật dựa trên thị giác khuôn mặt: ", "Soukupová & Čech (2016) giới thiệu chỉ số EAR; Abbas (2020) kết hợp CNN-RNN theo dõi chớp mắt. Tất cả đều giả định khuôn mặt lộ rõ trong cabin xe hơi, hoàn toàn tê liệt trước bối cảnh xe máy ngoài trời tại Việt Nam.")
    add_bullet("Hệ thống dựa trên động lực học xe máy: ", "Sayed & Eskandarian (2001) phân tích góc đánh vô-lăng qua cổng CAN-bus ô tô. Hoàn toàn không tương thích với xe gắn máy phổ thông.")
    add_bullet("Hệ thống sinh lý học y tế: ", "Lal & Craig (2001) đo EEG/ECG y tế. Gây vướng víu, không khả thi trong thực tế vận hành hàng ngày của tài xế xe ôm.")

    add_h2("2.2. Khoảng trống học thuật xác định (Research Gaps)")
    add_callout(
        "GAP 1: Chưa có nghiên cứu nào mô hình hóa bài toán phát hiện trạng thái ngủ gật thông qua Ước lượng tư thế Mũ bảo hiểm (Helmet Pose Estimation) thay thế cho khuôn mặt khi bị che khuất nghiêm ngặt.\n"
        "GAP 2: Thiếu khung dung hợp đa phương thức kết hợp Thị giác phi khuôn mặt và Động học chuỗi thời gian nhịp tim (HRV) nhằm triệt tiêu báo động giả khi tài xế cúi đầu quan sát đồng hồ xe.\n"
        "GAP 3: Thiếu vắng mô hình học máy nhạy cảm chi phí (Cost-Sensitive Learning) tối ưu hóa rủi ro kỳ vọng Bayes cho bài toán an toàn giao thông hai bánh."
    )

    # --- CHƯƠNG 3 ---
    add_h1("CHƯƠNG 3: ĐỊNH NGHĨA HÌNH THỨC BÀI TOÁN HỌC MÁY")
    add_h2("3.1. Mô hình toán học tổng quát")
    add_p("Xét bài toán học có giám sát đa phương thức (Multimodal Supervised Learning). Tại mỗi bước thời gian t, hệ thống nhận một bộ dữ liệu quan sát đầu vào x_t in X và cần phân loại trạng thái y_hat_t in Y.")
    
    add_callout(
        "MÔ HÌNH TOÁN HỌC HÌNH THỨC (MATHEMATICAL FORMULATION):\n\n"
        "• Không gian đầu vào: X = X_vis x X_bio x X_disc\n"
        "  - Tensor thị giác: I_t in R^{C x H x W} (Ảnh camera góc rộng 96x96 hoặc 160x160)\n"
        "  - Chuỗi thời gian: H_t = [hr_{t-k}, ..., hr_t]^T in R^{k+1} (Chuỗi nhịp tim PPG từ Smartband)\n"
        "  - Tín hiệu logic:   s_t in {0, 1} (Công tắc xúc giác mini giấu kín ở cùm tay lái)\n\n"
        "• Không gian nhãn mục tiêu: Y = {0: Normal, 1: Drowsiness, 2: Assault / Danger}\n\n"
        "• Lớp hàm giả thuyết: f_Theta: X -> Delta^2, trong đó Delta^2 là simplex xác suất 3 chiều\n"
        "  y_hat_t = argmax_{c in Y} f_Theta(x_t)_c\n\n"
        "• Mục tiêu tối ưu: min_Theta E_{(x, y) ~ D} [ L_task(f_Theta(x), y) ] + lambda * ||Theta||_2^2"
    )

    add_h2("3.2. Không gian đặc trưng đầu vào chi tiết")
    add_bullet("1. Không gian thị giác không gian (X_vis in R^{C x H x W}): ", "Khung hình số chuẩn hóa kích thước H = W = 96 (hoặc 160), số kênh C in {1, 3}, giá trị pixel p in [-1.0, 1.0]. Tập trung vào góc nghiêng chóp nón bảo hiểm và vùng chuyển động tay của người ngồi sau.")
    add_bullet("2. Không gian chuỗi thời gian sinh học (X_bio in R^{T x D}): ", "Chuỗi nhịp tim trong cửa sổ trượt quá khứ độ dài T cùng các chỉ số dẫn xuất biến thiên nhịp tim (HRV): Độ lệch chuẩn khoảng nhịp NN (SDNN), căn bậc hai trung bình bình phương hiệu số liên tiếp (RMSSD), và tỷ lệ sụt giảm nhịp tim tương đối Delta hr_t = (hr_t - hr_base) / hr_base * 100%.")
    add_bullet("3. Không gian kích hoạt logic (X_disc in {0, 1}): ", "Tín hiệu ngắt phần cứng từ nút bấm SOS khẩn cấp giấu kín tại cùm tay lái xe máy.")

    add_h2("3.3. Không gian nhãn mục tiêu (Y)")
    add_bullet("Lớp 0 (Normal - Lái xe bình thường): ", "Tư thế nón bảo hiểm thẳng hoặc dao động góc nhỏ (<20°), nhịp tim dao động bình thường khi lái xe (75 - 95 BPM), hành khách ngồi ngay ngắn.")
    add_bullet("Lớp 1 (Drowsiness - Ngủ gật tích tắc): ", "Nón bảo hiểm cúi gục về trước hoặc ngả sang bên (>=25°) duy trì liên tục qua ngưỡng trễ sinh lý (>=1.5s), kết hợp nhịp tim sinh học suy giảm sâu (Delta hr <= -15%).")
    add_bullet("Lớp 2 (Assault - Bị khống chế / Tấn công nguy hiểm): ", "Người ngồi sau có cử chỉ siết cổ, giơ hung khí sắc nhọn hoặc nhịp tim tài xế tăng vọt do phản xạ giao cảm hoảng sợ (hr_t > 130 BPM, Delta hr >= +50%) hoặc tài xế chủ động ấn nút SOS bí mật (s_t = 1).")

    add_h2("3.4. Các chiến lược dung hợp đa phương thức (Multimodal Fusion Strategies)")
    add_p("Đồ án thiết kế và tiến hành thực nghiệm so sánh đối chứng 3 cấu trúc dung hợp:")
    add_bullet("Chiến lược 1 - Dung hợp sớm (Early Fusion): ", "Nối chuỗi vector đặc trưng thị giác z_vis in R^{d1} và vector sinh trắc học z_bio in R^{d2} thành z_joint = [z_vis || z_bio || s_t], sau đó đưa qua mạng truyền thẳng Joint MLP để phân loại.")
    add_bullet("Chiến lược 2 - Dung hợp muộn (Late Fusion): ", "Huấn luyện 2 mô hình độc lập sinh ra phân phối xác suất p_vis và p_bio, kết hợp bằng bình quân gia quyền thích nghi cùng luật ghi đè ưu tiên an ninh (Priority Security Override Rule).")
    add_bullet("Chiến lược 3 - Dung hợp tầng trung gian dựa trên Chú ý chéo (Cross-Modal Attention Fusion): ", "Sử dụng đặc trưng nhịp tim chuỗi thời gian làm Query (Q_bio) để truy vấn bản đồ đặc trưng thị giác Keys/Values (K_vis, V_vis), giúp mô hình tập trung vào vùng ảnh nghi vấn khi nhịp tim biến động bất thường.")

    # --- CHƯƠNG 4 ---
    add_h1("CHƯƠNG 4: HÀM MỤC TIÊU TỐI ƯU HÓA & XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU CỰC ĐỘ")
    add_h2("4.1. Hàm mất mát Focal Loss cho bài toán phân loại đa lớp")
    add_p("Để giải quyết hiện tượng tổng Gradient bị chi phối hoàn toàn bởi các mẫu 'Bình thường' (easy negatives), nhóm áp dụng Multi-Class Focal Loss:")
    add_callout(
        "HÀM MẤT MÁT FOCAL LOSS:\n"
        "L_FL(p_t, y) = - sum_{c=0}^2 alpha_c * (1 - p_{t, c})^gamma * log(p_{t, c}) * I(y = c)\n\n"
        "Trong đó:\n"
        "• p_{t, c} là xác suất dự đoán mẫu t thuộc lớp c.\n"
        "• (1 - p_{t, c})^gamma là hệ số điều biến (Modulating Factor), tự động triệt tiêu trọng số gradient của các mẫu dễ phân loại khi p_{t, c} -> 1.\n"
        "• gamma >= 0 là tham số tập trung (Focusing Parameter), tinh chỉnh trong dải [1.5, 3.0].\n"
        "• alpha_c là trọng số cân bằng tần suất lớp, bù trừ tỷ lệ xuất hiện giữa các lớp."
    )

    add_h2("4.2. Hàm mất mát cân bằng lớp dựa trên số mẫu hiệu dụng (Class-Balanced Loss)")
    add_p("Theo lý thuyết của Cui et al. (CVPR 2019), nhóm áp dụng hàm mất mát cân bằng dựa trên thể tích mẫu hiệu dụng E_n = (1 - beta^n) / (1 - beta):")
    add_callout(
        "HÀM MẤT MÁT CLASS-BALANCED FOCAL LOSS:\n"
        "L_CB(p_t, y) = - [(1 - beta) / (1 - beta^{n_y})] * (1 - p_{t, y})^gamma * log(p_{t, y})\n"
        "Với n_y là tổng số mẫu của lớp y trong tập huấn luyện, beta in [0.9, 0.9999]."
    )

    add_h2("4.3. Mô hình hóa Rủi ro Nhạy cảm Chi phí (Cost-Sensitive Bayes Risk)")
    add_p("Do sai lầm bỏ sót nguy hiểm (False Negative) đe dọa trực tiếp đến tính mạng tài xế, nhóm xây dựng Ma trận Chi phí Sai lầm C in R^{3 x 3} với mức phạt cực nặng khi dự đoán nhầm lớp Nguy hiểm thành Bình thường:")
    
    cost_table = doc.add_table(rows=4, cols=4)
    cost_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Thực tế \\ Dự đoán", "y_hat = 0 (Normal)", "y_hat = 1 (Drowsy)", "y_hat = 2 (Assault)"]
    for j, h in enumerate(headers):
        cell = cost_table.cell(0, j)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        set_cell_background(cell, "1B365D")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 60, 60, 80, 80)
    
    matrix_data = [
        ("y = 0 (Normal)", "C_00 = 0", "C_01 = 1 (Còi báo phiền)", "C_02 = 2 (Cảnh báo nhầm)"),
        ("y = 1 (Drowsy)", "C_10 = 50 (Bỏ sót ngủ gật - NGUY HIỂM)", "C_11 = 0", "C_12 = 5"),
        ("y = 2 (Assault)", "C_20 = 200 (Bỏ sót cướp - CHÍ TỬ)", "C_21 = 20", "C_22 = 0")
    ]
    for i, row_data in enumerate(matrix_data):
        row = cost_table.rows[i+1]
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            if j == 0:
                r.font.bold = True
                set_cell_background(cell, "F2F5F9")
            elif (i == 1 and j == 1) or (i == 2 and j == 1):
                r.font.bold = True
                set_cell_background(cell, "FFEBEE")
                r.font.color.rgb = RGBColor(0xC6, 0x28, 0x28)
            else:
                set_cell_background(cell, "FAFAFA")
            set_cell_margins(cell, 50, 50, 80, 80)
    set_table_borders(cost_table, "B0C4DE")
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # --- CHƯƠNG 5 ---
    add_h1("CHƯƠNG 5: KIẾN TRÚC TẬP DỮ LIỆU & QUY TRÌNH KỸ THUẬT DỮ LIỆU")
    add_h2("5.1. Thành phần dữ liệu học máy đa nguồn")
    add_bullet("1. Dữ liệu ngủ gật che khuất mặt (NTHU-DDD & UTA-RLDD): ", "Trích xuất chuỗi hình ảnh tài xế đeo kính râm, đeo khẩu trang, đội mũ. Tập trung gán nhãn góc nghiêng nón bảo hiểm theo chuẩn độ nghiêng cơ học.")
    add_bullet("2. Dữ liệu hung khí & hành vi tấn công (Roboflow Universe): ", "Hơn 4.200 ảnh gán nhãn bounding-box hung khí nguy hiểm (dao, vật sắc nhọn) và các tư thế vươn tay siết cổ từ phía sau.")
    add_bullet("3. Dữ liệu sinh trắc học chuỗi thời gian (WESAD - Uni Freiburg): ", "Tập dữ liệu y tế đa cảm biến của 15 đối tượng ghi nhận trạng thái Thư giãn, Căng thẳng hoảng loạn (Panic) và Lơ mơ ngủ gật, tương thích với định dạng BLE Smartband.")
    add_bullet("4. Dữ liệu thực nghiệm đường phố Việt Nam (Local Validation Set): ", "1.200 ảnh nón bảo hiểm chụp ngoài phố thực tế và log chuỗi nhịp tim thời gian thực từ Xiaomi Smart Band chạy xe máy ngoài đường phố TP.HCM.")

    add_h2("5.2. Quy mô mẫu và phân bố tập dữ liệu")
    data_table = doc.add_table(rows=5, cols=5)
    data_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    d_headers = ["Lớp", "Trạng thái", "Nguồn dữ liệu chính", "Số lượng mẫu", "Tỷ lệ phân bố"]
    for j, h in enumerate(d_headers):
        cell = data_table.cell(0, j)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        set_cell_background(cell, "1B365D")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 60, 60, 80, 80)
    
    d_rows = [
        ("0", "Bình thường (Normal)", "NTHU-DDD + WESAD Baseline + Thực nghiệm", "4.200 mẫu", "70.0%"),
        ("1", "Ngủ gật (Drowsiness)", "NTHU-DDD Occluded + WESAD Drowsy + Thực nghiệm", "1.200 mẫu", "20.0%"),
        ("2", "Tấn công (Assault)", "Roboflow Weapon + WESAD Panic + Diễn tập", "600 mẫu", "10.0%"),
        ("TỔNG", "Toàn bộ tập dữ liệu", "Đa nguồn đồng bộ hóa", "6.000 mẫu", "100.0%")
    ]
    for i, rdata in enumerate(d_rows):
        row = data_table.rows[i+1]
        for j, val in enumerate(rdata):
            cell = row.cells[j]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            if i == 3:
                r.font.bold = True
                set_cell_background(cell, "E8EEF5")
            elif j == 0 or j == 1:
                r.font.bold = True
                set_cell_background(cell, "F2F5F9")
            else:
                set_cell_background(cell, "FAFAFA")
            set_cell_margins(cell, 50, 50, 80, 80)
    set_table_borders(data_table, "B0C4DE")
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_h2("5.3. Chiến lược tăng cường dữ liệu đặc thù (Data Augmentation)")
    add_bullet("Tăng cường không gian thị giác: ", "Làm mờ chuyển động (Motion Blur với kernel ngẫu nhiên 5x5 đến 9x9 mô phỏng xe rung sóc), thay đổi độ sáng ngẫu nhiên ±40% và tạo vệt bóng râm ngẫu nhiên (Shadow injection), biến đổi phối cảnh góc nghiêng ±15°.")
    add_bullet("Tăng cường chuỗi thời gian sinh học: ", "Cộng nhiễu Gaussian ngẫu nhiên mô phỏng rung lắc vòng tay, kéo giãn nén trục thời gian (Time-Warping với hệ số 0.85 - 1.15) để mô phỏng khác biệt phản xạ giữa các cá nhân.")

    # --- CHƯƠNG 6 ---
    add_h1("CHƯƠNG 6: KHUNG KIẾN TRÚC MÔ HÌNH HỌC MÁY ĐỀ XUẤT")
    add_h2("6.1. Mô hình đối chuẩn (Baselines)")
    add_bullet("1. LightGBM / XGBoost trên Đặc trưng Trích xuất Thủ công: ", "Trích xuất HOG & LBP từ vùng nón bảo hiểm kết hợp 8 chỉ số thống kê HRV từ chuỗi nhịp tim. Dùng làm chuẩn so sánh hiệu năng truyền thống.")
    add_bullet("2. Mô hình Thị giác Đơn lẻ (Vision-Only MobileNetV3 / ResNet-18): ", "Chỉ sử dụng ảnh để phân loại, dùng để đo lường định lượng giá trị gia tăng khi dung hợp thêm kênh nhịp tim sinh học.")

    add_h2("6.2. Khung mô hình Học sâu Đa phương thức đề xuất (Cross-Attention Multimodal Model)")
    add_p("Khung mô hình đề xuất bao gồm 3 khối kiến trúc phối hợp chặt chẽ:")
    add_bullet("Nhánh Thị giác (Visual Backbone): ", "Sử dụng MobileNetV3-Small trích xuất bản đồ đặc trưng không gian F_vis in R^{C' x H' x W'}, tối ưu hóa việc nhận diện góc nghiêng nón bảo hiểm và hình thái hung khí.")
    add_bullet("Nhánh Chuỗi thời gian Sinh học (Temporal Head): ", "Mạng tích chập 1D-CNN kết hợp Bidirectional LSTM (32 units) để nắm bắt xu hướng tăng vọt hay suy giảm dài hạn của nhịp tim.")
    add_bullet("Khối Dung hợp Chú ý chéo (Cross-Modal Attention Block): ", "Sử dụng vector biểu diễn nhịp tim làm Query để định hướng vùng chú ý thị giác. Đầu ra được kết hợp với tín hiệu nút bấm SOS đưa vào bộ phân loại MLP cuối cùng.")

    # --- CHƯƠNG 7 ---
    add_h1("CHƯƠNG 7: PHƯƠNG PHÁP LUẬN THẨM ĐỊNH KHOA HỌC & HỆ CHỈ SỐ ĐÁNH GIÁ")
    add_h2("7.1. Chiến lược phân chia dữ liệu độc lập đối tượng (Stratified Subject-Independent K-Fold)")
    add_p("Để triệt tiêu hoàn toàn nguy cơ rò rỉ dữ liệu (Data Leakage) khi cùng một cá nhân xuất hiện ở cả tập huấn luyện và kiểm thử, nhóm áp dụng phương pháp Stratified Subject-Independent Group K-Fold Cross-Validation (K=5). Toàn bộ dữ liệu của một tài xế/đối tượng chỉ được nằm trọn vẹn trong một tập tại mỗi Fold. Tách biệt 20% dữ liệu độc lập hoàn toàn làm tập kiểm thử cuối cùng (Held-out Test Set).")

    add_h2("7.2. Hệ chỉ số đo lường hiệu năng chuyên sâu")
    add_bullet("Recall từng lớp (Độ nhạy): ", "Yêu cầu Recall_{Drowsy} >= 95% và Recall_{Assault} >= 98% nhằm ngăn chặn tuyệt đối việc bỏ sót các tình huống nguy hiểm đến tính mạng.")
    add_bullet("Precision từng lớp (Độ chuẩn xác): ", "Yêu cầu Precision_{Drowsy} >= 90% nhằm hạn chế tối đa báo động giả làm phân tâm tài xế khi đang điều khiển phương tiện.")
    add_bullet("Macro-F1 Score & Weighted-F1 Score: ", "Thước đo trung bình điều hòa công bằng giữa các lớp, đặc biệt phản ánh chính xác hiệu năng trên lớp thiểu số (Tấn công).")
    add_bullet("Precision-Recall AUC (PR-AUC): ", "Diện tích dưới đường cong PR trên mọi ngưỡng phân loại, chuẩn mực đánh giá tin cậy nhất cho các bài toán mất cân bằng dữ liệu lớn.")
    add_bullet("Normalized Confusion Matrix & Cost Analysis: ", "Trực quan hóa ma trận nhầm lẫn chuẩn hóa và tính toán tổng chi phí sai lầm theo ma trận chi phí Bayes.")

    # --- CHƯƠNG 8 ---
    add_h1("CHƯƠNG 8: TÍNH MINH BẠCH & KHẢ NĂNG GIẢI THÍCH MÔ HÌNH (EXPLAINABLE AI - XAI)")
    add_h2("8.1. Grad-CAM cho Nhánh Thị giác Máy tính")
    add_p("Nhóm tích hợp Gradient-weighted Class Activation Mapping (Grad-CAM) tính đạo hàm ngược từ điểm số phân loại trạng thái về bản đồ đặc trưng tích chập cuối cùng để sinh ra bản đồ nhiệt (Heatmap). Kết quả giúp chứng minh mô hình thực sự học được tư thế nón bảo hiểm cúi gục và hình thái hung khí vươn lên từ phía sau, thay vì học vẹt hậu cảnh đường sá hay màu áo khoác.")

    add_h2("8.2. Phân tích giá trị đóng góp Shapley (SHAP Values cho Chuỗi Nhịp tim)")
    add_p("Ứng dụng lý thuyết trò chơi hợp tác (Shapley Values) để bóc tách mức độ đóng góp cận biên của từng thuộc tính sinh lý học (Delta hr_t, SDNN, RMSSD). Phân tích này giải thích cơ chế mô hình triệt tiêu báo động giả: Khi tài xế cúi đầu nhìn gương xe máy, nón bảo hiểm cúi xuống nhưng giá trị SHAP của nhịp tim sẽ kéo xác suất về trạng thái Bình thường do nhịp tim vẫn duy trì ổn định.")

    # --- CHƯƠNG 9 ---
    add_h1("CHƯƠNG 9: KẾ HOẠCH NGHIÊN CỨU & MA TRẬN PHÂN CÔNG NHIỆM VỤ CHI TIẾT")
    add_h2("9.1. Lộ trình thực hiện đồ án 8 tuần")
    add_bullet("Tuần 1-2 (27/10 – 08/11/2026): ", "Khảo sát học thuật, hoàn thiện tài liệu xác định bài toán, thu thập dữ liệu NTHU-DDD, WESAD, Roboflow Weapon.")
    add_bullet("Tuần 3-4 (09/11 – 22/11/2026): ", "Tiền xử lý dữ liệu, cài đặt Data Augmentation, huấn luyện các mô hình đối chuẩn Baseline (LightGBM, MobileNetV3-Single).")
    add_bullet("Tuần 5-6 (23/11 – 06/12/2026): ", "Xây dựng khung mô hình Đa phương thức (Early/Late/Cross-Attention), tích hợp Focal Loss và Class-Balanced Loss.")
    add_bullet("Tuần 7 (07/12 – 13/12/2026): ", "Thực nghiệm kiểm định Stratified 5-Fold, phân tích Confusion Matrix, tích hợp Grad-CAM và SHAP.")
    add_bullet("Tuần 8 (14/12 – 20/12/2026): ", "Tổng hợp kết quả thực nghiệm, hoàn thiện báo cáo khoa học và chuẩn bị slide bảo vệ cuối kỳ.")

    add_h2("9.2. Ma trận phân công trách nhiệm nhóm (RACI Matrix)")
    raci_table = doc.add_table(rows=7, cols=6)
    raci_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    r_headers = ["Hạng mục chuyên môn", "Hôn", "Khang", "Trung", "Duy", "Sơn"]
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
        ("Mô hình hóa toán học & Thiết kế kiến trúc", "A/R", "C", "C", "C", "I"),
        ("Xử lý dữ liệu ảnh nón bảo hiểm & CNN", "I", "A/R", "I", "C", "R"),
        ("Xử lý chuỗi nhịp tim PPG & Tính toán HRV", "I", "I", "A/R", "C", "R"),
        ("Tối ưu hóa Focal Loss & Đánh giá 5-Fold", "C", "I", "I", "A/R", "I"),
        ("Triển khai Explainable AI (Grad-CAM & SHAP)", "C", "R", "R", "C", "A/R"),
        ("Viết bản Thuyết minh bài toán & Báo cáo", "A/R", "R", "R", "R", "R")
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
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # --- CHƯƠNG 10 ---
    add_h1("CHƯƠNG 10: TÀI LIỆU THAM KHẢO HỌC THUẬT")
    refs = [
        "Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. IEEE TPAMI, 42(2), 318-327.",
        "Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. In CVPR, pp. 9268-9277.",
        "Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. In ACM ICMI, pp. 400-408.",
        "Weng, C. H., Lai, Y. H., & Lai, S. H. (2016). Driver Drowsiness Detection via a Smartphone Platform. In IEEE ICCVW, pp. 1-8.",
        "Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. In IEEE ICCV, pp. 618-626.",
        "Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS 30), pp. 4765-4774.",
        "Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for MobileNetV3. In IEEE ICCV, pp. 1314-1324.",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS 30, pp. 5998-6008.",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In IEEE CVPR, pp. 770-778.",
        "Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. In IJCAI, Vol. 17, No. 1, pp. 973-978."
    ]
    for r in refs:
        p = doc.add_paragraph(style='List Bullet')
        run = p.add_run(r)
        run.font.size = Pt(9.5)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15

    # Save to docs folder
    docs_output = r"D:\Trí tuệ nhân tạo\HK3\Máy học nâng cao\Đồ án\docs\Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.docx"
    root_output = r"D:\Trí tuệ nhân tạo\HK3\Máy học nâng cao\Đồ án\Dinh_Nghia_Bai_Toan_De_Tai_Machine_Learning.docx"
    doc.save(docs_output)
    doc.save(root_output)
    print("Docx generated successfully.")

if __name__ == "__main__":
    create_document()
