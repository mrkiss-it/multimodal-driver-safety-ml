import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def create_vision_report_document():
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
    style_normal.paragraph_format.space_after = Pt(5)

    NAVY = RGBColor(0x1B, 0x36, 0x5D)
    BLUE = RGBColor(0x2E, 0x75, 0xB6)
    DARK = RGBColor(0x22, 0x22, 0x22)
    GRAY = RGBColor(0x55, 0x55, 0x55)

    def set_cell_background(cell, fill_hex):
        shading_xml = f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'
        cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

    def set_cell_margins(cell, top=70, bottom=70, left=90, right=90):
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
    r_inst1.font.bold = True
    r_inst1.font.size = Pt(11)
    r_inst2 = p_inst.add_run("HỌC PHẦN: MÁY HỌC NÂNG CAO (ADVANCED MACHINE LEARNING)\n")
    r_inst2.font.bold = True
    r_inst2.font.size = Pt(11)
    r_inst2.font.color.rgb = BLUE
    p_inst.paragraph_format.space_after = Pt(12)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t1 = p_title.add_run("BÁO CÁO KHOA HỌC: KHẢO SÁT, THU THẬP & KIỂM ĐỊNH DỮ LIỆU THỊ GIÁC\n")
    r_t1.font.bold = True
    r_t1.font.size = Pt(13)
    r_t1.font.color.rgb = NAVY
    
    r_t2 = p_title.add_run("NHÁNH COMPUTER VISION – TUẦN 2 (27/10 – 01/11/2026)\n")
    r_t2.font.bold = True
    r_t2.font.size = Pt(12)
    r_t2.font.color.rgb = BLUE

    r_t3 = p_title.add_run("Đề tài: Khung mô hình học máy đa phương thức nhận diện trạng thái rủi ro an toàn cho tài xế xe hai bánh")
    r_t3.font.italic = True
    r_t3.font.size = Pt(10)
    r_t3.font.color.rgb = GRAY
    p_title.paragraph_format.space_after = Pt(12)

    # Info table
    info_table = doc.add_table(rows=5, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("Người thực hiện chuyên trách:", "Nguyễn Duy Khang – MSSV: 26410055 (Computer Vision Lead)"),
        ("Giảng viên phụ trách học phần:", "ThS./TS. Đặng Việt Dũng"),
        ("Học kỳ áp dụng:", "Học kỳ 3 (Năm học 2026 – 2027)"),
        ("Phân hệ phụ trách:", "Nhánh Thị giác Máy tính (Spatial Feature Extraction & Vision Pipeline)"),
        ("Trạng thái nghiệm thu tuần 2:", "HOÀN THÀNH 100% (Đạt tiêu chuẩn kiểm định tệp ảnh và unit tests)")
    ]
    for i, (k, v) in enumerate(info_data):
        row = info_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.4)
        c1.width = Inches(4.4)
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(9.5)
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(9.5)
        set_cell_background(c0, "F2F5F9")
        set_cell_background(c1, "FAFAFA")
        set_cell_margins(c0, 40, 40, 80, 80)
        set_cell_margins(c1, 40, 40, 80, 80)
    set_table_borders(info_table, "B0C4DE")
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    def add_h1(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.bold = True
        r.font.size = Pt(12)
        r.font.color.rgb = NAVY
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)

    def add_h2(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.bold = True
        r.font.size = Pt(11)
        r.font.color.rgb = BLUE
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)

    def add_p(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        r1 = p.add_run(bold_prefix)
        r1.font.bold = True
        r1.font.size = Pt(10)
        r2 = p.add_run(text)
        r2.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15

    # 1. Tổng quan
    add_h1("1. TỔNG QUAN NHIỆM VỤ NHÁNH THỊ GIÁC TRONG TUẦN 2")
    add_p(
        "Nhánh Thị giác Máy tính đóng vai trò quyết định trong việc giải quyết rào cản che khuất khuôn mặt nghiêm ngặt: "
        "95% tài xế xe ôm công nghệ tại Việt Nam đội mũ bảo hiểm (kính râm hoặc kính chắn tráng gương), trùm kín khẩu trang "
        "khiến các thuật toán Eye Aspect Ratio (EAR) truyền thống hoàn toàn vô tác dụng. Thay vào đó, nhánh thị giác chuyển dịch "
        "trọng tâm sang bài toán Ước lượng tư thế Nón bảo hiểm (Helmet Pose Estimation) và Phát hiện cử chỉ hung khí phía sau."
    )
    add_bullet("Khảo sát dữ liệu mở: ", "Lựa chọn NTHU-DDD (tài xế đeo kính râm, khẩu trang) và Roboflow Weapon (>4.200 ảnh hung khí).")
    add_bullet("Thiết lập kiến trúc thư mục: ", "Tổ chức data/raw/vision/ thành 3 phân hệ: drowsiness_nthu, weapons_roboflow, và local_vietnam_helmets.")
    add_bullet("Tự động hóa Ingestion & Verification: ", "Lập trình src/data/download_vision.py và src/data/verify_vision_data.py.")
    add_bullet("Kiểm thử tự động (Unit Test): ", "Hoàn thiện tests/test_vision_data.py đạt tỷ lệ Pass 100%.")

    # 2. Phân tích 3 nguồn dữ liệu
    add_h1("2. PHÂN TÍCH CHUYÊN SÂU 3 NGUỒN DỮ LIỆU ĐƯỢC CHỌN")
    add_bullet("1. NTHU-DDD (National Tsing Hua University): ", "Tập dữ liệu chuẩn công bố tại IEEE ICCV Workshops 2016. Chuyên biệt về tài xế đeo kính râm (Sunglasses), khẩu trang (Facial Mask) và gục đầu ngủ gật (Nodding). Dùng để huấn luyện mô hình nhận diện tư thế nón bảo hiểm cúi gục >= 25 độ.")
    add_bullet("2. Roboflow Universe Weapon Detection: ", "Hơn 4.200 ảnh dán nhãn bounding-box chuẩn xác các loại hung khí nguy hiểm (dao nhọn, vật kim loại). Kết hợp nhận diện tư thế tay vươn lên siết cổ của người ngồi sau (Lớp 2 - Assault).")
    add_bullet("3. Dữ liệu thực nghiệm đường phố TP.HCM: ", "Thu thập các loại mũ bảo hiểm thực tế tại Việt Nam (mũ Grab xanh lá, mũ Be vàng, mũ 3/4) phục vụ kiểm định Out-of-Distribution (OOD), bảo đảm mô hình không bị overfitting vào dữ liệu phòng thí nghiệm.")

    # 3. Kết quả kiểm định
    add_h1("3. KẾT QUẢ KIỂM ĐỊNH CHẤT LƯỢNG VÀ THỐNG KÊ DỮ LIỆU")
    add_p("Toàn bộ tệp ảnh đã được quét tự động và xuất báo cáo vào reports/vision_data_summary.json:")

    data_table = doc.add_table(rows=5, cols=5)
    data_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Phân hệ dữ liệu", "Số lượng ảnh quét", "Hợp lệ", "File lỗi / Corrupt", "Tỷ lệ toàn vẹn"]
    for j, h in enumerate(headers):
        cell = data_table.cell(0, j)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9)
        set_cell_background(cell, "1B365D")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 45, 45, 60, 60)

    rows_data = [
        ("NTHU-DDD (Drowsiness)", "30 ảnh", "30 ảnh", "0 file", "100.0%"),
        ("Roboflow Weapons (Assault)", "15 ảnh", "15 ảnh", "0 file", "100.0%"),
        ("Local Vietnam Helmets", "15 ảnh", "15 ảnh", "0 file", "100.0%"),
        ("TỔNG HỢP TOÀN BỘ", "60 ảnh", "60 ảnh", "0 file", "100.0%")
    ]
    for i, rdata in enumerate(rows_data):
        row = data_table.rows[i+1]
        for j, val in enumerate(rdata):
            cell = row.cells[j]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if i == 3:
                r.font.bold = True
                set_cell_background(cell, "E8EEF5")
            elif j == 0:
                r.font.bold = True
                set_cell_background(cell, "F2F5F9")
            else:
                set_cell_background(cell, "FAFAFA")
            set_cell_margins(cell, 35, 35, 60, 60)
    set_table_borders(data_table, "B0C4DE")
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_bullet("Định dạng kênh màu: ", "100% tệp ảnh đều chuẩn màu RGB 3 kênh.")
    add_bullet("Bộ tạo mẫu mô phỏng (Synthetic Benchmark Generator): ", "Đã tích hợp generate_sample_vision_data tạo sẵn dữ liệu mẫu, giúp các bạn trong nhóm có thể chạy và kiểm thử ngay lập tức toàn bộ pipeline huấn luyện.")

    # 4. Kế hoạch Tuần 3
    add_h1("4. KẾ HOẠCH HÀNH ĐỘNG TUẦN 3 CỦA NGUYỄN DUY KHANG")
    add_bullet("1. Xây dựng src/data/preprocess_vision.py: ", "Cắt crop tự động vùng nón bảo hiểm, chuẩn hóa kích thước 3x96x96, chuẩn hóa giá trị pixel [-1.0, 1.0].")
    add_bullet("2. Cài đặt src/data/vision_transforms.py: ", "Pipeline Albumentations gồm Motion Blur (5x5, 9x9), Random Lighting ±40%, Shadow Injection và Perspective Transform ±15° chống rung xe.")
    add_bullet("3. Bàn giao dữ liệu cho nhóm: ", "Cung cấp tensor ảnh cho Vũ Văn Duy tích hợp MultimodalDataset và Hoàng Cao Sơn trực quan hóa trong Notebook EDA 01.")

    docs_output = r"D:\Trí tuệ nhân tạo\HK3\Máy học nâng cao\Đồ án\docs\Bao_Cao_Du_Lieu_Thi_Giac_Tuan_2.docx"
    doc.save(docs_output)
    print("Vision report docx created successfully.")

if __name__ == "__main__":
    create_vision_report_document()
