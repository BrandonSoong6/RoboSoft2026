from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import cv2

# PDF 输出路径
outfile = "apriltags_id0_2_twosets.pdf"
c = canvas.Canvas(outfile, pagesize=A4)

# 生成 AprilTag 图像
def generate_apriltag(tag_id, size_px=600, tag_family=cv2.aruco.DICT_APRILTAG_36h11):
    aruco_dict = cv2.aruco.getPredefinedDictionary(tag_family)
    img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, size_px)
    return img

# 保存成临时 PNG
def save_tag_img(tag_id, filename):
    img = generate_apriltag(tag_id)
    cv2.imwrite(filename, img)

# 准备 ID 0–2，每个两份
ids = [0, 1, 2]
tags = []
for i in ids:
    for j in range(2):
        fname = f"tag_{i}_{j}.png"
        save_tag_img(i, fname)
        tags.append((i, fname))

# ======= 排版 =======
page_width, page_height = A4
tag_size = 45 * mm   # 每个 marker 45mm 边长

# 两行三列布局
margin_x = (page_width - 3*tag_size) / 4
margin_y = (page_height - 2*tag_size) / 3

x_positions = [margin_x + i*(tag_size+margin_x) for i in range(3)]
y_positions = [page_height - margin_y - tag_size - j*(tag_size+margin_y) for j in range(2)]

for idx, (tid, fpath) in enumerate(tags):
    col = idx % 3
    row = idx // 3
    x = x_positions[col]
    y = y_positions[row]
    c.drawImage(fpath, x, y, width=tag_size, height=tag_size)
    c.setFont("Helvetica", 8)
    c.drawCentredString(x + tag_size/2, y-10, f"ID {tid}")

c.showPage()
c.save()
print("生成完成:", outfile)
