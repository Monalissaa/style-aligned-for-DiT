import os
import base64

# 将图像文件转换为Base64编码
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# 生成HTML文件并嵌入Base64编码的图像
def generate_html_with_images(image_folder, output_html):
    html_content = """
    <html>
    <head><title>Image Gallery</title></head>
    <body>
        <h1>Image Gallery</h1>
        <table>
    """

    files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    files.sort()  # 根据需要排序文件名

    # 每行展示5个图片
    for i, file in enumerate(files):
        if i % 5 == 0:
            if i != 0:
                html_content += "</tr>"
            html_content += "<tr>"
        
        image_path = os.path.join(image_folder, file)
        base64_image = convert_image_to_base64(image_path)
        html_content += f"""
        <td>
            <img src="data:image/png;base64,{base64_image}" alt="{file}" style="width:150px;height:auto;">
            <p>{file}</p>
        </td>
        """
    
    html_content += """
        </tr>
        </table>
    </body>
    </html>
    """

    with open(output_html, "w") as html_file:
        html_file.write(html_content)

# 图片文件夹路径
image_folder = "results/recon_share_layer_specific"  # 替换为你的图片文件夹路径
# 输出HTML文件路径
output_html = "output_recon_share_layer_specific.html"

# 生成HTML
generate_html_with_images(image_folder, output_html)
