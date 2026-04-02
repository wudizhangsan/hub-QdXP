import fitz  
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI



# 1. 将 PDF 第一页转换为 Base64 格式的图片
def get_pdf_first_page_as_base64(pdf_path, dpi=200):
    print(f"正在读取 PDF: {pdf_path} ...")
    # 打开 PDF 文件
    doc = fitz.open(pdf_path)
    # 取第 0 页（也就是第一页）
    page = doc.load_page(0)

    # 将页面渲染为像素图 (DPI 越高越清晰，但 Base64 字符串也越长)
    pix = page.get_pixmap(dpi=dpi)

    # 转换为 PIL Image 对象
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 将图片暂存在内存中，并转为 Base64 编码
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    print("PDF 第一页已成功转换为 Base64 图像！")
    return base64_str


# 2. 调用阿里云 Qwen-VL 接口进行多模态解析
def parse_image_with_qwen_vl(base64_image):
    print("正在呼叫云端 Qwen-VL 大模型，请稍候...")

    # 初始化 OpenAI 客户端，指向阿里云百炼的网关
    client = OpenAI(
        # 请替换为你自己的阿里云 API Key
        api_key="sk-642d5cd9c606477badc2e08919f6fa2c",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 构造极其硬核的多模态 Prompt
    prompt_text = """
    你是一个专业的文档解析专家。请仔细观察这张图片（它是PDF的一页），完成以下任务：
    1. 提取出图片中的所有文字内容。
    2. 严格保持原本的排版结构。如果有标题，请用 Markdown 的 # 标出；如果有表格，请用 Markdown 表格还原；如果有数学公式，请用 LaTeX 格式输出。
    3. 不要输出任何无关的解释性废话，直接输出解析结果。
    """

    response = client.chat.completions.create(
        model="qwen3-vl-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # 按照标准协议拼接 Base64 数据头
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content


# 3. 执行主流程
if __name__ == "__main__":
    pdf_file_path = "NLP&大模型-知识点和面试题-2601版本 .pdf"  # 替换成你本地的 PDF 路径

    try:
        # 第一步：PDF 截屏转码
        img_base64 = get_pdf_first_page_as_base64(pdf_file_path)

        # 第二步：大模型解析
        result = parse_image_with_qwen_vl(img_base64)

        print("\n================ 大模型解析结果 ================\n")
        print(result)
        print("\n================================================\n")

    except Exception as e:
        print(f"程序运行出错: {e}")
