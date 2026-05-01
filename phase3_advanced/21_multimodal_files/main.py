"""
LangChain 1.0 - 多模态与文件处理 (Multimodal & File Handling)
===============================================================

本模块重点讲解：
1. 图像输入：让 LLM"看"图片并回答问题
2. PDF 文件解析：提取文档内容用于 RAG
3. 多模态混合消息：文本 + 图像一起发送给模型

说明：
- 详细知识点见 README.md
- 图像相关 demo 需要模型支持多模态（如 gpt-4o、glm-4v）
"""

import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# ============================================================================
# 环境配置
# ============================================================================

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")

if not API_KEY:
    raise ValueError("请先设置 OPENAI_API_KEY")

model = init_chat_model(
    "openai:glm-5.1",
    api_key=API_KEY,
    base_url=BASE_URL,
)

from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================================
# 示例 1：图像理解 —— 让 LLM"看"图片
# ============================================================================
def example_1_image_understanding():
    """
    示例1：图像理解

    LangChain 的多模态消息使用 content 列表格式：
    [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "..."}}
    ]

    支持的图片来源：
    - 在线 URL："https://example.com/image.jpg"
    - Base64："data:image/jpeg;base64,/9j/4AAQ..."
    - 本地文件：需要读取为 base64 后构造 data URI
    """
    print("\n" + "=" * 70)
    print("示例 1：图像理解")
    print("=" * 70)

    # 使用一张示例图片（Lorem Picsum 提供随机图片）
    # 生产环境中替换为实际业务图片
    image_url = "https://picsum.photos/400/300"

    print(f"\n图片 URL: {image_url}")
    print("（如果无法访问外网，请替换为本地图片的 base64 data URI）")

    # 构造多模态消息：content 是列表，包含文本块和图片块
    # 这是 OpenAI 兼容格式，也是 LangChain 支持的标准格式
    message = HumanMessage(content=[
        {"type": "text", "text": "请简要描述这张图片里有什么。"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ])

    print("\n发送消息给模型（包含图片）...")
    try:
        response = model.invoke([message])
        print(f"\n模型回答:\n  {response.content}")
    except Exception as e:
        print(f"\n⚠️ 模型可能不支持图像输入: {e}")
        print("  提示：需要多模态模型（如 gpt-4o、glm-4v、qwen-vl）")

    print("\n关键点：")
    print("  - content 从字符串变成列表，每项是一个内容块")
    print("  - image_url 块指向图片地址，模型会自动下载并理解")
    print("  - 也可以传 base64 编码的图片，避免外部依赖")


# ============================================================================
# 示例 2：PDF 文档解析
# ============================================================================
def example_2_pdf_parsing():
    """
    示例2：PDF 文档解析

    企业场景中最常见的文件处理需求。
    解析 PDF 后用于：RAG 检索、内容提取、信息抽取。
    """
    print("\n" + "=" * 70)
    print("示例 2：PDF 文档解析")
    print("=" * 70)

    # 先用 fpdf2 生成一个示例 PDF（不依赖外部文件）
    try:
        from fpdf import FPDF
    except ImportError:
        print("\n⚠️ 未安装 fpdf2，跳过 PDF 生成")
        print("  安装: pip install fpdf2")
        return

    pdf_path = "/tmp/sample_doc.pdf"
    # Windows 路径兼容
    if sys.platform == "win32":
        pdf_path = os.path.join(os.environ.get("TEMP", "."), "sample_doc.pdf")

    print(f"\n[1] 生成示例 PDF: {pdf_path}")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="LangChain 1.0 学习手册", ln=1, align="C")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt="LangChain 是一个用于开发 LLM 应用程序的框架。")
    pdf.multi_cell(0, 10, txt="LangGraph 是 LangChain 的扩展，用于构建复杂的状态驱动工作流。")
    pdf.multi_cell(0, 10, txt="本手册涵盖基础概念、实战技巧和最佳实践。")
    pdf.output(pdf_path)
    print("  已生成 3 段内容的示例 PDF")

    # 用 PyPDFLoader 解析 PDF
    print("\n[2] 使用 PyPDFLoader 解析 PDF：")
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print(f"  共解析 {len(documents)} 页")
        for i, doc in enumerate(documents):
            text = doc.page_content.strip()
            print(f"  第 {i+1} 页: {text[:60]}...")

    except ImportError:
        print("  ⚠️ 未安装 pypdf，跳过解析")
        print("    安装: pip install pypdf")

    # 可选：使用 UnstructuredPDFLoader（更强大，但需要额外依赖）
    print("\n[3] UnstructuredPDFLoader（表格/图片识别）：")
    print("  from langchain_unstructured import UnstructuredPDFLoader")
    print("  安装: pip install \"unstructured[pdf]\"")
    print("  优势: 能识别表格、图片、标题层级等结构")

    # 清理临时文件
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    print("\n关键点：")
    print("  - PyPDFLoader：轻量、速度快，适合纯文本文档")
    print("  - UnstructuredPDFLoader：功能强大，能识别复杂布局")
    print("  - 解析后的 Document 对象可直接用于 RAG 的切片和入库")


# ============================================================================
# 示例 3：多模态混合消息
# ============================================================================
def example_3_multimodal_message():
    """
    示例3：多模态混合消息

    一条消息里同时包含文本和图片，模型需要结合两者理解。
    这是构建"看图回答"、"文档理解"应用的核心技术。
    """
    print("\n" + "=" * 70)
    print("示例 3：多模态混合消息")
    print("=" * 70)

    # 构造一个 system message 定义角色
    system_msg = SystemMessage(content="你是一个能看懂图片并回答问题的助手。")

    # 构造多模态 human message：包含文字说明 + 图片
    # 多个内容块按顺序排列，模型会综合理解
    human_msg = HumanMessage(content=[
        {"type": "text", "text": "图中展示了什么内容？请用一句话概括。"},
        {"type": "image_url", "image_url": {"url": "https://picsum.photos/400/300"}},
    ])

    print("\n消息结构：")
    print("  System: 你是一个能看懂图片并回答问题的助手。")
    print("  Human: [文本] 图中展示了什么内容？")
    print("         [图片] https://picsum.photos/400/300")

    print("\n发送给模型...")
    try:
        response = model.invoke([system_msg, human_msg])
        print(f"\n模型回答:\n  {response.content}")
    except Exception as e:
        print(f"\n⚠️ 模型可能不支持多模态: {e}")

    print("\n关键点：")
    print("  - 一条 HumanMessage 可以包含多个内容块（文本+图片+...）")
    print("  - SystemMessage 先设定角色，HumanMessage 提供具体问题")
    print("  - 多模态 RAG = 文档中的图片也参与向量检索和生成")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - 多模态与文件处理")
    print("=" * 70)

    try:
        example_1_image_understanding()
        input("\n按 Enter 继续...")

        example_2_pdf_parsing()
        input("\n按 Enter 继续...")

        example_3_multimodal_message()

        print("\n" + "=" * 70)
        print(" 完成！")
        print("=" * 70)
        print("\n核心要点：")
        print("  ✅ 多模态消息用 content 列表，每项是一个内容块")
        print("  ✅ 图片可以是 URL、base64 data URI 或本地文件")
        print("  ✅ PyPDFLoader 解析纯文本 PDF，Unstructured 解析复杂布局")
        print("\n下一步：")
        print("  phase4_projects - 综合实战项目")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
