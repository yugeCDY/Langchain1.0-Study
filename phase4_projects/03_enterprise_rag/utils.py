"""
企业级 RAG 系统 - 工具函数
============================

打印辅助、文件工具、样本 PDF 生成等共享功能。
"""

import sys
import hashlib
from pathlib import Path

from config import SAMPLES_DIR


# ============================================================================
# Windows UTF-8 编码修复
# ============================================================================

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ============================================================================
# 打印辅助函数
# ============================================================================


def print_title(title: str):
    """打印主标题"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str):
    """打印小节标题"""
    print(f"\n{'-' * 70}")
    print(f" {title}")
    print("-" * 70)


def print_points(*points: str):
    """打印要点列表"""
    for p in points:
        print(f"  ✅ {p}")


def print_warning(msg: str):
    """打印警告"""
    print(f"  ⚠️  {msg}")


def print_tip(msg: str):
    """打印提示"""
    print(f"  💡 {msg}")


# ============================================================================
# 文件与数据工具
# ============================================================================


def ensure_dirs():
    """确保所有需要的目录存在"""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def content_hash(text: str) -> str:
    """计算文本内容的哈希值，用于去重"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def truncate_text(text: str, max_len: int = 100) -> str:
    """截断文本用于安全预览"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def format_sources(documents) -> list[dict]:
    """
    从文档列表中提取来源信息

    返回:
        来源字典列表，每个含 source、page、preview 字段
    """
    sources = []
    for doc in documents:
        meta = doc.metadata
        sources.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "preview": truncate_text(doc.page_content, 80),
        })
    return sources


# ============================================================================
# 样本文档生成
# ============================================================================


def create_sample_documents():
    """
    生成用于演示的示例 PDF 文档

    使用 fpdf2 库生成两个样本文件：
    1. tech_report_sample.pdf - 技术报告（含表格）
    2. mixed_content_sample.pdf - 混合内容文档（中英文）
    """
    try:
        from fpdf import FPDF
    except ImportError:
        print_warning("fpdf2 未安装，跳过样本 PDF 生成")
        print_tip("安装方式：pip install fpdf2")
        return

    ensure_dirs()

    # ---- 样本 1：技术报告 ----
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2024 AI Technology Report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6,
        "This report provides an overview of major AI developments in 2024. "
        "Large Language Models (LLMs) have seen significant improvements in "
        "reasoning capabilities, multilingual understanding, and tool use."
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Key Metrics Comparison", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # 表格
    pdf.set_font("Helvetica", size=10)
    headers = ["Model", "Parameters", "Context Length", "MMLU Score"]
    col_widths = [40, 35, 40, 35]

    # 表头
    pdf.set_fill_color(200, 220, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, fill=True)
    pdf.ln()

    # 表格内容
    rows = [
        ["GPT-4o", "1.8T (est.)", "128K", "88.7%"],
        ["Claude 3.5", "Unknown", "200K", "88.3%"],
        ["Gemini Ultra", "Unknown", "1M", "90.0%"],
        ["Llama 3.1", "405B", "128K", "85.2%"],
    ]
    for row in rows:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 7, cell, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6,
        "The table above shows that Gemini Ultra achieves the highest MMLU score "
        "at 90.0%, while Claude 3.5 offers the longest context window at 200K tokens. "
        "Open-source models like Llama 3.1 are closing the gap with proprietary models."
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Retrieval-Augmented Generation", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6,
        "RAG (Retrieval-Augmented Generation) combines information retrieval with "
        "text generation. The system first retrieves relevant documents from a knowledge "
        "base, then uses an LLM to generate answers grounded in those documents.\n\n"
        "Key RAG improvements in 2024:\n"
        "- Hybrid search combining BM25 and vector retrieval\n"
        "- Cross-encoder reranking for higher precision\n"
        "- Multi-modal RAG handling tables, images, and structured data\n"
        "- Query rewriting and expansion techniques"
    )

    pdf.output(str(SAMPLES_DIR / "tech_report_sample.pdf"))

    # ---- 样本 2：中英文混合内容 ----
    pdf2 = FPDF()
    pdf2.add_page()

    pdf2.set_font("Helvetica", "B", 14)
    pdf2.cell(0, 10, "LangChain 1.0 Technical Guide", new_x="LMARGIN", new_y="NEXT")
    pdf2.ln(3)

    pdf2.set_font("Helvetica", size=10)
    pdf2.multi_cell(0, 5,
        "LangChain 1.0 introduces several major changes:\n\n"
        "1. New `init_chat_model()` API for unified model initialization\n"
        "2. Middleware architecture for fine-grained execution control\n"
        "3. Built on LangGraph runtime for persistence and streaming\n"
        "4. Semantic versioning with API stability guarantees"
    )
    pdf2.ln(3)

    # 表格：API 对比
    pdf2.set_font("Helvetica", "B", 12)
    pdf2.cell(0, 8, "API Migration Table", new_x="LMARGIN", new_y="NEXT")
    pdf2.ln(2)

    pdf2.set_font("Helvetica", size=9)
    headers2 = ["Old API (0.x)", "New API (1.0)", "Notes"]
    col_widths2 = [50, 50, 55]

    pdf2.set_fill_color(255, 230, 200)
    for i, h in enumerate(headers2):
        pdf2.cell(col_widths2[i], 7, h, border=1, fill=True)
    pdf2.ln()

    rows2 = [
        ["ChatOpenAI()", "init_chat_model()", "Unified init"],
        ["AgentExecutor", "create_agent()", "Simplified API"],
        ["ConversationBuffer", "InMemorySaver", "Memory refactor"],
        ["RetrievalQA", "RAG Graph", "Graph-based RAG"],
    ]
    for row in rows2:
        for i, cell in enumerate(row):
            pdf2.cell(col_widths2[i], 6, cell, border=1)
        pdf2.ln()

    pdf2.ln(5)
    pdf2.set_font("Helvetica", size=10)
    pdf2.multi_cell(0, 5,
        "The migration from 0.x to 1.0 requires updating import paths "
        "and API calls. The init_chat_model() function supports multiple "
        "providers through a unified interface using the 'provider:model' format."
    )

    pdf2.output(str(SAMPLES_DIR / "mixed_content_sample.pdf"))

    print(f"  [OK] 样本 PDF 已生成:")
    print(f"    - {SAMPLES_DIR / 'tech_report_sample.pdf'}")
    print(f"    - {SAMPLES_DIR / 'mixed_content_sample.pdf'}")
