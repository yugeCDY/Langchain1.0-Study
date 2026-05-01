"""
企业级 RAG 系统 - 文档入库模块
================================

负责文档解析、切片、Embedding、入库的完整流水线。

支持分层处理策略：
- Tier 1：简单 PDF → PyPDFLoader + RecursiveCharacterTextSplitter
- Tier 2：含表格/图片 → UnstructuredPDFLoader（可选，需额外依赖）

核心函数：
- parse_simple_pdf()    使用 PyPDFLoader 解析
- parse_advanced_pdf()  使用 UnstructuredPDFLoader 解析（可选）
- chunk_documents()     文本切片
- embed_and_store()     向量化 + ChromaDB 入库
- build_ingestion_graph()  构建 LangGraph 入库工作流
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    get_embeddings,
    get_chroma_store,
)
from graph_state import IngestionState


# ============================================================================
# 文档解析
# ============================================================================


def parse_simple_pdf(file_path: str) -> list[Document]:
    """
    使用 PyPDFLoader 解析 PDF 文件

    PyPDFLoader 是最轻量的 PDF 解析器，适合纯文本 PDF。
    每页返回一个 Document，metadata 包含页码和文件路径。

    参数:
        file_path: PDF 文件路径

    返回:
        文档页列表
    """
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # PyPDFLoader 自动为每页设置 metadata: {source, page}
    return pages


def parse_advanced_pdf(file_path: str) -> list[Document]:
    """
    使用 UnstructuredPDFLoader 解析复杂 PDF（含表格/图片）

    UnstructuredPDFLoader 能识别文档中的结构化元素：
    - NarrativeText: 正文段落
    - Table: 表格
    - Image: 图片
    - Title / Header: 标题

    需要额外依赖：pip install "unstructured[pdf]"
    如果未安装，会 fallback 到 parse_simple_pdf()

    参数:
        file_path: PDF 文件路径

    返回:
        结构化文档元素列表
    """
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader

        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",  # elements 模式：按元素类型拆分
        )
        elements = loader.load()

        # 为每个元素标注类型，方便后续按类型处理
        for elem in elements:
            # unstructured 的 category 存在 metadata["category"] 中
            elem.metadata["element_type"] = elem.metadata.get(
                "category", "Unknown"
            )

        return elements

    except ImportError:
        print("  ⚠️  unstructured 未安装，使用 PyPDFLoader 替代")
        print("  💡 安装方式：pip install \"unstructured[pdf]\"")
        return parse_simple_pdf(file_path)


# ============================================================================
# 文档切片
# ============================================================================


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    将文档切片为适合检索的小段

    使用 RecursiveCharacterTextSplitter，支持中英文分隔符：
    - 优先按段落（\\n\\n）切分
    - 再按句子（\\n、。、！、？）切分
    - 最后按空格/字符切分

    chunk_overlap 确保相邻块有重叠，防止关键信息被截断。

    参数:
        documents: 原始文档列表
        chunk_size: 目标块大小（字符数）
        chunk_overlap: 重叠大小

    返回:
        切片后的文档列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def separate_by_element_type(
    documents: list[Document],
) -> tuple[list[Document], list[Document]]:
    """
    按元素类型分离文档：文本元素 vs 表格元素

    未标注 element_type 的文档（来自 PyPDFLoader）全部视为文本。

    参数:
        documents: 文档列表

    返回:
        (text_docs, table_docs) 元组
    """
    text_docs = []
    table_docs = []

    for doc in documents:
        elem_type = doc.metadata.get("element_type", "text")
        if elem_type == "Table":
            table_docs.append(doc)
        else:
            # NarrativeText, Title, Header, Unknown 等都归为文本
            text_docs.append(doc)

    return text_docs, table_docs


# ============================================================================
# Embedding + 入库
# ============================================================================


def embed_and_store(
    chunks: list[Document],
    collection_name: str = "enterprise_rag",
) -> Chroma:
    """
    将文档切片向量化并存入 ChromaDB

    流程：chunks → Embedding → ChromaDB.upsert
    ChromaDB 自动持久化到 config.CHROMA_DIR。

    参数:
        chunks: 文档切片列表
        collection_name: ChromaDB 集合名

    返回:
        ChromaDB 实例
    """
    embeddings = get_embeddings()
    vectorstore = get_chroma_store(embeddings, collection_name)

    # 为每个 chunk 生成唯一 ID（基于内容哈希 + 来源），避免重复入库
    ids = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        content_hash = hashlib.md5(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:8]
        ids.append(f"{Path(source).stem}_p{page}_{content_hash}_{i}")

    # ChromaDB 的 add_documents 支持批量写入
    vectorstore.add_documents(documents=chunks, ids=ids)

    return vectorstore


def reset_collection(collection_name: str = "enterprise_rag"):
    """
    清空指定集合（用于重复运行示例时避免重复数据）

    参数:
        collection_name: 要清空的集合名
    """
    import chromadb

    from config import CHROMA_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(collection_name)
        print(f"  [OK] 已清空集合: {collection_name}")
    except Exception:
        pass  # 集合不存在时忽略


# ============================================================================
# LangGraph 入库工作流
# ============================================================================

import hashlib

from langgraph.graph import StateGraph, START, END


def _parse_node(state: IngestionState) -> dict:
    """入库图节点：解析文档"""
    file_path = state["file_path"]
    try:
        documents = parse_simple_pdf(file_path)
        return {
            "raw_documents": documents,
            "status": f"已解析 {len(documents)} 页",
        }
    except Exception as e:
        return {
            "raw_documents": [],
            "errors": [f"解析失败: {e}"],
            "status": "解析失败",
        }


def _chunk_node(state: IngestionState) -> dict:
    """入库图节点：切片"""
    documents = state["raw_documents"]
    if not documents:
        return {"text_chunks": [], "status": "无文档可切片"}

    chunks = chunk_documents(documents)
    return {
        "text_chunks": chunks,
        "chunk_count": len(chunks),
        "status": f"已切片为 {len(chunks)} 块",
    }


def _store_node(state: IngestionState) -> dict:
    """入库图节点：向量化 + 入库"""
    # 合并文本切片和表格切片
    all_chunks = state.get("text_chunks", []) + state.get("table_chunks", [])
    if not all_chunks:
        return {"status": "无切片可入库"}

    try:
        embed_and_store(all_chunks)
        return {"status": f"已入库 {len(all_chunks)} 个切片"}
    except Exception as e:
        return {
            "errors": [f"入库失败: {e}"],
            "status": "入库失败",
        }


def build_ingestion_graph() -> StateGraph:
    """
    构建文档入库 LangGraph 工作流

    流程：START → parse → chunk → store → END

    这是一个简单的线性图，演示如何用 StateGraph 编排入库流水线。
    更复杂的图可以在 parse 后按元素类型分支处理。

    返回:
        编译后的 StateGraph
    """
    builder = StateGraph(IngestionState)

    # 添加节点
    builder.add_node("parse", _parse_node)
    builder.add_node("chunk", _chunk_node)
    builder.add_node("store", _store_node)

    # 定义边：线性流水线
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "chunk")
    builder.add_edge("chunk", "store")
    builder.add_edge("store", END)

    return builder.compile()


def build_multimodal_ingestion_graph() -> StateGraph:
    """
    构建多模态入库图（支持表格/图片分类处理）

    流程：
        START → parse → [chunk_text + extract_tables] → store → END
                          ↑ 并行 fan-out        ↑ fan-in (reducer)

    parse 解析后的文档按 element_type 分流：
    - 文本类元素 → chunk_text 节点（切片）
    - 表格类元素 → extract_tables 节点（保留原文）
    两个路径的结果通过 reducer 自动合并，最后统一入库。

    返回:
        编译后的 StateGraph
    """
    def _route_by_content(state: IngestionState) -> list[str]:
        """路由函数：根据内容类型决定走哪些分支"""
        docs = state.get("raw_documents", [])
        if not docs:
            return ["chunk"]

        # 简单 PDF 只有文本，走 chunk 分支
        # 有 element_type 标注的走双分支
        has_table = any(
            d.metadata.get("element_type") == "Table" for d in docs
        )

        if has_table:
            return ["chunk_text", "extract_tables"]
        return ["chunk"]

    def _chunk_text_node(state: IngestionState) -> dict:
        """只处理文本元素的切片"""
        docs = state["raw_documents"]
        text_docs, _ = separate_by_element_type(docs)
        if not text_docs:
            return {"text_chunks": []}
        chunks = chunk_documents(text_docs)
        return {"text_chunks": chunks}

    def _extract_tables_node(state: IngestionState) -> dict:
        """提取表格元素，保留原文不切片"""
        docs = state["raw_documents"]
        _, table_docs = separate_by_element_type(docs)
        return {"table_chunks": table_docs}

    builder = StateGraph(IngestionState)

    builder.add_node("parse", _parse_node)
    builder.add_node("chunk_text", _chunk_text_node)
    builder.add_node("extract_tables", _extract_tables_node)
    builder.add_node("chunk", _chunk_node)  # fallback：无表格时走这个
    builder.add_node("store", _store_node)

    builder.add_edge(START, "parse")
    builder.add_conditional_edges("parse", _route_by_content)
    # chunk_text、extract_tables、chunk 的输出都汇入 store
    builder.add_edge("chunk_text", "store")
    builder.add_edge("extract_tables", "store")
    builder.add_edge("chunk", "store")
    builder.add_edge("store", END)

    return builder.compile()
