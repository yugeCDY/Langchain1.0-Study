"""
企业级 RAG 系统 - LangGraph 节点函数
======================================

这个文件将 ingestion.py 和 retrieval.py 中的功能封装为
可直接用于 StateGraph 的节点函数和路由函数。

节点函数的约定：
- 接收完整的 State 字典
- 返回需要更新的字段字典（不需要更新的字段省略）
"""

from graph_state import IngestionState, RetrievalState
from config import RELEVANCE_THRESHOLD


# ============================================================================
# 入库图节点
# ============================================================================


def parse_node(state: IngestionState) -> dict:
    """
    入库节点：解析 PDF 文档

    尝试使用 UnstructuredPDFLoader（支持表格/图片识别），
    不可用时 fallback 到 PyPDFLoader。
    """
    from ingestion import parse_simple_pdf

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


def parse_advanced_node(state: IngestionState) -> dict:
    """入库节点：使用 UnstructuredPDFLoader 解析（支持表格/图片）"""
    from ingestion import parse_advanced_pdf

    file_path = state["file_path"]
    try:
        documents = parse_advanced_pdf(file_path)
        element_types = set(
            d.metadata.get("element_type", "unknown") for d in documents
        )
        return {
            "raw_documents": documents,
            "status": f"已解析 {len(documents)} 个元素，类型: {element_types}",
        }
    except Exception as e:
        return {
            "raw_documents": [],
            "errors": [f"高级解析失败: {e}"],
            "status": "高级解析失败",
        }


def chunk_node(state: IngestionState) -> dict:
    """入库节点：文本切片"""
    from ingestion import chunk_documents

    documents = state["raw_documents"]
    if not documents:
        return {"text_chunks": [], "status": "无文档可切片"}

    chunks = chunk_documents(documents)
    return {
        "text_chunks": chunks,
        "chunk_count": len(chunks),
        "status": f"已切片为 {len(chunks)} 块",
    }


def chunk_text_node(state: IngestionState) -> dict:
    """入库节点：只处理文本类元素"""
    from ingestion import chunk_documents, separate_by_element_type

    docs = state["raw_documents"]
    text_docs, _ = separate_by_element_type(docs)
    if not text_docs:
        return {"text_chunks": []}
    chunks = chunk_documents(text_docs)
    return {"text_chunks": chunks}


def extract_tables_node(state: IngestionState) -> dict:
    """入库节点：提取表格元素（保留原文不切片）"""
    from ingestion import separate_by_element_type

    docs = state["raw_documents"]
    _, table_docs = separate_by_element_type(docs)
    return {"table_chunks": table_docs}


def store_node(state: IngestionState) -> dict:
    """入库节点：向量化 + ChromaDB 入库"""
    from ingestion import embed_and_store

    all_chunks = state.get("text_chunks", []) + state.get("table_chunks", [])
    if not all_chunks:
        return {"status": "无切片可入库"}

    try:
        embed_and_store(all_chunks)
        return {"status": f"已入库 {len(all_chunks)} 个切片"}
    except Exception as e:
        return {"errors": [f"入库失败: {e}"], "status": "入库失败"}


# ============================================================================
# 检索图节点
# ============================================================================


def rewrite_node(state: RetrievalState) -> dict:
    """检索节点：查询改写"""
    from retrieval import rewrite_query

    queries = rewrite_query(state["original_query"])
    return {"rewritten_queries": queries}


def make_vector_search_node(vectorstore):
    """
    创建向量检索节点的工厂函数

    通过闭包捕获 vectorstore 实例，
    避免将 ChromaDB 实例存入 State。

    参数:
        vectorstore: ChromaDB 实例

    返回:
        节点函数
    """
    from retrieval import vector_search

    def _node(state: RetrievalState) -> dict:
        queries = state.get("rewritten_queries", [state["original_query"]])
        results = vector_search(queries, vectorstore)
        return {"vector_results": results}

    return _node


def make_bm25_search_node(all_documents):
    """
    创建 BM25 检索节点的工厂函数

    参数:
        all_documents: 用于构建 BM25 索引的文档集合

    返回:
        节点函数
    """
    from retrieval import bm25_search

    def _node(state: RetrievalState) -> dict:
        queries = state.get("rewritten_queries", [state["original_query"]])
        results = bm25_search(queries, all_documents)
        return {"bm25_results": results}

    return _node


def merge_node(state: RetrievalState) -> dict:
    """检索节点：合并去重"""
    from retrieval import merge_and_deduplicate

    merged = merge_and_deduplicate(
        state.get("vector_results", []),
        state.get("bm25_results", []),
    )
    return {"merged_results": merged}


def rerank_node(state: RetrievalState) -> dict:
    """检索节点：CrossEncoder 重排序"""
    from retrieval import rerank_documents

    reranked = rerank_documents(
        state["original_query"],
        state["merged_results"],
    )
    return {"reranked_results": reranked}


def grade_node(state: RetrievalState) -> dict:
    """检索节点：文档充分性评分"""
    from retrieval import grade_documents

    score = grade_documents(
        state["original_query"],
        state.get("reranked_results", []),
    )
    return {
        "relevance_score": score,
        "needs_fallback": score < RELEVANCE_THRESHOLD,
    }


def generate_node(state: RetrievalState) -> dict:
    """检索节点：生成回答"""
    from retrieval import generate_answer

    answer, sources = generate_answer(
        state["original_query"],
        state.get("reranked_results", []),
    )
    return {"answer": answer, "sources": sources}


def fallback_node(state: RetrievalState) -> dict:
    """检索节点：外部检索兜底（stub）"""
    return {
        "answer": (
            f"抱歉，知识库中没有找到足够的信息来回答"
            f"「{state['original_query']}」。"
            "建议查阅其他资料或尝试换个问法。"
        ),
        "sources": [],
    }


# ============================================================================
# 路由函数
# ============================================================================


def route_by_score(state: RetrievalState) -> str:
    """路由函数：评分够 → generate，不够 → fallback"""
    if state.get("needs_fallback", False):
        return "fallback"
    return "generate"


def route_by_content_type(state: IngestionState) -> list[str]:
    """
    路由函数：根据文档内容类型决定走哪些分支

    有表格元素 → 并行走 chunk_text + extract_tables
    纯文本 → 只走 chunk
    """
    docs = state.get("raw_documents", [])
    if not docs:
        return ["chunk"]

    has_table = any(
        d.metadata.get("element_type") == "Table" for d in docs
    )
    if has_table:
        return ["chunk_text", "extract_tables"]
    return ["chunk"]
