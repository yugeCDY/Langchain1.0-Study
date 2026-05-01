"""
企业级 RAG 系统 - LangGraph 状态定义
======================================

定义入库图和检索图的共享状态（TypedDict）。
每个节点接收完整状态，返回需要更新的字段。
使用 Annotated + reducer 实现并行结果合并。
"""

from typing import Annotated
from typing import TypedDict
import operator

from langchain_core.documents import Document


# ============================================================================
# 文档入库图状态
# ============================================================================

class IngestionState(TypedDict):
    """
    入库流水线的共享状态

    file_path: 待处理的文件路径
    raw_documents: 解析后的原始文档页/元素
    text_chunks: 文本切片（reducer: 并行路径追加）
    table_chunks: 表格切片（reducer: 并行路径追加）
    chunk_count: 总切片数
    status: 流水线状态文本
    errors: 错误日志（reducer: 追加）
    """
    file_path: str
    raw_documents: list[Document]
    text_chunks: Annotated[list[Document], operator.add]
    table_chunks: Annotated[list[Document], operator.add]
    chunk_count: int
    status: str
    errors: Annotated[list[str], operator.add]


# ============================================================================
# 检索生成图状态
# ============================================================================

class RetrievalState(TypedDict):
    """
    检索+生成流水线的共享状态

    original_query: 用户原始提问
    rewritten_queries: LLM 改写后的查询变体列表
    vector_results: 向量检索结果（reducer: 并行追加）
    bm25_results: BM25 检索结果（reducer: 并行追加）
    merged_results: 去重合并后的文档列表
    reranked_results: 重排序后的文档列表
    relevance_score: 文档充分性评分（0-1）
    context: 拼接给 LLM 的最终上下文
    answer: LLM 生成的回答
    sources: 来源引用列表
    needs_fallback: 是否需要外部检索兜底
    """
    original_query: str
    rewritten_queries: list[str]
    vector_results: Annotated[list[Document], operator.add]
    bm25_results: Annotated[list[Document], operator.add]
    merged_results: list[Document]
    reranked_results: list[Document]
    relevance_score: float
    context: str
    answer: str
    sources: list[dict]
    needs_fallback: bool
