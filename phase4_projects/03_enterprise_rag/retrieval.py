"""
企业级 RAG 系统 - 检索引擎模块
================================

负责查询改写、多路检索、重排序、文档评分、答案生成的完整检索链。

核心组件：
- rewrite_query()       LLM 驱动的查询改写
- vector_search()       向量语义检索
- bm25_search()         BM25 关键词检索
- merge_and_deduplicate()  去重合并
- rerank_documents()    CrossEncoder 重排序
- grade_documents()     LLM 文档充分性评估
- generate_answer()     基于上下文生成回答
- build_retrieval_graph()  构建检索 LangGraph
"""

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import (
    RETRIEVAL_K,
    RERANK_TOP_N,
    ENSEMBLE_WEIGHTS,
    RELEVANCE_THRESHOLD,
    get_llm,
    get_embeddings,
    get_chroma_store,
)
from graph_state import RetrievalState


# ============================================================================
# 查询改写
# ============================================================================


def rewrite_query(query: str, llm=None) -> list[str]:
    """
    使用 LLM 改写查询为多个变体，提升召回率

    改写策略：从不同角度表达同一问题，覆盖更多语义空间。
    例如 "Python 性能优化" 可能改写为：
    - "如何提升 Python 代码执行速度"
    - "Python 性能瓶颈和解决方案"
    - "Python performance optimization best practices"

    参数:
        query: 用户原始查询
        llm: LLM 实例（可选，默认用 config.get_llm()）

    返回:
        包含原始查询 + 改写变体的列表
    """
    if llm is None:
        llm = get_llm()

    prompt = f"""你是一个查询改写助手。请将用户的查询改写为 2 个不同角度的变体，
保持原始查询的核心意图不变，但用不同的表达方式。

原始查询：{query}

请直接输出改写后的查询，每行一个，不要编号，不要额外解释。"""

    response = llm.invoke(prompt)
    variants = [
        line.strip()
        for line in response.content.strip().split("\n")
        if line.strip()
    ]

    # 始终包含原始查询
    all_queries = [query] + variants[:2]
    return all_queries


# ============================================================================
# 多路检索
# ============================================================================


def vector_search(
    queries: list[str],
    vectorstore: Chroma,
    k: int = RETRIEVAL_K,
) -> list[Document]:
    """
    向量语义检索：对每个查询变体执行 similarity_search，合并结果

    参数:
        queries: 查询列表（原始 + 改写变体）
        vectorstore: ChromaDB 向量库实例
        k: 每个查询召回的文档数

    返回:
        去重后的文档列表
    """
    seen_hashes = set()
    results = []

    for q in queries:
        docs = vectorstore.similarity_search(q, k=k)
        for doc in docs:
            h = hash(doc.page_content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                results.append(doc)

    return results


def bm25_search(
    queries: list[str],
    documents: list[Document],
    k: int = RETRIEVAL_K,
) -> list[Document]:
    """
    BM25 关键词检索：基于词频的精确匹配检索

    BM25 不需要 Embedding，适合精确关键词匹配场景。
    需要传入完整文档集合来构建 BM25 索引。

    参数:
        queries: 查询列表
        documents: 用于构建 BM25 索引的文档集合
        k: 每个查询召回的文档数

    返回:
        去重后的文档列表
    """
    from langchain_community.retrievers import BM25Retriever

    bm25 = BM25Retriever.from_documents(documents, k=k)

    seen_hashes = set()
    results = []

    for q in queries:
        docs = bm25.invoke(q)
        for doc in docs:
            h = hash(doc.page_content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                results.append(doc)

    return results


# ============================================================================
# 结果合并
# ============================================================================


def merge_and_deduplicate(
    vector_results: list[Document],
    bm25_results: list[Document],
) -> list[Document]:
    """
    合并向量检索和 BM25 检索结果，按内容去重

    去重策略：基于 page_content 的哈希值。
    保留先出现的文档（通常来自更相关的路径）。

    参数:
        vector_results: 向量检索结果
        bm25_results: BM25 检索结果

    返回:
        去重合并后的文档列表
    """
    from utils import content_hash

    seen = set()
    merged = []

    # 向量结果优先
    for doc in vector_results:
        h = content_hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    for doc in bm25_results:
        h = content_hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    return merged


# ============================================================================
# 重排序
# ============================================================================


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """
    使用 CrossEncoder 对检索结果进行重排序

    CrossEncoder 比双塔模型（bi-encoder）更精确：
    - 双塔：query 和 doc 分别编码，计算向量距离（快但粗糙）
    - CrossEncoder：query 和 doc 拼接后一起编码（慢但精确）

    重排序流程：query + each doc → cross-encoder → 按分数排序 → 取 top_n

    参数:
        query: 用户查询
        documents: 待重排序的文档列表
        top_n: 保留的文档数

    返回:
        重排序后的文档列表（按相关性降序）
    """
    if not documents:
        return []

    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

    # 加载 CrossEncoder 模型（首次运行会自动下载约 80MB）
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    # reranker.invoke() 返回排序后的 Document 列表
    reranked = reranker.compress_documents(
        documents=documents,
        query=query,
    )

    return reranked


# ============================================================================
# 文档评分（LLM-as-Judge）
# ============================================================================


def grade_documents(
    query: str,
    documents: list[Document],
    llm=None,
) -> float:
    """
    使用 LLM 评估检索到的文档是否足以回答用户问题

    评分标准：0.0 ~ 1.0
    - >= 0.7: 文档充分，可以直接生成回答
    - < 0.7: 文档不足，可能需要外部检索兜底

    这是 CRAG (Corrective RAG) 的核心思想：
    不盲目信任检索结果，先评估再决定下一步。

    参数:
        query: 用户查询
        documents: 重排序后的文档列表
        llm: LLM 实例

    返回:
        充分性评分（0.0 ~ 1.0）
    """
    if llm is None:
        llm = get_llm()

    if not documents:
        return 0.0

    # 拼接文档内容作为评估依据
    docs_text = "\n\n---\n\n".join(
        f"[文档 {i+1}]\n{doc.page_content[:300]}"
        for i, doc in enumerate(documents[:5])  # 最多评估 5 个文档
    )

    prompt = f"""请评估以下检索到的文档是否能充分回答用户的问题。

用户问题：{query}

检索到的文档：
{docs_text}

请只输出一个 0.0 到 1.0 之间的数字评分：
- 1.0 = 完全可以回答
- 0.7 = 基本可以回答，但可能缺少细节
- 0.4 = 有部分相关信息
- 0.0 = 完全无关

只输出数字，不要其他内容。"""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.5  # 解析失败时给中间分


# ============================================================================
# 答案生成
# ============================================================================


def generate_answer(
    query: str,
    documents: list[Document],
    llm=None,
) -> tuple[str, list[dict]]:
    """
    基于检索到的文档生成回答

    生成策略：
    1. 将文档拼接为上下文
    2. 构造 system prompt 要求基于上下文回答
    3. LLM 生成带来源引用的回答

    参数:
        query: 用户查询
        documents: 检索到的文档
        llm: LLM 实例

    返回:
        (回答文本, 来源引用列表)
    """
    if llm is None:
        llm = get_llm()

    from utils import format_sources

    context = "\n\n---\n\n".join(
        f"[来源 {i+1}: {doc.metadata.get('source', 'unknown')}, "
        f"第 {doc.metadata.get('page', '?')} 页]\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )

    prompt = f"""请根据以下检索到的文档内容回答用户问题。
要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确说明
3. 在回答中标注信息来源（如"根据文档1..."）

检索到的文档：
{context}

用户问题：{query}"""

    response = llm.invoke(prompt)
    sources = format_sources(documents)

    return response.content, sources


# ============================================================================
# LangGraph 检索工作流
# ============================================================================

from langgraph.graph import StateGraph, START, END


def _rewrite_node(state: RetrievalState) -> dict:
    """检索图节点：查询改写"""
    queries = rewrite_query(state["original_query"])
    return {"rewritten_queries": queries}


def _merge_node(state: RetrievalState) -> dict:
    """检索图节点：合并去重"""
    merged = merge_and_deduplicate(
        state.get("vector_results", []),
        state.get("bm25_results", []),
    )
    return {"merged_results": merged}


def _rerank_node(state: RetrievalState) -> dict:
    """检索图节点：重排序"""
    reranked = rerank_documents(
        state["original_query"],
        state["merged_results"],
    )
    return {"reranked_results": reranked}


def _grade_node(state: RetrievalState) -> dict:
    """检索图节点：文档充分性评分"""
    score = grade_documents(
        state["original_query"],
        state.get("reranked_results", []),
    )
    return {
        "relevance_score": score,
        "needs_fallback": score < RELEVANCE_THRESHOLD,
    }


def _generate_node(state: RetrievalState) -> dict:
    """检索图节点：生成回答"""
    answer, sources = generate_answer(
        state["original_query"],
        state.get("reranked_results", []),
    )
    return {"answer": answer, "sources": sources}


def _fallback_node(state: RetrievalState) -> dict:
    """
    检索图节点：外部检索兜底（stub）

    在生产环境中，这里可以接入 Tavily/Bing 等网络搜索。
    学习仓库中仅演示模式，不实际调用外部搜索。
    """
    return {
        "answer": (
            f"抱歉，知识库中没有找到足够的信息来回答「{state['original_query']}」。"
            "建议查阅其他资料或尝试换个问法。"
        ),
        "sources": [],
        "needs_fallback": True,
    }


def _route_by_score(state: RetrievalState) -> str:
    """路由函数：根据文档评分决定走生成还是兜底"""
    if state.get("needs_fallback", False):
        return "fallback"
    return "generate"


def build_retrieval_graph(
    vectorstore: Chroma,
    all_documents: list[Document],
) -> StateGraph:
    """
    构建检索生成 LangGraph 工作流

    流程：
        START → rewrite → [vector_search + bm25_search] → merge
        → rerank → grade → generate / fallback → END

    并行部分（fan-out）：vector_search 和 bm25_search 并行执行
    合并部分（fan-in）：通过 reducer 自动合并结果到 merge 节点

    参数:
        vectorstore: ChromaDB 实例
        all_documents: 用于构建 BM25 索引的完整文档集

    返回:
        编译后的 StateGraph
    """
    # 通过闭包捕获 vectorstore 和 documents
    def _vector_node(state: RetrievalState) -> dict:
        queries = state.get("rewritten_queries", [state["original_query"]])
        results = vector_search(queries, vectorstore)
        return {"vector_results": results}

    def _bm25_node(state: RetrievalState) -> dict:
        queries = state.get("rewritten_queries", [state["original_query"]])
        results = bm25_search(queries, all_documents)
        return {"bm25_results": results}

    builder = StateGraph(RetrievalState)

    # 添加节点
    builder.add_node("rewrite", _rewrite_node)
    builder.add_node("vector_search", _vector_node)
    builder.add_node("bm25_search", _bm25_node)
    builder.add_node("merge", _merge_node)
    builder.add_node("rerank", _rerank_node)
    builder.add_node("grade", _grade_node)
    builder.add_node("generate", _generate_node)
    builder.add_node("fallback", _fallback_node)

    # 定义边
    builder.add_edge(START, "rewrite")

    # fan-out：rewrite 后并行执行两路检索
    builder.add_edge("rewrite", "vector_search")
    builder.add_edge("rewrite", "bm25_search")

    # fan-in：两路检索结果汇入 merge（需要 reducer）
    builder.add_edge("vector_search", "merge")
    builder.add_edge("bm25_search", "merge")

    builder.add_edge("merge", "rerank")
    builder.add_edge("rerank", "grade")

    # 条件路由：评分够走 generate，不够走 fallback
    builder.add_conditional_edges("grade", _route_by_score)

    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    return builder.compile()
