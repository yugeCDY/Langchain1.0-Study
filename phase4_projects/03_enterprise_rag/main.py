"""
LangChain 1.0 - 企业级 RAG 系统 (Enterprise RAG System)
=========================================================

本模块重点讲解：
1. 文档入库流水线（解析 → 切片 → Embedding → ChromaDB）
2. LangGraph 编排入库和检索工作流
3. 多模态文档处理（表格提取）
4. Embedding 模型选择与对比
5. 混合检索 + CrossEncoder 重排序
6. CRAG 文档评分 + 自适应生成
7. 端到端企业级 RAG 完整系统
"""

import os
import sys

# Windows UTF-8 编码修复
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from config import (
    API_KEY,
    BASE_URL,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_DEFAULT,
    EMBEDDING_MODEL_MULTILINGUAL,
    get_llm,
    get_embeddings,
    get_chroma_store,
    SAMPLES_DIR,
    CHROMA_DIR,
)
from utils import (
    print_title,
    print_section,
    print_points,
    print_warning,
    print_tip,
    ensure_dirs,
    truncate_text,
    format_sources,
    create_sample_documents,
)


# ============================================================================
# 环境配置
# ============================================================================

load_dotenv()

if not API_KEY:
    raise ValueError("请先在 .env 中设置 OPENAI_API_KEY")

model = init_chat_model(
    LLM_MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ============================================================================
# 示例 1：基础文档入库流水线
# ============================================================================


def example_1_basic_ingestion():
    """
    示例1：基础文档入库流水线

    核心流程：PDF → PyPDFLoader → RecursiveCharacterTextSplitter
             → HuggingFaceEmbeddings → ChromaDB

    这是最基础的 RAG 入库链，不涉及 LangGraph，
    先理解每个组件的作用，再在后续示例中用图编排。
    """
    print_title("示例 1：基础文档入库流水线")

    # 生成样本 PDF
    create_sample_documents()

    pdf_path = str(SAMPLES_DIR / "tech_report_sample.pdf")
    if not os.path.exists(pdf_path):
        print_warning("样本 PDF 不存在，跳过此示例")
        return None

    # Step 1: 解析 PDF
    print_section("Step 1: 解析 PDF")
    from ingestion import parse_simple_pdf

    pages = parse_simple_pdf(pdf_path)
    print(f"  解析结果: {len(pages)} 页")
    for i, page in enumerate(pages[:2]):
        print(f"  - 第 {page.metadata.get('page', i)} 页: "
              f"{truncate_text(page.page_content, 60)}")

    # Step 2: 文本切片
    print_section("Step 2: 文本切片")
    from ingestion import chunk_documents

    chunks = chunk_documents(pages)
    print(f"  切片结果: {len(chunks)} 块 (chunk_size=500, overlap=50)")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  - 块 {i}: {truncate_text(chunk.page_content, 60)}")

    # Step 3: Embedding + ChromaDB 入库
    print_section("Step 3: Embedding + 入库")
    from ingestion import embed_and_store, reset_collection

    # 先清空集合，避免重复数据
    reset_collection("example1_basic")

    vectorstore = embed_and_store(chunks, collection_name="example1_basic")
    print(f"  向量库文档数: {vectorstore._collection.count()}")

    # Step 4: 验证检索
    print_section("Step 4: 验证检索效果")
    results = vectorstore.similarity_search("RAG 技术", k=2)
    print(f"  查询: 'RAG 技术' → 命中 {len(results)} 条")
    for doc in results:
        print(f"  - {truncate_text(doc.page_content, 80)}")

    print("\n关键点：")
    print_points(
        "PyPDFLoader 按页解析，每页一个 Document",
        "RecursiveCharacterTextSplitter 优先按段落切分",
        "ChromaDB 自动持久化到本地目录",
        "similarity_search 返回语义最相关的文档",
    )

    return vectorstore


# ============================================================================
# 示例 2：LangGraph 入库工作流
# ============================================================================


def example_2_graph_ingestion():
    """
    示例2：用 LangGraph StateGraph 编排入库流水线

    将示例 1 的步骤封装为一个 StateGraph：
    START → parse → chunk → store → END

    对比示例 1 的手动调用，图编排的优势：
    - 流程可视化（可以生成图结构图）
    - 状态自动传递（不需要手动传参）
    - 易于扩展（加节点只需 add_node + add_edge）
    """
    print_title("示例 2：LangGraph 入库工作流")

    from ingestion import reset_collection, build_ingestion_graph

    pdf_path = str(SAMPLES_DIR / "tech_report_sample.pdf")
    if not os.path.exists(pdf_path):
        print_warning("样本 PDF 不存在，跳过")
        return None

    # 构建入库图
    graph = build_ingestion_graph()
    print("  入库图结构: START → parse → chunk → store → END")

    # 清空旧数据
    reset_collection("enterprise_rag")

    # 执行入库图
    print_section("执行入库图")
    result = graph.invoke({
        "file_path": pdf_path,
        "raw_documents": [],
        "text_chunks": [],
        "table_chunks": [],
        "chunk_count": 0,
        "status": "初始化",
        "errors": [],
    })

    print(f"  状态: {result['status']}")
    print(f"  切片数: {result['chunk_count']}")
    if result.get("errors"):
        for err in result["errors"]:
            print_warning(err)

    # 验证入库结果
    vs = get_chroma_store(collection_name="enterprise_rag")
    print(f"  ChromaDB 文档数: {vs._collection.count()}")

    print("\n关键点：")
    print_points(
        "StateGraph 自动管理状态在节点间传递",
        "IngestionState TypedDict 定义了图的数据结构",
        "线性图适合简单的顺序流水线",
        "graph.invoke() 执行完整流程并返回最终状态",
    )


# ============================================================================
# 示例 3：多模态文档处理
# ============================================================================


def example_3_multimodal_processing():
    """
    示例3：多模态文档处理（表格/图片识别）

    使用 UnstructuredPDFLoader 识别文档中的结构化元素：
    - NarrativeText: 正文
    - Table: 表格
    - Title: 标题

    表格元素不切片（保留完整结构），文本元素正常切片。
    两种元素通过不同路径处理后统一入库。

    注意：需要 pip install "unstructured[pdf]"
    未安装时自动 fallback 到 PyPDFLoader。
    """
    print_title("示例 3：多模态文档处理")

    pdf_path = str(SAMPLES_DIR / "tech_report_sample.pdf")
    if not os.path.exists(pdf_path):
        print_warning("样本 PDF 不存在，跳过")
        return None

    # 尝试高级解析
    print_section("解析文档结构")
    from ingestion import parse_advanced_pdf, separate_by_element_type

    elements = parse_advanced_pdf(pdf_path)
    text_docs, table_docs = separate_by_element_type(elements)

    print(f"  总元素数: {len(elements)}")
    print(f"  文本元素: {len(text_docs)}")
    print(f"  表格元素: {len(table_docs)}")

    # 统计元素类型
    from collections import Counter
    type_counts = Counter(
        d.metadata.get("element_type", "unknown") for d in elements
    )
    print(f"  元素类型分布: {dict(type_counts)}")

    if table_docs:
        print_section("提取的表格内容")
        for i, table in enumerate(table_docs[:2]):
            print(f"  表格 {i+1}: {truncate_text(table.page_content, 80)}")

    # 使用多模态入库图
    print_section("多模态入库图")
    from ingestion import build_multimodal_ingestion_graph, reset_collection

    reset_collection("example3_multimodal")

    graph = build_multimodal_ingestion_graph()
    # ↑ 这个图在 parse 后按 element_type 分流，表格和文本走不同路径

    # 重新编译使用不同的集合名
    # 这里直接用函数式演示流程
    from ingestion import chunk_documents, embed_and_store

    text_chunks = chunk_documents(text_docs) if text_docs else []
    all_chunks = text_chunks + table_docs
    if all_chunks:
        reset_collection("example3_multimodal")
        embed_and_store(all_chunks, collection_name="example3_multimodal")
        print(f"  入库切片数: {len(all_chunks)} (文本 {len(text_chunks)} + 表格 {len(table_docs)})")

    print("\n关键点：")
    print_points(
        "UnstructuredPDFLoader 识别文档中的结构化元素",
        "表格元素保留原文不切片，避免破坏表格结构",
        "未安装 unstructured 时自动 fallback 到 PyPDFLoader",
        "多模态入库图用条件路由按元素类型分流处理",
    )


# ============================================================================
# 示例 4：Embedding 模型对比
# ============================================================================


def example_4_embedding_comparison():
    """
    示例4：对比不同 Embedding 模型的检索质量

    对比：
    - all-MiniLM-L6-v2：英文为主，384 维，轻量
    - paraphrase-multilingual-MiniLM-L12-v2：50+ 语言，384 维

    用同一个查询测试两个模型的召回效果，
    帮助理解 Embedding 选择对检索质量的影响。
    """
    print_title("示例 4：Embedding 模型对比")

    queries = ["RAG 检索增强生成", "model performance comparison"]

    print(f"  测试查询: {queries}")

    for model_name in [EMBEDDING_MODEL_DEFAULT, EMBEDDING_MODEL_MULTILINGUAL]:
        print_section(f"模型: {model_name.split('/')[-1]}")

        embeddings = get_embeddings(model_name)
        # 用临时内存集合测试（不持久化）
        from langchain_chroma import Chroma

        # 加载文档
        pdf_path = str(SAMPLES_DIR / "tech_report_sample.pdf")
        if not os.path.exists(pdf_path):
            print_warning("样本 PDF 不存在，跳过")
            return

        from ingestion import parse_simple_pdf, chunk_documents
        pages = parse_simple_pdf(pdf_path)
        chunks = chunk_documents(pages)

        # 创建临时向量库
        temp_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"test_{model_name.split('/')[-1]}",
        )

        for query in queries:
            results = temp_store.similarity_search_with_score(query, k=2)
            print(f"  查询: '{query}'")
            for doc, score in results:
                # ChromaDB 的 score 是距离（越小越相关）
                print(f"    距离={score:.4f} | {truncate_text(doc.page_content, 50)}")

    print("\n关键点：")
    print_points(
        "multilingual 模型对中文查询效果更好",
        "英文查询两个模型差异不大",
        "选择 Embedding 模型要考虑目标语言和文档语言",
        "模型维度影响存储空间和检索速度",
    )


# ============================================================================
# 示例 5：混合检索 + 重排序
# ============================================================================


def example_5_hybrid_retrieval_reranking():
    """
    示例5：混合检索 + CrossEncoder 重排序

    检索策略：
    1. 向量检索：理解语义，但可能漏掉精确匹配
    2. BM25 检索：精确关键词匹配，但不理解语义
    3. 合并去重：两路结果合并
    4. CrossEncoder 重排序：精确排序，取 top-3

    这就是企业级 RAG 的标准检索管线。
    """
    print_title("示例 5：混合检索 + 重排序")

    # 确保有数据
    vs = get_chroma_store(collection_name="enterprise_rag")
    if vs._collection.count() == 0:
        print_warning("向量库为空，先运行示例 2 入库数据")
        return None

    query = "哪个模型 MMLU 分数最高？"
    print(f"  查询: {query}")

    # Step 1: 向量检索
    print_section("Step 1: 向量检索")
    from retrieval import vector_search

    vec_results = vector_search([query], vs)
    print(f"  向量检索命中: {len(vec_results)} 条")
    for doc in vec_results[:3]:
        print(f"  - {truncate_text(doc.page_content, 70)}")

    # Step 2: BM25 检索
    print_section("Step 2: BM25 关键词检索")
    # 从 ChromaDB 获取所有文档构建 BM25 索引
    all_docs = vs.similarity_search("", k=vs._collection.count())
    from retrieval import bm25_search

    bm25_results = bm25_search([query], all_docs)
    print(f"  BM25 命中: {len(bm25_results)} 条")
    for doc in bm25_results[:3]:
        print(f"  - {truncate_text(doc.page_content, 70)}")

    # Step 3: 合并去重
    print_section("Step 3: 合并去重")
    from retrieval import merge_and_deduplicate

    merged = merge_and_deduplicate(vec_results, bm25_results)
    print(f"  合并后: {len(merged)} 条（去重后）")

    # Step 4: CrossEncoder 重排序
    print_section("Step 4: CrossEncoder 重排序")
    print("  加载重排序模型（首次运行需下载约 80MB）...")
    from retrieval import rerank_documents

    reranked = rerank_documents(query, merged)
    print(f"  重排序后: {len(reranked)} 条")
    for i, doc in enumerate(reranked):
        print(f"  [{i+1}] {truncate_text(doc.page_content, 70)}")

    print("\n关键点：")
    print_points(
        "向量检索擅长语义理解（同义词、近义词）",
        "BM25 擅长精确关键词匹配",
        "两路合并后用 CrossEncoder 精确重排序",
        "CrossEncoder 比 bi-encoder 更准但更慢",
    )

    return vs, all_docs


# ============================================================================
# 示例 6：文档评分 + 自适应生成（CRAG）
# ============================================================================


def example_6_adaptive_generation():
    """
    示例6：CRAG（Corrective RAG）模式

    核心思想：不盲目信任检索结果，先评估再决定下一步。

    流程：
    1. 检索 + 重排序（同示例 5）
    2. LLM 评估文档充分性（0.0 ~ 1.0）
    3. 评分 >= 0.7 → 生成回答
    4. 评分 < 0.7 → 标记需要外部检索

    这是企业级 RAG 的关键质量保障机制。
    """
    print_title("示例 6：CRAG 文档评分 + 自适应生成")

    vs = get_chroma_store(collection_name="enterprise_rag")
    if vs._collection.count() == 0:
        print_warning("向量库为空，先运行示例 2 入库数据")
        return None

    # 测试两个查询：一个知识库能答，一个不能答
    test_queries = [
        "哪个 AI 模型的 MMLU 分数最高？",  # 应该能答
        "2024 年诺贝尔物理学奖得主是谁？",   # 知识库里没有
    ]

    from retrieval import (
        rewrite_query, vector_search, bm25_search,
        merge_and_deduplicate, rerank_documents,
        grade_documents, generate_answer,
    )

    all_docs = vs.similarity_search("", k=vs._collection.count())

    for query in test_queries:
        print_section(f"查询: {query}")

        # 完整检索链
        queries = rewrite_query(query)
        print(f"  改写后: {queries}")

        vec_results = vector_search(queries, vs)
        bm25_results = bm25_search(queries, all_docs)
        merged = merge_and_deduplicate(vec_results, bm25_results)

        if merged:
            reranked = rerank_documents(query, merged)
        else:
            reranked = []

        # 关键步骤：文档评分
        score = grade_documents(query, reranked)
        print(f"  文档充分性评分: {score:.2f}")

        if score >= 0.7:
            print("  → 评分达标，生成回答")
            answer, sources = generate_answer(query, reranked)
            print(f"  回答: {truncate_text(answer, 200)}")
            if sources:
                print(f"  来源: {len(sources)} 条引用")
        else:
            print("  → 评分不足，触发 fallback")
            print("  [Fallback] 知识库中没有足够信息，建议查阅其他资料")

    print("\n关键点：")
    print_points(
        "CRAG = Corrective RAG，先评估检索质量再决定下一步",
        "LLM 充当 judge 评估文档充分性",
        "评分阈值可调（当前 0.7）",
        "fallback 机制避免基于不足信息生成错误回答",
    )


# ============================================================================
# 示例 7：端到端企业级 RAG 完整系统
# ============================================================================


def example_7_full_pipeline():
    """
    示例7：端到端完整系统

    整合所有组件：
    - 入库图：文档解析 → 切片 → 入库
    - 检索图：查询改写 → 混合检索 → 重排序 → 评分 → 生成

    用 LangGraph 的双图架构编排整个流程。
    """
    print_title("示例 7：端到端企业级 RAG 系统")

    from retrieval import build_retrieval_graph
    from ingestion import reset_collection

    # Step 1: 入库
    print_section("Step 1: 文档入库")

    pdf_path = str(SAMPLES_DIR / "tech_report_sample.pdf")
    if not os.path.exists(pdf_path):
        print_warning("样本 PDF 不存在，跳过")
        return

    from ingestion import parse_simple_pdf, chunk_documents, embed_and_store

    reset_collection("example7_full")
    pages = parse_simple_pdf(pdf_path)
    chunks = chunk_documents(pages)
    vectorstore = embed_and_store(chunks, collection_name="example7_full")
    all_docs = chunks  # BM25 需要原始文档

    print(f"  入库完成: {len(chunks)} 个切片")

    # Step 2: 构建检索图
    print_section("Step 2: 构建检索图")
    print("  图结构:")
    print("  START → rewrite → [vector + bm25] → merge → rerank → grade")
    print("         grade → generate (评分够) / fallback (评分不够) → END")

    retrieval_graph = build_retrieval_graph(vectorstore, all_docs)

    # Step 3: 执行查询
    test_queries = [
        "RAG 在 2024 年有哪些改进？",
        "比较不同模型的上下文窗口长度",
    ]

    for query in test_queries:
        print_section(f"查询: {query}")

        result = retrieval_graph.invoke({
            "original_query": query,
            "rewritten_queries": [],
            "vector_results": [],
            "bm25_results": [],
            "merged_results": [],
            "reranked_results": [],
            "relevance_score": 0.0,
            "context": "",
            "answer": "",
            "sources": [],
            "needs_fallback": False,
        })

        print(f"  评分: {result['relevance_score']:.2f}")
        print(f"  回答: {truncate_text(result['answer'], 200)}")
        if result.get("sources"):
            print(f"  引用: {len(result['sources'])} 条")

    print("\n" + "=" * 70)
    print(" 企业级 RAG 系统架构总结")
    print("=" * 70)
    print("\n双图架构：")
    print("  入库图: parse → chunk → store")
    print("  检索图: rewrite → [vector + bm25] → merge → rerank → grade → generate")
    print("\n核心技术：")
    print_points(
        "ChromaDB 向量库（本地持久化）",
        "HuggingFace Embeddings（免费离线）",
        "BM25 + Vector 混合检索",
        "CrossEncoder 精确重排序",
        "CRAG 文档评分 + 自适应生成",
        "LangGraph StateGraph 工作流编排",
    )


# ============================================================================
# 主程序
# ============================================================================


def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - 企业级 RAG 系统")
    print("=" * 70)

    # 确保目录存在
    ensure_dirs()

    try:
        example_1_basic_ingestion()
        input("\n按 Enter 继续...")

        example_2_graph_ingestion()
        input("\n按 Enter 继续...")

        example_3_multimodal_processing()
        input("\n按 Enter 继续...")

        example_4_embedding_comparison()
        input("\n按 Enter 继续...")

        example_5_hybrid_retrieval_reranking()
        input("\n按 Enter 继续...")

        example_6_adaptive_generation()
        input("\n按 Enter 继续...")

        example_7_full_pipeline()

        print("\n" + "=" * 70)
        print(" 完成！")
        print("=" * 70)
        print("\n核心要点：")
        print_points(
            "文档入库：解析 → 切片 → Embedding → ChromaDB",
            "LangGraph 编排：StateGraph 管理流水线状态",
            "多模态支持：UnstructuredPDFLoader 识别表格/图片",
            "混合检索：BM25（精确）+ Vector（语义）双路召回",
            "CrossEncoder 重排序：精确排序提升检索质量",
            "CRAG 模式：文档评分 + 自适应生成保障回答质量",
        )
        print("\n下一步：")
        print("  - 阅读 README.md 了解完整的架构设计和 API 解释")
        print("  - 尝试用自己的 PDF 文档运行入库和检索")
        print("  - 调整 config.py 中的参数观察效果变化")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
