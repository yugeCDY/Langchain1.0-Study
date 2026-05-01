"""
企业级 RAG 系统 - 验证脚本
============================

验证各组件是否正确安装和配置。
大部分测试不需要 API key。

运行方式：python test.py
"""

import sys
import os

# Windows UTF-8 编码修复
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# 测试结果统计
passed = 0
failed = 0
skipped = 0


def test(name: str, func):
    """执行单个测试"""
    global passed, failed, skipped
    try:
        result = func()
        if result is None:
            print(f"  [SKIP] {name}")
            skipped += 1
        elif result:
            print(f"  [OK]   {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1


def main():
    print("\n" + "=" * 70)
    print(" 企业级 RAG 系统 - 组件验证")
    print("=" * 70)

    # ---- 测试 1：Python 版本 ----
    def test_python_version():
        version = sys.version_info
        if version >= (3, 10):
            print(f"    Python {version.major}.{version.minor}.{version.micro}")
            return True
        print(f"    需要 Python 3.10+，当前 {version.major}.{version.minor}")
        return False

    test("Python 版本 >= 3.10", test_python_version)

    # ---- 测试 2：环境变量 ----
    def test_env():
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        if key:
            print(f"    OPENAI_API_KEY 已设置 ({len(key)} 字符)")
            return True
        print("    OPENAI_API_KEY 未设置")
        return False

    test("环境变量配置", test_env)

    # ---- 测试 3：核心包导入 ----
    def test_langchain():
        from langchain.chat_models import init_chat_model
        from langchain_core.documents import Document
        from langchain_core.tools import tool
        print("    langchain, langchain-core 导入成功")
        return True

    test("LangChain 核心包", test_langchain)

    def test_langgraph():
        from langgraph.graph import StateGraph, START, END
        print("    langgraph 导入成功")
        return True

    test("LangGraph", test_langgraph)

    def test_langchain_chroma():
        from langchain_chroma import Chroma
        print("    langchain-chroma 导入成功")
        return True

    test("langchain-chroma", test_langchain_chroma)

    def test_langchain_classic():
        from langchain_classic.retrievers import EnsembleRetriever
        print("    langchain-classic 导入成功")
        return True

    test("langchain-classic", test_langchain_classic)

    # ---- 测试 4：Embedding 模型 ----
    def test_embeddings():
        from langchain_huggingface import HuggingFaceEmbeddings
        print("    正在加载 Embedding 模型（首次需下载约 90MB）...")
        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        result = emb.embed_query("test")
        print(f"    Embedding 维度: {len(result)}")
        return len(result) == 384

    test("HuggingFace Embeddings", test_embeddings)

    # ---- 测试 5：文本切片器 ----
    def test_splitter():
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=10
        )
        doc = Document(page_content="A" * 300)
        chunks = splitter.split_documents([doc])
        print(f"    300 字符 → {len(chunks)} 块")
        return len(chunks) >= 2

    test("RecursiveCharacterTextSplitter", test_splitter)

    # ---- 测试 6：PyPDFLoader ----
    def test_pypdf():
        from langchain_community.document_loaders import PyPDFLoader
        print("    PyPDFLoader 导入成功")
        return True

    test("PyPDFLoader", test_pypdf)

    # ---- 测试 7：BM25Retriever ----
    def test_bm25():
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        docs = [
            Document(page_content="Python 是一种编程语言"),
            Document(page_content="RAG 是检索增强生成"),
        ]
        retriever = BM25Retriever.from_documents(docs, k=2)
        results = retriever.invoke("Python")
        print(f"    BM25 检索 'Python' → {len(results)} 条")
        return len(results) >= 1

    test("BM25Retriever", test_bm25)

    # ---- 测试 8：EnsembleRetriever ----
    def test_ensemble():
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        docs = [
            Document(page_content="测试文档一", metadata={"source": "test"}),
            Document(page_content="测试文档二", metadata={"source": "test"}),
        ]

        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vs = Chroma.from_documents(docs, emb, collection_name="test_ensemble")
        bm25 = BM25Retriever.from_documents(docs, k=2)

        ensemble = EnsembleRetriever(
            retrievers=[bm25, vs.as_retriever(k=2)],
            weights=[0.4, 0.6],
        )
        results = ensemble.invoke("测试")
        print(f"    Ensemble 检索 '测试' → {len(results)} 条")
        return len(results) >= 1

    test("EnsembleRetriever 混合检索", test_ensemble)

    # ---- 测试 9：CrossEncoder ----
    def test_cross_encoder():
        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            print("    HuggingFaceCrossEncoder 导入成功")
            return True
        except ImportError as e:
            print(f"    导入失败: {e}")
            return False

    test("HuggingFaceCrossEncoder", test_cross_encoder)

    # ---- 测试 10：ChromaDB 持久化 ----
    def test_chroma_persist():
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        import tempfile

        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            vs = Chroma(
                collection_name="test_persist",
                embedding_function=emb,
                persist_directory=tmpdir,
            )
            vs.add_documents([
                Document(page_content="持久化测试", metadata={"test": True})
            ])
            count = vs._collection.count()
            print(f"    写入后文档数: {count}")
            return count == 1

    test("ChromaDB 持久化", test_chroma_persist)

    # ---- 测试 11：State 定义 ----
    def test_state():
        from graph_state import IngestionState, RetrievalState

        # 验证 TypedDict 能正常实例化
        ing_state: IngestionState = {
            "file_path": "test.pdf",
            "raw_documents": [],
            "text_chunks": [],
            "table_chunks": [],
            "chunk_count": 0,
            "status": "test",
            "errors": [],
        }
        ret_state: RetrievalState = {
            "original_query": "test",
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
        }
        print("    IngestionState, RetrievalState 定义正确")
        return True

    test("TypedDict 状态定义", test_state)

    # ---- 测试 12：入库图编译 ----
    def test_ingestion_graph():
        from ingestion import build_ingestion_graph

        graph = build_ingestion_graph()
        print("    入库图编译成功")
        return True

    test("入库图编译", test_ingestion_graph)

    # ---- 测试 13：UnstructuredPDFLoader（可选）----
    def test_unstructured():
        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
            print("    UnstructuredPDFLoader 可用")
            return True
        except ImportError:
            print("    UnstructuredPDFLoader 未安装（可选依赖）")
            return None  # Skip

    test("UnstructuredPDFLoader（可选）", test_unstructured)

    # ---- 测试 14：fpdf2 样本生成 ----
    def test_fpdf2():
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 10, "Test PDF")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                pdf.output(f.name)
                size = os.path.getsize(f.name)
                os.unlink(f.name)
            print(f"    PDF 生成成功 ({size} bytes)")
            return True
        except ImportError:
            print("    fpdf2 未安装（样本生成需要）")
            return None

    test("fpdf2 PDF 生成", test_fpdf2)

    # ---- 结果汇总 ----
    total = passed + failed + skipped
    print("\n" + "=" * 70)
    print(f" 验证结果: {passed} 通过 / {failed} 失败 / {skipped} 跳过 / {total} 总计")
    print("=" * 70)

    if failed > 0:
        print("\n失败项需要修复后才能正常运行 main.py")
        print("建议检查：")
        print("  1. pip install -r requirements.txt")
        print("  2. 确认 .env 文件中设置了 OPENAI_API_KEY")
    elif skipped > 0:
        print("\n核心组件全部通过，部分可选组件被跳过")
        print("如需完整功能：")
        print('  pip install "unstructured[pdf]" fpdf2')
    else:
        print("\n所有组件验证通过！可以运行 python main.py 开始学习")

    return failed == 0


if __name__ == "__main__":
    main()
