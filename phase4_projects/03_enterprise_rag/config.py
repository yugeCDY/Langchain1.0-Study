"""
企业级 RAG 系统 - 集中配置
===========================

所有模型名称、路径常量、超参数统一在此管理。
修改配置只需改这个文件，不用动业务代码。
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================================
# 环境变量
# ============================================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")

if not API_KEY:
    raise ValueError("请在 .env 中设置 OPENAI_API_KEY")

# ============================================================================
# 路径常量
# ============================================================================

# 项目根目录（config.py 所在目录）
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CHROMA_DIR = PROJECT_DIR / "chroma_store"

# ============================================================================
# 模型配置
# ============================================================================

# LLM：通过 LiteLLM 代理调用
LLM_MODEL_NAME = "openai:glm-5.1"

# Embedding 模型
EMBEDDING_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
# ↑ 384 维，英文为主，轻量快速

EMBEDDING_MODEL_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# ↑ 384 维，支持 50+ 语言（含中文），适合中英混合文档

# Cross-Encoder 重排序模型
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# ↑ 轻量级重排序模型，本地运行，无需 API

# ============================================================================
# 切片参数
# ============================================================================

CHUNK_SIZE = 500           # 每个文本块的目标字符数
CHUNK_OVERLAP = 50         # 相邻块重叠字符数（防止信息截断）

# ============================================================================
# 检索参数
# ============================================================================

RETRIEVAL_K = 5            # 每路召回的文档数
RERANK_TOP_N = 3           # 重排序后保留的文档数
ENSEMBLE_WEIGHTS = [0.4, 0.6]  # [BM25 权重, Vector 权重]
RELEVANCE_THRESHOLD = 0.7  # 文档充分性评分阈值（低于此值触发 fallback）

# ============================================================================
# 工厂函数
# ============================================================================


def get_llm():
    """获取 LLM 实例"""
    return init_chat_model(
        LLM_MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """
    获取 Embedding 模型实例

    参数:
        model_name: 模型名称，默认使用 EMBEDDING_MODEL_DEFAULT
    """
    return HuggingFaceEmbeddings(
        model_name=model_name or EMBEDDING_MODEL_DEFAULT
    )


def get_chroma_store(
    embeddings: HuggingFaceEmbeddings | None = None,
    collection_name: str = "enterprise_rag",
):
    """
    获取 ChromaDB 向量库实例

    参数:
        embeddings: Embedding 模型，默认用 get_embeddings()
        collection_name: 集合名称
    """
    from langchain_chroma import Chroma

    if embeddings is None:
        embeddings = get_embeddings()

    # 确保持久化目录存在
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
