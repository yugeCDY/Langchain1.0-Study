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
import psycopg

# ============================================================================
# 环境变量
# ============================================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "03_enterprise_rag")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_WORKSPACE_ID = os.getenv("LANGSMITH_WORKSPACE_ID")

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
LLM_MODEL_NAME = "deepseek:deepseek-v4-flash"

# Embedding 模型
EMBEDDING_MODEL_DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# ↑ 384 维，支持 50+ 语言（含中文），适合中英混合文档

EMBEDDING_MODEL_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# ↑ 显式保留多语言模型常量，便于示例代码对比或手动切换

# Cross-Encoder 重排序模型
RERANKER_MODEL = "BAAI/bge-reranker-base"
# ↑ 较轻量的中英双语重排序模型，适合本地资源受限场景

# ============================================================================
# 切片参数
# ============================================================================

CHUNK_SIZE = 500           # 每个文本块的目标字符数
CHUNK_OVERLAP = 50         # 相邻块重叠字符数（防止信息截断）

# ============================================================================
# 检索参数
# ============================================================================

RETRIEVAL_K = 5            # 每路召回的文档数
RERANK_TOP_N = 5           # 重排序后保留的文档数
ENSEMBLE_WEIGHTS = [0.4, 0.6]  # [BM25 权重, Vector 权重]
RRF_K = 60                 # RRF 融合平滑项（越大越平滑）
RELEVANCE_THRESHOLD = 0.7  # 文档充分性评分阈值（低于此值触发 fallback）

# ============================================================================
# PostgreSQL 配置（生产推荐）
# ============================================================================

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://postgres:153250@111.229.94.194:5432/vector_db",
)
PG_TABLE_NAME = os.getenv("PG_TABLE_NAME", "rag_documents")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
PG_TSV_CONFIG_ZH = os.getenv("PG_TSV_CONFIG_ZH", "chinese")
PG_TSV_CONFIG_EN = os.getenv("PG_TSV_CONFIG_EN", "simple")

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


def get_pg_connection():
    """获取 PostgreSQL 连接。"""
    return psycopg.connect(POSTGRES_DSN)


def init_postgres_schema():
    """
    初始化 PostgreSQL 表结构与索引。

    依赖扩展：
    - pgvector（向量检索）
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PG_TABLE_NAME} (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    content_tsv_zh tsvector,
                    content_tsv_en tsvector,
                    embedding vector({EMBEDDING_DIM}),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # 兼容已有表：按需补齐双语 FTS 列。
            cur.execute(
                f"ALTER TABLE {PG_TABLE_NAME} ADD COLUMN IF NOT EXISTS content_tsv_zh tsvector"
            )
            cur.execute(
                f"ALTER TABLE {PG_TABLE_NAME} ADD COLUMN IF NOT EXISTS content_tsv_en tsvector"
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{PG_TABLE_NAME}_collection
                ON {PG_TABLE_NAME} (collection_name)
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{PG_TABLE_NAME}_content_tsv_zh
                ON {PG_TABLE_NAME}
                USING GIN (content_tsv_zh)
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{PG_TABLE_NAME}_content_tsv_en
                ON {PG_TABLE_NAME}
                USING GIN (content_tsv_en)
                """
            )
            # ivfflat 需要在数据量较大时配合 ANALYZE 才能发挥效果。
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{PG_TABLE_NAME}_embedding_ivfflat
                ON {PG_TABLE_NAME}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        conn.commit()
