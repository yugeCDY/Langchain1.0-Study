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
- embed_and_store()     向量化 + PostgreSQL 入库
- build_ingestion_graph()  构建 LangGraph 入库工作流
"""

from pathlib import Path
import csv
import hashlib
import json
import re
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PG_TABLE_NAME,
    PG_TSV_CONFIG_EN,
    PG_TSV_CONFIG_ZH,
    get_embeddings,
    get_pg_connection,
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
        print("  [WARN] unstructured 未安装，使用 PyPDFLoader 替代")
        print("  [TIP] 安装方式：pip install \"unstructured[pdf]\"")
        return parse_simple_pdf(file_path)


def _detect_language(text: str) -> str:
    """粗粒度语言识别：中文优先，其次英文，其他记为 unknown。"""
    if not text:
        return "unknown"
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if zh > en:
        return "zh"
    if en > 0:
        return "en"
    return "unknown"


def _normalize_acl(allowed_roles: list[str] | None) -> list[str]:
    """ACL 归一化：去重并保持稳定顺序。"""
    roles = allowed_roles or ["public"]
    seen: set[str] = set()
    normalized: list[str] = []
    for role in roles:
        r = str(role).strip()
        if not r or r in seen:
            continue
        seen.add(r)
        normalized.append(r)
    return normalized or ["public"]


def _build_doc_id(file_path: str, domain: str) -> str:
    """为业务文档生成稳定 doc_id（不含版本号）。"""
    stem = Path(file_path).stem
    basis = f"{domain}:{stem}".encode("utf-8")
    return f"{domain}_{hashlib.md5(basis).hexdigest()[:10]}"


def _load_docx(file_path: str) -> list[Document]:
    """DOCX 解析：优先结构化 loader，失败回退到纯文本 loader。"""
    try:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader

        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
        docs = loader.load()
        for doc in docs:
            doc.metadata["element_type"] = doc.metadata.get("category", "NarrativeText")
        return docs
    except ImportError:
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(file_path).load()
        for doc in docs:
            doc.metadata["element_type"] = "NarrativeText"
        return docs


def _read_csv_rows(file_path: str) -> list[list[str]]:
    """读取 CSV 行，兼容 utf-8 与 gb18030。"""
    encodings = ["utf-8-sig", "gb18030", "utf-8"]
    last_error = None
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                return list(reader)
        except Exception as e:  # pragma: no cover - 编码兼容兜底
            last_error = e
            continue
    raise ValueError(f"CSV 读取失败: {last_error}")


def _load_csv_as_tables(file_path: str, rows_per_chunk: int = 30) -> list[Document]:
    """将 CSV 作为表格块导入（保留结构，不做字符级切分）。"""
    rows = _read_csv_rows(file_path)
    if not rows:
        return []

    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    docs: list[Document] = []

    if not body:
        content = " | ".join(header)
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "element_type": "Table",
                    "row_start": 1,
                    "row_end": 1,
                },
            )
        )
        return docs

    for start in range(0, len(body), rows_per_chunk):
        part = body[start:start + rows_per_chunk]
        lines = [" | ".join(header)] + [" | ".join(row) for row in part]
        content = "\n".join(lines)
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "element_type": "Table",
                    "row_start": start + 2,
                    "row_end": start + 1 + len(part),
                },
            )
        )
    return docs


def parse_document_by_type(
    file_path: str,
    domain: str,
    allowed_roles: list[str] | None = None,
    language_hint: str | None = None,
    version: int = 1,
    is_active: bool = True,
) -> list[Document]:
    """
    按文件类型解析单文档并标准化 metadata。

    支持：PDF / DOCX / CSV
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        docs = parse_advanced_pdf(file_path)
    elif suffix == ".docx":
        docs = _load_docx(file_path)
    elif suffix == ".csv":
        docs = _load_csv_as_tables(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")

    acl = _normalize_acl(allowed_roles)
    doc_id = _build_doc_id(file_path, domain)
    for doc in docs:
        text = doc.page_content or ""
        lang = language_hint or _detect_language(text)
        doc.metadata["source"] = file_path
        doc.metadata["file_type"] = suffix.lstrip(".")
        doc.metadata["domain"] = domain
        doc.metadata["language"] = lang
        doc.metadata["allowed_roles"] = acl
        doc.metadata["doc_id"] = doc_id
        doc.metadata["version"] = int(version)
        doc.metadata["is_active"] = bool(is_active)
        if "element_type" not in doc.metadata:
            doc.metadata["element_type"] = "NarrativeText"
    return docs


def _get_next_doc_version(collection_name: str, doc_id: str) -> int:
    """读取同 doc_id 的下一版本号（metadata 方案，无需迁移表结构）。"""
    # init_postgres_schema()  # 生产环境使用预先执行的 SQL 管理表结构
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COALESCE(MAX((metadata->>'version')::int), 0)
                FROM {PG_TABLE_NAME}
                WHERE collection_name = %s
                  AND metadata ? 'doc_id'
                  AND metadata->>'doc_id' = %s
                """,
                (collection_name, doc_id),
            )
            current = int(cur.fetchone()[0] or 0)
    return current + 1


def parse_documents_in_dir(
    directory: str,
    domain: str,
    collection_name: str,
    allowed_roles: list[str] | None = None,
) -> tuple[list[Document], dict]:
    """
    批量解析目录中的异构文档，逐文件容错并返回统计信息。
    """
    support_ext = {".pdf", ".docx", ".csv"}
    all_docs: list[Document] = []
    stats = {
        "total_files": 0,
        "parsed_files": 0,
        "failed_files": 0,
        "by_type": {"pdf": 0, "docx": 0, "csv": 0},
        "errors": [],
        "timings_sec": {},
    }

    for path in sorted(Path(directory).glob("*")):
        if not path.is_file() or path.suffix.lower() not in support_ext:
            continue

        stats["total_files"] += 1
        start = time.time()
        file_path = str(path)
        doc_id = _build_doc_id(file_path, domain)
        try:
            version = _get_next_doc_version(collection_name, doc_id)
            docs = parse_document_by_type(
                file_path=file_path,
                domain=domain,
                allowed_roles=allowed_roles,
                version=version,
                is_active=True,
            )
            all_docs.extend(docs)
            stats["parsed_files"] += 1
            stats["by_type"][path.suffix.lower().lstrip(".")] += 1
            stats["timings_sec"][path.name] = round(time.time() - start, 3)
        except Exception as e:
            stats["failed_files"] += 1
            stats["timings_sec"][path.name] = round(time.time() - start, 3)
            stats["errors"].append(f"{path.name}: {e}")

    return all_docs, stats


# ============================================================================
# 文档切片
# ============================================================================


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    将文档切片为适合企业检索的小段（两阶段切分）

    阶段 1（结构块）：
    - 优先按段落/句子切成较大的 parent chunk，尽量保持语义完整

    阶段 2（检索块）：
    - 在每个 parent chunk 内继续细分为 child chunk，控制检索粒度
    - 为 child chunk 写入 parent_chunk_id，便于后续父子块回溯/重排

    参数:
        documents: 原始文档列表
        chunk_size: 目标块大小（字符数）
        chunk_overlap: 重叠大小

    返回:
        切片后的文档列表
    """
    # 第一阶段：结构化粗切，优先保持段落和句子完整。
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(chunk_size * 3, chunk_size + 1),
        chunk_overlap=min(chunk_overlap, max(chunk_size // 10, 20)),
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
        add_start_index=True,
    )
    parent_docs = parent_splitter.split_documents(documents)

    # 第二阶段：在每个 parent chunk 内细切，得到检索友好的 child chunk。
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        add_start_index=True,
    )

    chunks: list[Document] = []
    for parent_idx, parent_doc in enumerate(parent_docs):
        source = parent_doc.metadata.get("source", "unknown")
        page = parent_doc.metadata.get("page", 0)
        parent_chunk_id = f"{Path(source).stem}_p{page}_parent_{parent_idx}"

        child_docs = child_splitter.split_documents([parent_doc])
        for child in child_docs:
            # 记录父子关系，方便检索后做上下文扩展（parent recall）。
            child.metadata["parent_chunk_id"] = parent_chunk_id
            child.metadata["chunk_level"] = "child"
            chunks.append(child)

    return chunks


def apply_quality_gate(chunks: list[Document]) -> tuple[list[Document], int]:
    """
    数据质量门控：过滤空块、超短块、低信息密度块。

    返回：(保留块, 被过滤数量)
    """
    kept: list[Document] = []
    dropped = 0

    for chunk in chunks:
        text = (chunk.page_content or "").strip()
        elem = str(chunk.metadata.get("element_type", "")).lower()
        if not text:
            dropped += 1
            continue

        # 表格块保留结构，长度门控更宽松
        if elem == "table":
            if len(text) < 8:
                dropped += 1
                continue
            kept.append(chunk)
            continue

        if len(text) < 20:
            dropped += 1
            continue

        info_chars = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", text))
        density = info_chars / max(len(text), 1)
        if density < 0.18:
            dropped += 1
            continue
        kept.append(chunk)

    return kept, dropped


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
        elem_type = str(doc.metadata.get("element_type", "text"))
        file_type = str(doc.metadata.get("file_type", ""))
        if elem_type.lower() == "table" or file_type == "csv":
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
) -> int:
    """
    将文档切片向量化并存入 PostgreSQL（pgvector + FTS）

    参数:
        chunks: 文档切片列表
        collection_name: 逻辑集合名（多租户/多业务隔离字段）

    返回:
        实际写入条数
    """
    if not chunks:
        return 0

    # P0：质量门控，避免低质量块污染检索
    chunks, dropped = apply_quality_gate(chunks)
    if dropped:
        print(f"  [质量门控] 过滤低质量切片: {dropped}")
    if not chunks:
        return 0

    # init_postgres_schema()  # 生产环境使用预先执行的 SQL 管理表结构
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents([c.page_content for c in chunks])

    # 为每个 chunk 生成唯一 ID（基于内容哈希 + 来源），避免重复入库
    rows: list[tuple[str, str, str, str, str]] = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        content_hash = hashlib.md5(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:8]
        doc_id = f"{collection_name}_{Path(source).stem}_p{page}_{content_hash}_{i}"
        rows.append(
            (
                doc_id,
                collection_name,
                chunk.page_content,
                json.dumps(chunk.metadata, ensure_ascii=False),
                str(vec),
            )
        )

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # 文档版本管理：新版本入库前将同 doc_id 历史版本标记为非激活。
            active_doc_ids = sorted(
                {
                    str(c.metadata.get("doc_id"))
                    for c in chunks
                    if c.metadata.get("doc_id") and c.metadata.get("is_active", True)
                }
            )
            for doc_id in active_doc_ids:
                cur.execute(
                    f"""
                    UPDATE {PG_TABLE_NAME}
                    SET metadata = jsonb_set(metadata, '{{is_active}}', 'false'::jsonb, true)
                    WHERE collection_name = %s
                      AND metadata ? 'doc_id'
                      AND metadata->>'doc_id' = %s
                      AND COALESCE(metadata->>'is_active', 'true') = 'true'
                    """,
                    (collection_name, doc_id),
                )

            cur.executemany(
                f"""
                INSERT INTO {PG_TABLE_NAME}
                (
                    id,
                    collection_name,
                    content,
                    metadata,
                    content_tsv_zh,
                    content_tsv_en,
                    embedding
                )
                VALUES (
                    %s, %s, %s, %s::jsonb,
                    to_tsvector(%s, %s),
                    to_tsvector(%s, %s),
                    %s::vector
                )
                ON CONFLICT (id) DO UPDATE SET
                    collection_name = EXCLUDED.collection_name,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    content_tsv_zh = EXCLUDED.content_tsv_zh,
                    content_tsv_en = EXCLUDED.content_tsv_en,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        r[0],
                        r[1],
                        r[2],
                        r[3],
                        PG_TSV_CONFIG_ZH,
                        r[2],
                        PG_TSV_CONFIG_EN,
                        r[2],
                        r[4],
                    )
                    for r in rows
                ],
            )
        conn.commit()

    return len(rows)


def rebuild_search_vectors(collection_name: str | None = None) -> int:
    """
    回填/重建双语 FTS 列。

    适用于：
    - 新增 content_tsv_zh / content_tsv_en 后回填历史数据
    - 调整中文分词配置后批量重建检索索引源数据
    """
    where_clause = ""
    params: list = []
    if collection_name:
        where_clause = "WHERE collection_name = %s"
        params.append(collection_name)

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {PG_TABLE_NAME}
                SET content_tsv_zh = to_tsvector(%s, content),
                    content_tsv_en = to_tsvector(%s, content)
                {where_clause}
                """,
                [PG_TSV_CONFIG_ZH, PG_TSV_CONFIG_EN, *params],
            )
            updated = cur.rowcount or 0
        conn.commit()
    return updated


def reset_collection(collection_name: str = "enterprise_rag"):
    """
    清空指定逻辑集合（用于重复运行示例时避免重复数据）

    参数:
        collection_name: 要清空的集合名
    """
    # init_postgres_schema()  # 生产环境使用预先执行的 SQL 管理表结构
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {PG_TABLE_NAME} WHERE collection_name = %s",
                (collection_name,),
            )
        conn.commit()
    print(f"  [OK] 已清空集合: {collection_name}")


# ============================================================================
# LangGraph 入库工作流
# ============================================================================

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
