from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import AsyncIterator

from fastapi import HTTPException, UploadFile

from ingestion import (
    chunk_documents,
    embed_and_store,
    parse_document_by_type,
    separate_by_element_type,
)
from retrieval import (
    bm25_search,
    build_retrieval_graph,
    generate_answer,
    grade_documents,
    merge_and_deduplicate,
    rerank_documents,
    rewrite_query,
    astream_generate_answer,
    vector_search,
)
from schemas import (
    ChatMessage,
    MultimodalIngestionResponse,
    RetrievalDocument,
    RetrievalRequest,
    RetrievalResponse,
    StreamChunk,
)


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv"}


def parse_allowed_roles(value: str | None) -> list[str] | None:
    if value is None:
        return None

    roles = [item.strip() for item in value.split(",")]
    normalized = [role for role in roles if role]
    return normalized or None


def get_collection_count(collection_name: str) -> int:
    from config import PG_TABLE_NAME, get_pg_connection

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {PG_TABLE_NAME} WHERE collection_name = %s",
                (collection_name,),
            )
            return int(cur.fetchone()[0])


async def run_multimodal_ingestion(
    files: list[UploadFile],
    domain: str,
    allowed_roles: list[str] | None = None,
    collection_name: str | None = None,
) -> MultimodalIngestionResponse:
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个 PDF、DOCX 或 CSV 文件")

    roles = allowed_roles or ["hr", "ops"]
    target_collection = collection_name or domain

    documents = []
    stats = {
        "total_files": 0,
        "parsed_files": 0,
        "failed_files": 0,
        "by_type": {"pdf": 0, "docx": 0, "csv": 0},
        "errors": [],
    }

    with TemporaryDirectory(prefix="enterprise-rag-") as temp_dir:
        temp_root = Path(temp_dir)

        for upload in files:
            filename = Path(upload.filename or "").name
            suffix = Path(filename).suffix.lower()

            if suffix not in SUPPORTED_EXTENSIONS:
                stats["failed_files"] += 1
                stats["errors"].append(
                    f"{filename or 'unknown'}: 不支持的文件类型，仅支持 PDF、DOCX、CSV"
                )
                continue

            stats["total_files"] += 1
            temp_path = temp_root / filename

            try:
                with temp_path.open("wb") as buffer:
                    shutil.copyfileobj(upload.file, buffer)

                parsed_docs = parse_document_by_type(
                    file_path=str(temp_path),
                    domain=domain,
                    allowed_roles=roles,
                )
                documents.extend(parsed_docs)
                stats["parsed_files"] += 1
                stats["by_type"][suffix.lstrip(".")] += 1
            except Exception as exc:
                stats["failed_files"] += 1
                stats["errors"].append(f"{filename}: {exc}")
            finally:
                await upload.close()

    if not documents:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "未解析到可入库文档",
                "errors": stats["errors"],
            },
        )

    text_docs, table_docs = separate_by_element_type(documents)
    text_chunks = chunk_documents(text_docs) if text_docs else []
    all_chunks = text_chunks + table_docs
    inserted = embed_and_store(all_chunks, collection_name=target_collection)

    return MultimodalIngestionResponse(
        domain=domain,
        collection_name=target_collection,
        allowed_roles=roles,
        total_files=stats["total_files"],
        parsed_files=stats["parsed_files"],
        failed_files=stats["failed_files"],
        format_distribution=stats["by_type"],
        text_documents=len(text_docs),
        table_documents=len(table_docs),
        text_chunks=len(text_chunks),
        total_chunks=len(all_chunks),
        inserted_chunks=inserted,
        collection_total_documents=get_collection_count(target_collection),
        errors=stats["errors"],
    )


def _to_sse_line(data: dict, event: str = "message") -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


def _stage_chunk(stage: str, detail: str) -> dict:
    return StreamChunk(
        type="stage",
        stage=stage,
        detail=detail,
    ).model_dump(exclude_none=True)


def _build_effective_query(query: str, chat_history: list[ChatMessage]) -> str:
    if not chat_history:
        return query

    recent_messages = [
        f"{message.role}: {message.content}"
        for message in chat_history[-6:]
        if message.content.strip()
    ]
    if not recent_messages:
        return query

    return "\n".join(
        [
            "以下是最近对话，请结合上下文理解当前问题：",
            *recent_messages,
            f"user: {query}",
        ]
    )


async def stream_full_pipeline(request: RetrievalRequest) -> AsyncIterator[str]:
    effective_query = _build_effective_query(request.query, request.chat_history)
    print(
        f"[stream_full_pipeline] {datetime.now().isoformat(timespec='seconds')} "
        f"query={request.query!r}",
        flush=True,
    )
    yield _to_sse_line(
        {
            "type": "open",
            "ts": datetime.now().isoformat(timespec="milliseconds"),
        },
        event="open",
    )
    yield _to_sse_line(_stage_chunk("rewrite", "正在改写查询并理解上下文"), event="stage")
    queries = rewrite_query(effective_query)
    yield _to_sse_line(
        StreamChunk(
            type="meta",
            rewritten_queries=queries,
        ).model_dump(exclude_none=True),
        event="meta",
    )

    yield _to_sse_line(_stage_chunk("retrieval", "正在执行向量检索"), event="stage")
    vector_results = vector_search(
        queries,
        collection_name=request.collection_name,
        acl_roles=request.roles,
        domain=request.domain,
        language=request.language,
        active_only=request.active_only,
    )
    yield _to_sse_line(_stage_chunk("retrieval", "正在执行 BM25 检索"), event="stage")
    bm25_results = bm25_search(
        queries,
        collection_name=request.collection_name,
        acl_roles=request.roles,
        domain=request.domain,
        language=request.language,
        active_only=request.active_only,
    )
    yield _to_sse_line(_stage_chunk("merge", "正在合并多路召回结果"), event="stage")
    merged_results = merge_and_deduplicate(vector_results, bm25_results)
    yield _to_sse_line(_stage_chunk("rerank", "正在执行重排序"), event="stage")
    reranked_results = rerank_documents(effective_query, merged_results) if merged_results else []
    yield _to_sse_line(_stage_chunk("grade", "正在评估文档是否足够回答"), event="stage")
    score = grade_documents(effective_query, reranked_results)
    needs_fallback = score < 0.7

    yield _to_sse_line(
        StreamChunk(
            type="status",
            relevance_score=score,
            needs_fallback=needs_fallback,
        ).model_dump(exclude_none=True),
        event="status",
    )

    if needs_fallback:
        yield _to_sse_line(_stage_chunk("fallback", "知识库内容不足，返回兜底提示"), event="stage")
        answer = (
            f"抱歉，知识库中没有找到足够的信息来回答「{request.query}」。"
            "建议补充文档、放宽筛选条件，或尝试换个问法。"
        )
        sources: list[dict] = []
        for char in answer:
            yield _to_sse_line(
                StreamChunk(
                    type="token",
                    content=char,
                ).model_dump(exclude_none=True),
                event="token",
            )
            await asyncio.sleep(0.015)
    else:
        yield _to_sse_line(_stage_chunk("generate", "正在基于检索内容生成答案"), event="stage")
        answer_chunks: list[str] = []
        from utils import format_sources

        sources = format_sources(reranked_results)
        async for text in astream_generate_answer(effective_query, reranked_results):
            answer_chunks.append(text)
            preview = text.replace("\n", "\\n")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            print(
                f"[stream_full_pipeline] {datetime.now().isoformat(timespec='milliseconds')} "
                f"sse_token_len={len(text)} token={preview!r}",
                flush=True,
            )
            yield _to_sse_line(
                StreamChunk(
                    type="token",
                    content=text,
                ).model_dump(exclude_none=True),
                event="token",
            )
            await asyncio.sleep(0.015)
        answer = "".join(answer_chunks)

    reranked_documents = [
        RetrievalDocument(content=doc.page_content, metadata=doc.metadata)
        for doc in reranked_results
    ]
    yield _to_sse_line(
        StreamChunk(
            type="done",
            answer=answer,
            sources=sources,
            relevance_score=score,
            needs_fallback=needs_fallback,
            reranked_documents=reranked_documents,
        ).model_dump(exclude_none=True),
        event="done",
    )


async def stream_fast_pipeline(request: RetrievalRequest) -> AsyncIterator[str]:
    effective_query = _build_effective_query(request.query, request.chat_history)
    print(
        f"[stream_fast_pipeline] {datetime.now().isoformat(timespec='seconds')} "
        f"query={request.query!r}",
        flush=True,
    )

    queries = [effective_query]
    yield _to_sse_line(
        {
            "type": "open",
            "ts": datetime.now().isoformat(timespec="milliseconds"),
        },
        event="open",
    )
    yield _to_sse_line(_stage_chunk("retrieval", "正在执行极速检索"), event="stage")
    yield _to_sse_line(
        StreamChunk(
            type="meta",
            rewritten_queries=queries,
        ).model_dump(exclude_none=True),
        event="meta",
    )

    vector_results = vector_search(
        queries,
        collection_name=request.collection_name,
        acl_roles=request.roles,
        domain=request.domain,
        language=request.language,
        active_only=request.active_only,
    )
    yield _to_sse_line(_stage_chunk("retrieval", "正在补充关键词检索"), event="stage")
    bm25_results = bm25_search(
        queries,
        collection_name=request.collection_name,
        acl_roles=request.roles,
        domain=request.domain,
        language=request.language,
        active_only=request.active_only,
    )

    yield _to_sse_line(_stage_chunk("merge", "正在合并结果并选择最相关文档"), event="stage")
    merged_results = merge_and_deduplicate(vector_results, bm25_results)
    selected_documents = merged_results[:5]

    yield _to_sse_line(
        StreamChunk(
            type="status",
            relevance_score=1.0 if selected_documents else 0.0,
            needs_fallback=not bool(selected_documents),
        ).model_dump(exclude_none=True),
        event="status",
    )

    if not selected_documents:
        yield _to_sse_line(_stage_chunk("fallback", "未检索到足够文档，返回兜底提示"), event="stage")
        answer = (
            f"抱歉，知识库中没有找到足够的信息来回答「{request.query}」。"
            "建议补充文档、放宽筛选条件，或尝试换个问法。"
        )
        sources: list[dict] = []
        for char in answer:
            yield _to_sse_line(
                StreamChunk(
                    type="token",
                    content=char,
                ).model_dump(exclude_none=True),
                event="token",
            )
            await asyncio.sleep(0.015)
    else:
        yield _to_sse_line(_stage_chunk("generate", "正在快速生成答案"), event="stage")
        answer_chunks: list[str] = []
        from utils import format_sources

        sources = format_sources(selected_documents)
        async for text in astream_generate_answer(effective_query, selected_documents):
            answer_chunks.append(text)
            yield _to_sse_line(
                StreamChunk(
                    type="token",
                    content=text,
                ).model_dump(exclude_none=True),
                event="token",
            )
            await asyncio.sleep(0.015)
        answer = "".join(answer_chunks)

    selected_reranked_documents = [
        RetrievalDocument(content=doc.page_content, metadata=doc.metadata)
        for doc in selected_documents
    ]
    yield _to_sse_line(
        StreamChunk(
            type="done",
            answer=answer,
            sources=sources,
            relevance_score=1.0 if selected_documents else 0.0,
            needs_fallback=not bool(selected_documents),
            reranked_documents=selected_reranked_documents,
        ).model_dump(exclude_none=True),
        event="done",
    )


def run_full_pipeline(request: RetrievalRequest) -> RetrievalResponse:
    effective_query = _build_effective_query(request.query, request.chat_history)
    retrieval_graph = build_retrieval_graph(
        collection_name=request.collection_name,
        acl_roles=request.roles,
        domain=request.domain,
        language=request.language,
        active_only=request.active_only,
    )

    result = retrieval_graph.invoke(
        {
            "original_query": effective_query,
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
    )

    reranked_documents = [
        RetrievalDocument(content=doc.page_content, metadata=doc.metadata)
        for doc in result.get("reranked_results", [])
    ]

    return RetrievalResponse(
        query=request.query,
        rewritten_queries=result.get("rewritten_queries", []),
        vector_result_count=len(result.get("vector_results", [])),
        bm25_result_count=len(result.get("bm25_results", [])),
        merged_result_count=len(result.get("merged_results", [])),
        reranked_result_count=len(result.get("reranked_results", [])),
        relevance_score=result.get("relevance_score", 0.0),
        needs_fallback=result.get("needs_fallback", False),
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        reranked_documents=reranked_documents,
    )
