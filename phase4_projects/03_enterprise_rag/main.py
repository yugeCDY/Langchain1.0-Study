"""
FastAPI 版企业级 RAG 项目入口。

启动方式：
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import API_KEY, LANGSMITH_PROJECT, LANGSMITH_TRACING
from schemas import (
    MultimodalIngestionResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from services import (
    parse_allowed_roles,
    stream_fast_pipeline,
    run_full_pipeline,
    run_multimodal_ingestion,
    stream_full_pipeline,
)


load_dotenv()

if not API_KEY:
    raise ValueError("请先在 .env 中设置 OPENAI_API_KEY")


app = FastAPI(
    title="Enterprise RAG API",
    description="将多模态入库与完整检索流水线封装为 FastAPI 接口。",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "message": "Enterprise RAG API is running",
        "langsmith_tracing": LANGSMITH_TRACING,
        "langsmith_project": LANGSMITH_PROJECT,
        "endpoints": {
            "health": "/health",
            "multimodal_ingestion": "/api/v1/ingestion/multimodal",
            "full_pipeline": "/api/v1/retrieval/full-pipeline",
            "full_pipeline_stream": "/api/v1/retrieval/full-pipeline/stream",
            "fast_pipeline_stream": "/api/v1/retrieval/full-pipeline/fast-stream",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/api/v1/ingestion/multimodal",
    response_model=MultimodalIngestionResponse,
    summary="多模态文档入库",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["domain", "files"],
                        "properties": {
                            "domain": {
                                "type": "string",
                                "title": "Domain",
                            },
                            "allowed_roles": {
                                "type": "string",
                                "title": "Allowed Roles",
                                "description": "ACL 角色，使用逗号分隔，例如：hr,ops",
                            },
                            "collection_name": {
                                "type": "string",
                                "title": "Collection Name",
                            },
                            "files": {
                                "type": "array",
                                "title": "Files",
                                "items": {
                                    "type": "string",
                                    "format": "binary",
                                },
                            },
                        },
                    }
                }
            },
        }
    },
)
async def multimodal_ingestion(
    domain: str = Form(...),
    files: list[UploadFile] = File(
        ...,
        description="上传一个或多个 PDF、DOCX、CSV 文件",
    ),
    allowed_roles: str | None = Form(
        default=None,
        description="ACL 角色，使用逗号分隔，例如：hr,ops",
    ),
    collection_name: str | None = Form(default=None),
) -> MultimodalIngestionResponse:
    return await run_multimodal_ingestion(
        files=files,
        domain=domain,
        allowed_roles=parse_allowed_roles(allowed_roles),
        collection_name=collection_name,
    )


@app.post(
    "/api/v1/retrieval/full-pipeline",
    response_model=RetrievalResponse,
    summary="完整检索生成流水线",
)
def full_pipeline(request: RetrievalRequest) -> RetrievalResponse:
    return run_full_pipeline(request)


@app.post(
    "/api/v1/retrieval/full-pipeline/stream",
    summary="完整检索生成流水线（流式）",
)
def full_pipeline_stream(request: RetrievalRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_full_pipeline(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/api/v1/retrieval/full-pipeline/fast-stream",
    summary="完整检索生成流水线（极速流式）",
)
def full_pipeline_fast_stream(request: RetrievalRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_fast_pipeline(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/api/v1/retrieval/full-pipeline",
    response_model=RetrievalResponse,
    summary="完整检索生成流水线（GET 便捷版）",
)
def full_pipeline_get(
    query: str,
    roles: Annotated[list[str], Query()] = ["hr"],
    collection_name: str | None = None,
    domain: str | None = None,
    language: str | None = None,
    active_only: bool = True,
) -> RetrievalResponse:
    request = RetrievalRequest(
        query=query,
        roles=roles,
        collection_name=collection_name,
        domain=domain,
        language=language,
        active_only=active_only,
    )
    return run_full_pipeline(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8087, reload=True)
