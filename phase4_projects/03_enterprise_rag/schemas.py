from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色，例如 user / assistant")
    content: str = Field(..., description="消息内容")


class MultimodalIngestionResponse(BaseModel):
    domain: str
    collection_name: str
    allowed_roles: list[str]
    total_files: int
    parsed_files: int
    failed_files: int
    format_distribution: dict[str, int]
    text_documents: int
    table_documents: int
    text_chunks: int
    total_chunks: int
    inserted_chunks: int
    collection_total_documents: int
    errors: list[str] = Field(default_factory=list)


class RetrievalRequest(BaseModel):
    query: str = Field(..., description="用户查询")
    roles: list[str] = Field(default_factory=lambda: ["hr"], description="访问角色列表")
    collection_name: str | None = Field(default=None, description="检索集合名；为 null 时不按集合过滤")
    domain: str | None = Field(default=None, description="按业务域过滤")
    language: str | None = Field(default=None, description="按语言过滤")
    active_only: bool = Field(default=True, description="是否仅检索激活版本")
    chat_history: list[ChatMessage] = Field(default_factory=list, description="多轮对话历史")


class RetrievalDocument(BaseModel):
    content: str
    metadata: dict


class RetrievalResponse(BaseModel):
    query: str
    rewritten_queries: list[str]
    vector_result_count: int
    bm25_result_count: int
    merged_result_count: int
    reranked_result_count: int
    relevance_score: float
    needs_fallback: bool
    answer: str
    sources: list[dict]
    reranked_documents: list[RetrievalDocument]


class StreamChunk(BaseModel):
    type: str
    content: str | None = None
    answer: str | None = None
    stage: str | None = None
    detail: str | None = None
    rewritten_queries: list[str] | None = None
    relevance_score: float | None = None
    needs_fallback: bool | None = None
    sources: list[dict] | None = None
    reranked_documents: list[RetrievalDocument] | None = None
