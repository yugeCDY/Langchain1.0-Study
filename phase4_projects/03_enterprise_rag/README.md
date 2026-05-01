# 企业级 RAG 系统 (Enterprise RAG)

## 为什么学这个

前面的模块 13 讲了基础 RAG（单路向量检索），模块 14 讲了混合检索（BM25 + Vector）。但一个真正的企业级 RAG 系统远不止"检索 + 生成"这么简单：

- **检索质量不够** → 需要查询改写、多路召回、重排序
- **检索到无关内容** → 需要文档评分、自适应生成
- **文档有表格/图片** → 需要多模态解析
- **流程复杂难维护** → 需要用 LangGraph 编排

本模块将前面学过的所有知识整合为一个完整的企业级 RAG 系统。

## 和已有概念的类比

| 已知概念 | 企业级 RAG 中的对应 |
|----------|---------------------|
| 基础 RAG（模块 13） | 增加了查询改写和重排序 |
| 混合检索（模块 14） | 整合进 LangGraph 工作流 |
| StateGraph（模块 15） | 编排入库和检索两条流水线 |
| 条件路由（模块 15） | 按文档类型分流、按评分决定下一步 |
| 多 Agent（模块 16） | 入库 Agent + 检索 Agent 各司其职 |

可以把企业级 RAG 想象成**搜索引擎 + AI 问答**的组合：
- 搜索引擎负责"找"（多路召回、重排序、评分）
- AI 问答负责"答"（基于检索结果生成回答）
- LangGraph 是整个流程的"调度中心"

## 快速开始

### 安装依赖

```bash
pip install langchain langchain-chroma langchain-classic langgraph
pip install langchain-huggingface langchain-community langchain-text-splitters
pip install sentence-transformers rank_bm25 pypdf chromadb fpdf2 torch

# 可选：高级 PDF 解析（表格/图片识别）
pip install "unstructured[pdf]"
```

### 运行

```bash
python test.py    # 验证组件安装
python main.py    # 运行完整示例
```

---

## 项目架构

### 目录结构

```
03_enterprise_rag/
├── config.py         # 集中配置：模型名、路径、超参数、工厂函数
├── graph_state.py    # 状态定义：入库图和检索图的 TypedDict
├── ingestion.py      # 文档入库：解析、切片、Embedding、ChromaDB 存储
├── retrieval.py      # 检索引擎：改写、多路检索、重排序、评分、生成
├── graph_nodes.py    # LangGraph 节点：将业务函数封装为图节点
├── utils.py          # 工具函数：打印、哈希、样本 PDF 生成
├── main.py           # 入口文件：7 个渐进式示例
├── test.py           # 验证脚本：14 项组件检查
├── data/samples/     # 示例 PDF（程序自动生成）
└── chroma_store/     # ChromaDB 持久化目录（自动创建）
```

### 双图设计

```
[入库图]                              [检索图]

START                                 START
  |                                     |
  v                                     v
parse_pdf                           rewrite_query
  |                                     |
  +----------+              +-----------+-----------+
  |          |              |                       |
  v          v              v                       v
chunk    extract_table   vector_search          bm25_search
  |          |              |                       |
  +----+-----+              +-----------+-----------+
       |                                |
       v                                v
  embed_and_store                   merge_results
       |                                |
       v                                v
      END                           rerank_docs
                                        |
                                        v
                                   grade_docs
                                        |
                                  +-----+------+
                                  |             |
                                  v             v
                             generate      fallback
                                  |             |
                                  +------+------+
                                         |
                                         v
                                        END
```

两条图通过共享的 ChromaDB 向量库连接。入库图负责"把文档存进去"，检索图负责"从文档中找答案"。

---

## 文件详解

### 1. config.py — 集中配置

**作用**：所有模型名称、文件路径、超参数、工厂函数统一管理。其他文件通过 `from config import ...` 获取配置。修改任何参数只需改这一个文件。

#### 常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `API_KEY` | 从 `.env` 读取 | OpenAI 兼容 API Key |
| `BASE_URL` | 从 `.env` 读取 | API 代理地址 |
| `PROJECT_DIR` | 项目根目录 | `config.py` 所在目录 |
| `DATA_DIR` | `PROJECT_DIR/data` | 数据目录 |
| `SAMPLES_DIR` | `DATA_DIR/samples` | 示例 PDF 目录 |
| `CHROMA_DIR` | `PROJECT_DIR/chroma_store` | ChromaDB 持久化目录 |
| `LLM_MODEL_NAME` | `"openai:glm-5.1"` | LLM 模型标识 |
| `EMBEDDING_MODEL_DEFAULT` | `"sentence-transformers/all-MiniLM-L6-v2"` | 默认 Embedding（384维，英文为主） |
| `EMBEDDING_MODEL_MULTILINGUAL` | `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"` | 多语言 Embedding（384维，50+语言） |
| `RERANKER_MODEL` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | CrossEncoder 重排序模型 |
| `CHUNK_SIZE` | `500` | 文本切片目标字符数 |
| `CHUNK_OVERLAP` | `50` | 相邻切片重叠字符数 |
| `RETRIEVAL_K` | `5` | 每路检索召回文档数 |
| `RERANK_TOP_N` | `3` | 重排序后保留的文档数 |
| `ENSEMBLE_WEIGHTS` | `[0.4, 0.6]` | BM25 和 Vector 的权重 |
| `RELEVANCE_THRESHOLD` | `0.7` | 文档充分性评分阈值 |

#### 工厂函数

```python
get_llm()
```
- **返回**：`init_chat_model("openai:glm-5.1", ...)` 实例
- **用途**：所有需要调用 LLM 的地方统一用这个函数

```python
get_embeddings(model_name=None)
```
- **参数**：`model_name` — 模型名，默认 `EMBEDDING_MODEL_DEFAULT`
- **返回**：`HuggingFaceEmbeddings` 实例
- **用途**：向量化和 ChromaDB 检索

```python
get_chroma_store(embeddings=None, collection_name="enterprise_rag")
```
- **参数**：`embeddings` — Embedding 模型（默认自动获取）；`collection_name` — ChromaDB 集合名
- **返回**：`Chroma` 实例（自动持久化到 `CHROMA_DIR`）
- **用途**：向量库的统一入口

---

### 2. graph_state.py — 状态定义

**作用**：用 `TypedDict` 定义 LangGraph 图的共享状态。每个节点接收完整状态，返回需要更新的字段。使用 `Annotated[list, operator.add]` 实现 reducer——并行节点的结果自动追加合并而不是覆盖。

#### IngestionState — 入库图状态

```python
class IngestionState(TypedDict):
    file_path: str                                          # 输入：文件路径
    raw_documents: list[Document]                           # 解析后的文档页
    text_chunks: Annotated[list[Document], operator.add]    # 文本切片（reducer）
    table_chunks: Annotated[list[Document], operator.add]   # 表格切片（reducer）
    chunk_count: int                                        # 切片总数
    status: str                                             # 流水线状态文本
    errors: Annotated[list[str], operator.add]              # 错误日志（reducer）
```

**reducer 的作用**：当多模态入库图并行处理文本和表格时，两个节点分别返回 `text_chunks` 和 `table_chunks`。`operator.add` 告诉 LangGraph 把这些列表拼接起来，而不是用后一个覆盖前一个。

#### RetrievalState — 检索图状态

```python
class RetrievalState(TypedDict):
    original_query: str                                      # 用户原始提问
    rewritten_queries: list[str]                             # LLM 改写后的查询变体
    vector_results: Annotated[list[Document], operator.add]  # 向量检索结果（reducer）
    bm25_results: Annotated[list[Document], operator.add]    # BM25 检索结果（reducer）
    merged_results: list[Document]                           # 去重合并后
    reranked_results: list[Document]                         # 重排序后
    relevance_score: float                                   # 充分性评分（0~1）
    context: str                                             # 拼接的上下文
    answer: str                                              # LLM 回答
    sources: list[dict]                                      # 来源引用
    needs_fallback: bool                                     # 是否需要兜底
```

**调用图时需传入完整初始状态**：

```python
result = graph.invoke({
    "file_path": "doc.pdf",
    "raw_documents": [],
    "text_chunks": [],
    "table_chunks": [],
    "chunk_count": 0,
    "status": "",
    "errors": [],
})
```

---

### 3. ingestion.py — 文档入库模块

**作用**：负责文档解析 → 切片 → Embedding → ChromaDB 入库的完整流水线。支持两种解析策略：简单的 PyPDFLoader 和高级的 UnstructuredPDFLoader（识别表格/图片）。提供两个 LangGraph 图：线性入库图和多模态入库图。

#### 函数清单

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `parse_simple_pdf(file_path)` | `file_path: str` — PDF 路径 | `list[Document]` | PyPDFLoader 解析，每页一个 Document |
| `parse_advanced_pdf(file_path)` | `file_path: str` — PDF 路径 | `list[Document]` | UnstructuredPDFLoader 解析，按元素类型拆分。未安装时 fallback 到 `parse_simple_pdf` |
| `chunk_documents(documents, chunk_size, chunk_overlap)` | `documents: list[Document]`；`chunk_size: int`（默认 500）；`chunk_overlap: int`（默认 50） | `list[Document]` | RecursiveCharacterTextSplitter 切片，支持中英文分隔符 |
| `separate_by_element_type(documents)` | `documents: list[Document]` | `tuple[list[Document], list[Document]]` | 分离文本元素和表格元素，返回 `(text_docs, table_docs)` |
| `embed_and_store(chunks, collection_name)` | `chunks: list[Document]`；`collection_name: str`（默认 `"enterprise_rag"`） | `Chroma` | 为每个 chunk 生成唯一 ID，向量化后存入 ChromaDB |
| `reset_collection(collection_name)` | `collection_name: str`（默认 `"enterprise_rag"`） | `None` | 清空指定 ChromaDB 集合，避免重复数据 |
| `build_ingestion_graph()` | 无 | `CompiledStateGraph` | 线性入库图：`START → parse → chunk → store → END` |
| `build_multimodal_ingestion_graph()` | 无 | `CompiledStateGraph` | 多模态入库图：parse 后按元素类型分流，文本切片、表格保留原文，两路 reducer 合并后统一入库 |

#### embed_and_store 的 ID 生成策略

每个 chunk 生成唯一 ID：`{文件名}_p{页码}_{内容哈希8位}_{序号}`。这确保同一文档重复入库时不会产生重复向量（ChromaDB 的 `add_documents` 遇到相同 ID 会覆盖而不是追加）。

#### 两个入库图的区别

| | `build_ingestion_graph()` | `build_multimodal_ingestion_graph()` |
|---|---|---|
| 流程 | 线性：parse → chunk → store | 分支：parse → [chunk_text + extract_tables] → store |
| 适用 | 纯文本 PDF | 含表格的 PDF |
| 表格处理 | 和文本一起切片 | 表格保留原文不切片 |
| 路由 | 无条件路由 | `add_conditional_edges` 按元素类型分流 |

---

### 4. retrieval.py — 检索引擎模块

**作用**：实现完整的检索链——查询改写、向量检索、BM25 检索、合并去重、CrossEncoder 重排序、LLM 文档评分、LLM 答案生成。同时提供 `build_retrieval_graph()` 将这些步骤编排为 LangGraph 工作流。

#### 函数清单

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `rewrite_query(query, llm)` | `query: str`；`llm`（可选，默认 `get_llm()`） | `list[str]` | LLM 改写查询为 2 个变体 + 原始查询，共 3 条 |
| `vector_search(queries, vectorstore, k)` | `queries: list[str]`；`vectorstore: Chroma`；`k: int`（默认 5） | `list[Document]` | 对每个查询变体执行 `similarity_search`，自动去重 |
| `bm25_search(queries, documents, k)` | `queries: list[str]`；`documents: list[Document]`（用于构建索引）；`k: int`（默认 5） | `list[Document]` | BM25Retriever 关键词检索，每次调用重建索引，自动去重 |
| `merge_and_deduplicate(vector_results, bm25_results)` | 两路检索结果 | `list[Document]` | 基于内容哈希去重，向量结果优先排前 |
| `rerank_documents(query, documents, top_n)` | `query: str`；`documents: list[Document]`；`top_n: int`（默认 3） | `list[Document]` | CrossEncoder 重排序。首次运行下载约 80MB 模型 |
| `grade_documents(query, documents, llm)` | `query: str`；`documents: list[Document]`；`llm`（可选） | `float`（0.0~1.0） | LLM 评估文档充分性。>= 0.7 为充分，< 0.7 触发 fallback |
| `generate_answer(query, documents, llm)` | `query: str`；`documents: list[Document]`；`llm`（可选） | `tuple[str, list[dict]]` | LLM 基于检索结果生成回答，返回 `(回答文本, 来源引用)` |
| `build_retrieval_graph(vectorstore, all_documents)` | `vectorstore: Chroma`；`all_documents: list[Document]` | `CompiledStateGraph` | 构建完整检索图 |

#### build_retrieval_graph 图结构

```
START → rewrite → [vector_search + bm25_search]（并行 fan-out）
                    ↓                    ↓
                  merge（fan-in，reducer 合并）
                    ↓
                 rerank（CrossEncoder 重排序）
                    ↓
                  grade（LLM 评分）
                    ↓
          route_by_score（条件路由）
           ↓                  ↓
       generate           fallback
           ↓                  ↓
          END                END
```

**关键设计**：
- `vectorstore` 和 `all_documents` 通过闭包注入到节点函数中，不存入 State（State 只存可序列化数据）
- 并行部分通过两条 `add_edge` 实现，LangGraph 自动检测并并发执行
- fan-in 时 `vector_results` 和 `bm25_results` 都有 reducer（`operator.add`），自动拼接

#### 去重策略

所有检索函数内部都有去重：基于 `page_content` 的哈希值（`hash()` 或 `content_hash()`）。同一个文档可能被多个查询变体或不同检索路径命中，去重避免重复处理。

---

### 5. graph_nodes.py — LangGraph 节点封装

**作用**：将 `ingestion.py` 和 `retrieval.py` 中的纯函数包装为 LangGraph 节点函数（接收 State 字典，返回更新字典）。还包含路由函数和工厂函数。

#### 节点函数约定

```python
def node_function(state: MyState) -> dict:
    # 从 state 中读取输入
    value = state["field"]
    # 调用业务逻辑
    result = do_something(value)
    # 只返回需要更新的字段
    return {"field_to_update": result}
```

#### 入库图节点

| 节点 | 读取字段 | 返回字段 | 说明 |
|------|----------|----------|------|
| `parse_node` | `file_path` | `raw_documents`, `status`, `errors` | PyPDFLoader 解析 |
| `parse_advanced_node` | `file_path` | `raw_documents`, `status`, `errors` | UnstructuredPDFLoader 解析 |
| `chunk_node` | `raw_documents` | `text_chunks`, `chunk_count`, `status` | 全量切片 |
| `chunk_text_node` | `raw_documents` | `text_chunks` | 只切文本元素 |
| `extract_tables_node` | `raw_documents` | `table_chunks` | 只提取表格元素（不切片） |
| `store_node` | `text_chunks` + `table_chunks` | `status`, `errors` | Embedding + ChromaDB 入库 |

#### 检索图节点

| 节点 | 读取字段 | 返回字段 | 说明 |
|------|----------|----------|------|
| `rewrite_node` | `original_query` | `rewritten_queries` | 查询改写 |
| `merge_node` | `vector_results` + `bm25_results` | `merged_results` | 去重合并 |
| `rerank_node` | `original_query` + `merged_results` | `reranked_results` | CrossEncoder 重排序 |
| `grade_node` | `original_query` + `reranked_results` | `relevance_score`, `needs_fallback` | LLM 充分性评分 |
| `generate_node` | `original_query` + `reranked_results` | `answer`, `sources` | 生成回答 |
| `fallback_node` | `original_query` | `answer`, `sources` | 兜底回复（stub） |

#### 工厂函数

`vectorstore` 和 `all_documents` 不适合存入 State（太大且不可序列化），用工厂函数通过闭包注入：

```python
def make_vector_search_node(vectorstore):
    """创建向量检索节点——闭包捕获 vectorstore"""
    def _node(state: RetrievalState) -> dict:
        queries = state.get("rewritten_queries", [state["original_query"]])
        results = vector_search(queries, vectorstore)
        return {"vector_results": results}
    return _node

# 使用
node = make_vector_search_node(my_chroma)
builder.add_node("vector_search", node)
```

#### 路由函数

```python
route_by_score(state) -> "generate" 或 "fallback"
# 评分 >= 0.7 走 generate，< 0.7 走 fallback

route_by_content_type(state) -> ["chunk"] 或 ["chunk_text", "extract_tables"]
# 有表格元素走双分支并行，纯文本走单分支
```

---

### 6. utils.py — 工具函数

**作用**：提供打印辅助、Windows 编码修复、内容哈希、来源格式化、样本 PDF 生成等共享功能。

#### 函数清单

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `print_title(title)` | `title: str` | 无 | 打印 `====` 包裹的主标题 |
| `print_section(title)` | `title: str` | 无 | 打印 `----` 包裹的小节标题 |
| `print_points(*points)` | `*points: str` | 无 | 打印 `✅` 开头的要点列表 |
| `print_warning(msg)` | `msg: str` | 无 | 打印 `⚠️` 开头的警告 |
| `print_tip(msg)` | `msg: str` | 无 | 打印 `💡` 开头的提示 |
| `ensure_dirs()` | 无 | 无 | 确保 `data/samples/` 目录存在 |
| `content_hash(text)` | `text: str` | `str`（12 字符） | MD5 哈希前 12 位，用于文档去重 |
| `truncate_text(text, max_len)` | `text: str`；`max_len: int`（默认 100） | `str` | 截断文本，超出部分加 `...` |
| `format_sources(documents)` | `documents: list[Document]` | `list[dict]` | 提取来源信息（source、page、preview） |
| `create_sample_documents()` | 无 | 无 | 用 fpdf2 生成两个示例 PDF 到 `data/samples/` |

#### 样本 PDF

`create_sample_documents()` 生成两个文件：
- **tech_report_sample.pdf**：AI 技术报告，含模型对比表格、RAG 说明
- **mixed_content_sample.pdf**：LangChain 1.0 迁移指南，含 API 对比表格

#### Windows 编码修复

```python
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
```
导入 `utils` 时自动执行，解决 Windows 控制台中文乱码。

---

### 7. main.py — 入口文件（7 个示例）

**作用**：7 个渐进式示例，从最基础的入库到端到端完整系统。每个示例用 `input("按 Enter 继续...")` 暂停。

| # | 函数 | 核心流程 | 演示的组件 |
|---|------|----------|-----------|
| 1 | `example_1_basic_ingestion()` | PDF → 切片 → Embedding → ChromaDB → 验证检索 | `parse_simple_pdf`, `chunk_documents`, `embed_and_store`, `similarity_search` |
| 2 | `example_2_graph_ingestion()` | 用 StateGraph 编排示例 1 的流程 | `build_ingestion_graph`, `graph.invoke()` |
| 3 | `example_3_multimodal_processing()` | UnstructuredPDFLoader 解析 + 按元素类型分流 | `parse_advanced_pdf`, `separate_by_element_type`, `build_multimodal_ingestion_graph` |
| 4 | `example_4_embedding_comparison()` | 对比两个 Embedding 模型的检索质量 | `get_embeddings`, `Chroma.from_documents`, `similarity_search_with_score` |
| 5 | `example_5_hybrid_retrieval_reranking()` | 完整检索链：向量 + BM25 + 重排序 | `vector_search`, `bm25_search`, `merge_and_deduplicate`, `rerank_documents` |
| 6 | `example_6_adaptive_generation()` | CRAG：评分 → 生成/fallback | `rewrite_query`, `grade_documents`, `generate_answer` |
| 7 | `example_7_full_pipeline()` | 双图端到端系统 | `build_retrieval_graph`, 两个查询完整演示 |

**依赖关系**：示例 1 和 2 都会入库数据。示例 5 依赖示例 2 的数据（`enterprise_rag` 集合）。示例 7 独立入库（`example7_full` 集合）。

---

### 8. test.py — 验证脚本

**作用**：14 项组件验证，大部分不需要 API key。运行 `python test.py` 检查环境是否就绪。

| # | 测试项 | 需要 API key | 说明 |
|---|--------|-------------|------|
| 1 | Python 版本 >= 3.10 | 否 | — |
| 2 | 环境变量配置 | 否 | 检查 OPENAI_API_KEY |
| 3 | LangChain 核心包 | 否 | import 检查 |
| 4 | LangGraph | 否 | import 检查 |
| 5 | langchain-chroma | 否 | import 检查 |
| 6 | langchain-classic | 否 | import 检查 |
| 7 | HuggingFace Embeddings | 否 | 加载模型 + 验证 384 维 |
| 8 | RecursiveCharacterTextSplitter | 否 | 300 字符切 4 块 |
| 9 | PyPDFLoader | 否 | import 检查 |
| 10 | BM25Retriever | 否 | 2 条文档检索验证 |
| 11 | EnsembleRetriever | 否 | 混合检索验证 |
| 12 | HuggingFaceCrossEncoder | 否 | import 检查 |
| 13 | TypedDict 状态定义 | 否 | 实例化验证 |
| 14 | 入库图编译 | 否 | `build_ingestion_graph()` 编译通过 |

---

## 核心概念详解

### 多路召回（Multi-Path Recall）

**一句话概括**：用多种不同的检索方式分别找一遍，把结果合在一起，比只用一种方式找得更全。

#### 为什么需要多路

假设用户问："哪个模型 MMLU 分数最高？"

| 检索方式 | 能找到 | 可能漏掉 |
|----------|--------|----------|
| 向量检索（语义） | "性能最好的 AI"、"模型评估结果" | "MMLU" 这个精确关键词不一定命中 |
| BM25（关键词） | 精确包含 "MMLU" 的段落 | "模型评分" 这类同义表达会漏掉 |
| **两者都跑一遍** | **都找到** | — |

#### 本项目的多路召回流程

```
用户查询 → [查询改写：生成 2-3 个变体]
                ↓
        ┌───────┴───────┐
        ↓               ↓
   vector_search     bm25_search    ← 两路并行（fan-out）
   （语义理解）       （精确匹配）
        ↓               ↓
        └───────┬───────┘
                ↓
         merge + 去重          ← 合并结果（fan-in）
                ↓
          rerank（重排序）      ← 精排取 top-N
```

涉及 4 个步骤：

| 步骤 | 函数 | 做什么 |
|------|------|--------|
| 查询改写 | `rewrite_query()` | LLM 把 "MMLU 分数最高" 改写成 "哪个 AI 模型评测成绩最好" 等变体 |
| 向量检索 | `vector_search()` | 对每个变体做 `similarity_search`，找到语义相关的 chunk |
| BM25 检索 | `bm25_search()` | 对每个变体做关键词匹配，找到精确包含关键词的 chunk |
| 合并去重 | `merge_and_deduplicate()` | 两路结果按内容哈希去重，向量结果优先排前 |

#### 为什么加上查询改写

单路召回只用原始查询，但同一个意图有很多表达方式：

```
原始：  "Python 性能优化"
改写1： "如何提升 Python 代码执行速度"
改写2： "Python 性能瓶颈和解决方案"
```

3 个查询分别去向量库和 BM25 检索，每个查 5 条（`k=5`），最多能召回 `3 × 5 × 2 = 30` 条候选。候选多了，后续重排序从中挑最相关的 3 条（`top_n=3`），最终质量就高了。

#### 召回 vs 重排序的分工

```
召回（Recall）          重排序（Re-ranking）
━━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━━━
目标：找全               目标：排准
策略：宁可多找，不能漏     策略：精确排序，只留最好的
速度：快（毫秒级）        速度：慢（需要逐条打分）
结果：5-10 条候选         结果：3 条精选
```

这是搜索系统的经典两阶段架构，Google/Bing 也是这个思路。

#### 召回数量说明

`RETRIEVAL_K = 5` 是 **chunk 数**，不是原始文档数。入库时 PDF 先被切成 chunk，存入 ChromaDB 的也是 chunk。检索返回的每个 `Document` 就是一个 chunk，它的 `metadata` 里有 `source`（来源文件）和 `page`（页码），可以溯源到原始文档。

---

### 交叉编码器重排序（CrossEncoder Reranking）

**要解决什么问题**：多路召回拿回了 5-10 个 chunk，排在前面的不一定是最相关的：

```
向量检索返回（按语义相似度排序）：
  1. "2024 AI Technology Report..."         ← 包含"MMLU"但不是重点
  2. "Llama 3.1 405B 128K 85.2%..."         ← 包含具体分数
  3. "RAG improvements in 2024..."           ← 完全不相关，但语义相近
  4. "Gemini Ultra achieves 90.0%..."        ← 最相关，但排第 4

BM25 返回（按关键词匹配排序）：
  1. "Llama 3.1 405B 128K 85.2%..."         ← 命中"MMLU"关键词
  2. "Key Metrics Comparison..."             ← 命中"score"
```

向量检索觉得"AI 报告概述"跟"MMLU 分数"语义相关（都是 AI 话题），但实际上它并没有回答"谁最高"。两种方式**排序精度都不够**。

#### CrossEncoder 的工作原理

**双塔编码器（Bi-Encoder）**——向量检索用的方式：

```
阶段1（离线）：chunk → 编码器 → 向量 [0.12, 0.85, ...]
阶段2（查询时）：query → 编码器 → 向量 [0.15, 0.79, ...]
                            ↓
              算余弦距离 = 0.95 → "好像相关"

问题：query 和 chunk 从来没"见过面"
     编码器不知道 query 里"MMLU分数最高"到底看重哪个词
```

**交叉编码器（CrossEncoder）**——重排序用的方式：

```
输入：[CLS] 哪个模型MMLU分数最高 [SEP] Gemini Ultra achieves 90.0% [SEP]
                ↓
        Transformer 编码器（同时看到 query 和 chunk）
                ↓
        输出：0.87 ← 精确相关性分数

关键：编码时 query 和 chunk 放在一起
     模型能直接看到"MMLU分数最高"和"Gemini 90%"之间的关联
```

"交叉"指的是 Transformer 的 self-attention 机制在编码时能**同时看到 query 和 doc 的每个词**，让两者交叉交互，所以打分更准。

#### 逐条打分过程

query = `"哪个模型 MMLU 分数最高？"`，5 个候选 chunk：

```
第 1 对：[CLS] 哪个模型MMLU分数最高 [SEP] Gemini Ultra achieves 90.0%... [SEP]
         → 分数 0.92  ★★★ 最相关：直接说 Gemini 90%

第 2 对：[CLS] 哪个模型MMLU分数最高 [SEP] Llama 3.1 405B 128K 85.2%... [SEP]
         → 分数 0.74  ★★  相关：提到了具体分数

第 3 对：[CLS] 哪个模型MMLU分数最高 [SEP] The table above shows... [SEP]
         → 分数 0.68  ★   部分相关：引用了表格数据

第 4 对：[CLS] 哪个模型MMLU分数最高 [SEP] 2024 AI Technology Report... [SEP]
         → 分数 0.31       不相关：只是报告概述

第 5 对：[CLS] 哪个模型MMLU分数最高 [SEP] RAG improvements in 2024... [SEP]
         → 分数 0.08       完全不相关：说的不是模型评分
```

按分数排序后截断（`top_n=3`）：

```
  0.92  Gemini Ultra achieves 90.0%...     ← 留
  0.74  Llama 3.1 405B 128K 85.2%...      ← 留
  0.68  The table above shows...           ← 留
  ────  ─────────────────────────────       ← 截断线
  0.31  2024 AI Technology Report...       ✗ 丢掉
  0.08  RAG improvements in 2024...       ✗ 丢掉
```

#### 模型参数

```
模型名：cross-encoder/ms-marco-MiniLM-L-6-v2

来源：MS MARCO（微软搜索问答数据集）训练
架构：MiniLM（轻量 Transformer），6 层
大小：约 80MB（首次运行自动下载到 ~/.cache/huggingface/）
运行：纯 CPU，本地执行，不需要 API
速度：5 个 chunk 约 50ms
```

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|----------|
| `top_n` | `3`（`config.RERANK_TOP_N`） | 重排序后保留几个 chunk | 3-5 个适合大多数场景。太少可能丢信息，太多会让 LLM 分散注意力 |
| `model_name` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder 模型 | 更高精度可换 `BAAI/bge-reranker-base`，但更大更慢 |

---

### CRAG 文档评分与自适应生成

重排序后拿到了 3 个 chunk，但这 3 个 chunk 真的能回答用户的问题吗？

- 问文档里有的内容 → chunk 相关，能回答
- 问文档里没有的内容（如"诺贝尔奖得主"） → chunk 可能都是凑数的，硬答就是胡说

**CRAG = Corrective RAG**：在生成之前先检查，而不是无脑生成。

#### 评分过程

LLM 当裁判，给"充分性"打分：

```
LLM 收到的 prompt：
─────────────────
请评估以下文档是否能回答用户问题。

用户问题：哪个模型 MMLU 分数最高？

[文档 1] Llama 3.1 405B 128K 85.2%... Gemini Ultra 90%...
[文档 2] 2024 AI Technology Report...
[文档 3] RAG improvements in 2024...

只输出 0.0 ~ 1.0 的数字。

LLM 输出：0.85
```

评分标准：

```
1.0  ████████████████████  完全能回答
0.7  ██████████████▌─────  基本能回答，可能缺细节 ← 阈值线（RELEVANCE_THRESHOLD）
0.4  ████████────────────  有部分相关信息
0.0  ────────────────────  完全无关
```

#### 条件路由

```
              score ≥ 0.7                    score < 0.7
                  │                              │
                  ▼                              ▼
          ┌──────────────┐              ┌──────────────┐
          │ generate_node│              │fallback_node │
          │              │              │              │
          │ LLM 基于检索 │              │ 返回固定文本：│
          │ 结果生成回答  │              │ "知识库没有  │
          │ 带来源引用    │              │  足够信息"   │
          └──────┬───────┘              └──────┬───────┘
                 │                             │
                 ▼                             ▼
              最终回答                       兜底回复
```

对应代码（`graph_nodes.py`）：

```python
# grade 节点计算 needs_fallback
def grade_node(state):
    score = grade_documents(state["original_query"], state["reranked_results"])
    return {
        "relevance_score": score,
        "needs_fallback": score < 0.7,  # ← 这里决定走哪条路
    }

# 路由函数
def route_by_score(state):
    if state.get("needs_fallback", False):
        return "fallback"
    return "generate"

# 在图中用条件路由
builder.add_conditional_edges("grade", route_by_score)
```

#### 生成回答（generate_node）

评分够了，LLM 基于检索到的 chunk 生成回答。prompt 明确要求"只基于文档内容回答，不要编造"，这叫 **grounded generation**（有据生成），回答必须有检索结果支撑：

```
LLM 收到的 prompt：
─────────────────
请根据以下文档回答用户问题。
要求：只基于文档内容，不要编造，标注来源。

[来源 1: tech_report.pdf, 第 0 页]
Gemini Ultra achieves the highest MMLU score at 90.0%...

[来源 2: tech_report.pdf, 第 0 页]
Llama 3.1 405B 128K 85.2%...

用户问题：哪个模型 MMLU 分数最高？

LLM 输出：根据文档1，Gemini Ultra 的 MMLU 分数最高，达到 90.0%。
```

#### 完整示例对比

**能回答的查询**：

```
问："哪个模型 MMLU 分数最高？"
  → rerank: [Gemini 90%, AI Report, RAG improvements]
  → grade:  0.85 ✅ 充分
  → generate: "Gemini Ultra 的 MMLU 分数最高，达到 90.0%"
```

**不能回答的查询**：

```
问："2024 诺贝尔物理学奖得主是谁？"
  → rerank: [勉强相关的 3 个 chunk]
  → grade:  0.2 ❌ 不充分
  → fallback: "抱歉，知识库中没有找到足够的信息..."
```

CRAG 的价值：**不是所有检索结果都值得回答，先检查再决定**。

---

## 关键代码解析

### ChromaDB 向量库

```python
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="my_docs",          # 集合名（类似数据库表名）
    embedding_function=embeddings,      # Embedding 模型
    persist_directory="./chroma_store",  # 持久化目录
)

vectorstore.add_documents(documents=chunks, ids=ids)  # 入库
results = vectorstore.similarity_search("查询文本", k=5)  # 检索
```

**注意**：LangChain 1.0 中 Chroma 迁移到了独立包 `langchain-chroma`，不再从 `langchain_community.vectorstores` 导入。

### 混合检索

```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs, k=5)
vector_retriever = vectorstore.as_retriever(k=5)

ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.4, 0.6],  # BM25 40%, Vector 60%
)
results = ensemble.invoke("查询")
```

- BM25 擅长精确匹配（"Python 3.12"），Vector 擅长语义理解（"最新的 Python 版本"）
- RRF (Reciprocal Rank Fusion) 算法融合两路排序

### CrossEncoder 重排序

```python
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=encoder, top_n=3)
reranked = reranker.compress_documents(documents=docs, query="查询")
```

- bi-encoder：query 和 doc **分别**编码 → 快但粗
- cross-encoder：query 和 doc **拼接**后编码 → 慢但精

### LangGraph 并行（fan-out/fan-in）

```python
builder = StateGraph(RetrievalState)
builder.add_node("rewrite", rewrite_node)
builder.add_node("vector_search", vector_node)
builder.add_node("bm25_search", bm25_node)
builder.add_node("merge", merge_node)

builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "vector_search")   # fan-out
builder.add_edge("rewrite", "bm25_search")     # fan-out
builder.add_edge("vector_search", "merge")     # fan-in
builder.add_edge("bm25_search", "merge")       # fan-in
```

LangGraph 自动检测并行边并并发执行。fan-in 时 reducer（`operator.add`）自动合并结果。

## 常见问题

### 1. `ModuleNotFoundError: No module named 'langchain_chroma'`

```bash
pip install langchain-chroma
```

### 2. CrossEncoder 首次运行很慢

首次下载模型约 80MB，之后缓存在 `~/.cache/huggingface/`。

### 3. ChromaDB 数据重复

调用 `reset_collection("集合名")` 清空，或使用不同的 `collection_name`。

### 4. UnstructuredPDFLoader 安装失败

依赖系统级工具（poppler、tesseract），Windows 较复杂。代码自动 fallback 到 PyPDFLoader。

### 5. Windows 中文乱码

代码已含 UTF-8 修复（在 `utils.py` 中）。

### 6. `similarity_search` 返回空

确保已运行入库、`collection_name` 一致、ChromaDB 目录存在。

## 最佳实践

### 切片参数调优

| 场景 | chunk_size | chunk_overlap | 理由 |
|------|-----------|---------------|------|
| FAQ/短文本 | 200-300 | 20-30 | 小粒度更精确 |
| 技术文档 | 500-800 | 50-100 | 平衡完整性和精度 |
| 长报告 | 800-1200 | 100-200 | 保留更多上下文 |

### 检索权重调优

| 场景 | BM25 | Vector | 理由 |
|------|------|--------|------|
| 精确查找 | 0.6-0.7 | 0.3-0.4 | 关键词更重要 |
| 概念问答 | 0.3-0.4 | 0.6-0.7 | 语义更重要 |
| 混合 | 0.4-0.5 | 0.5-0.6 | 均衡 |

### 企业级 RAG 检查清单

- [ ] Embedding 模型是否支持目标语言
- [ ] 切片大小是否适合文档类型
- [ ] 是否使用混合检索
- [ ] 是否有重排序步骤
- [ ] 是否有文档充分性评估（CRAG）
- [ ] 是否有 fallback 机制
- [ ] 是否有来源引用
- [ ] 是否处理了多模态内容

## 常用 API 速查

```python
# 文档解析
from langchain_community.document_loaders import PyPDFLoader
pages = PyPDFLoader("file.pdf").load()

# 文本切片
from langchain_text_splitters import RecursiveCharacterTextSplitter
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)

# Embedding
from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 向量库
from langchain_chroma import Chroma
vs = Chroma(collection_name="docs", embedding_function=emb, persist_directory="./db")
vs.add_documents(chunks)
results = vs.similarity_search("query", k=5)

# BM25
from langchain_community.retrievers import BM25Retriever
bm25 = BM25Retriever.from_documents(chunks, k=5)

# 混合检索
from langchain_classic.retrievers import EnsembleRetriever
ensemble = EnsembleRetriever(retrievers=[bm25, vs.as_retriever(k=5)], weights=[0.4, 0.6])

# 重排序
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    top_n=3
)

# LangGraph
from langgraph.graph import StateGraph, START, END
builder = StateGraph(MyState)
builder.add_node("name", node_func)
builder.add_edge(START, "name")
builder.add_edge("name", END)
graph = builder.compile()
result = graph.invoke({"field": "value"})
```

## 进一步学习

- **向量模型选型**：参考 [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **生产部署**：ChromaDB → Milvus/Weaviate（分布式）、加入 API 层和缓存
- **高级 RAG**：多跳检索、自适应检索、Query Fusion、HyDE
- **LangSmith**：接入 LangSmith 进行可观测性追踪
- **多模态 RAG**：图像理解、OCR、视觉语言模型结合
