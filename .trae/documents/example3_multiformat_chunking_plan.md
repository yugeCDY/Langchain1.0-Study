# 示例 3 多格式文档切分改造计划（PDF+DOCX+CSV）

## 1) Summary

目标是在 `示例 3：多模态文档处理` 中，直接支持你当前确认的企业场景：

* 支持格式：`PDF + DOCX + CSV`（同一批次混合处理）

* 切分策略：`表格整块保留 + 文本分层切分（parent/child）`

* 集合策略：`按业务域分集合`（如 `resume_kb`、`crawl_kb`、`science_kb`）

* 多语言策略：`统一多语 Embedding + 语言元数据`

* 权限策略：`文档级 ACL`（检索阶段强制过滤）

交付形态为：在不重构整体架构的前提下，重点改造 `main.py` 的示例 3 入口与 `ingestion.py/retrieval.py` 的相关函数，使其可直接处理你 `data/samples` 下的异构文档并具备权限过滤能力。

***

## 2) Current State Analysis

基于仓库现状确认：

* `main.py` 的 `example_3_multimodal_processing()` 目前只处理单个 `tech_report_sample.pdf`，没有批量扫描目录，也没有按文件类型分派解析器。

* `ingestion.py` 现有解析函数：

  * `parse_simple_pdf()`：仅 PDF（PyPDFLoader）

  * `parse_advanced_pdf()`：仅 PDF（UnstructuredPDFLoader elements 模式，失败回退 simple）

* `chunk_documents()` 已具备两阶段切分（parent/child），且对中英文分隔符有初步支持，适合作为文本主切分逻辑继续复用。

* 表格处理目前只在“有 `element_type=Table` 时不切片”；该机制主要来自 PDF elements，对 DOCX/CSV 尚未统一。

* `retrieval.py` 当前检索 SQL 仅按 `collection_name` 过滤，无 ACL/租户/角色过滤条件。

* `data/samples` 已有多类型样本（PDF + DOCX），CSV 将作为本次新支持类型纳入示例 3 流程。

***

## 3) Proposed Changes

### A. `ingestion.py`：多格式解析与统一切分入口

#### 变更文件

* `phase4_projects/03_enterprise_rag/ingestion.py`

#### 改什么

* 新增统一解析入口（如 `parse_document_by_type()`）：

  * `.pdf`：优先 `parse_advanced_pdf()`，失败回退 `parse_simple_pdf()`

  * `.docx`：使用 `UnstructuredWordDocumentLoader`（或可用等价 loader）解析为结构元素；不可用时做文本级回退

  * `.csv`：按行/分块转 `Document`，并标记 `element_type="Table"`

* 新增目录批量解析入口（如 `parse_documents_in_dir()`）：

  * 扫描指定目录，按扩展名过滤 `pdf/docx/csv`

  * 逐文件容错（单文件失败记录到错误列表，不中断全批）

  * 元数据标准化：`source`、`file_type`、`domain`、`language`、`acl`、`element_type`

* 扩展元素分离逻辑（`separate_by_element_type`）：

  * 保持 `Table` 走整块路径

  * 文本元素（Narrative/Title/Header/Paragraph 等）走 `chunk_documents()`

* 语言感知切分增强（在 `chunk_documents()` 中）：

  * 保留现有中英文分隔符

  * 明确写入 `chunk_level`、`parent_chunk_id`、`language`

#### 为什么这样做

* 最小改动复用现有 `chunk_documents()` 双阶段能力。

* 让“格式处理”与“切分策略”解耦，后续扩展 `xlsx/html` 成本低。

* 元数据标准化后，ACL 和业务域过滤才能在检索端可靠执行。

#### 如何实现（关键约束）

* 不引入全新图结构，优先在现有函数层完成，保持教学代码可读性。

* 对不可用依赖（如 unstructured word）保持 graceful fallback，不阻断示例运行。

***

### B. `main.py`：重写示例 3 为“目录级多格式入库演示”

#### 变更文件

* `phase4_projects/03_enterprise_rag/main.py`

#### 改什么

* 改造 `example_3_multimodal_processing()`：

  * 输入从单文件改为目录（默认 `SAMPLES_DIR`）

  * 支持业务域参数（默认示例值，可在函数内演示 `resume_kb/crawl_kb/science_kb` 之一）

  * 调用新的统一解析/批处理函数，输出每种格式的解析统计、切片统计、失败统计

  * 执行“文本切分 + 表格整块”合并入库

  * 入库集合使用业务域集合名（非按格式）

* 增加多语言与 ACL 示例元数据注入说明（演示级）：

  * 如 `language` 自动识别/默认值

  * `acl`（如 `allowed_roles`）写入 metadata

#### 为什么这样做

* 用户明确要求“直接在示例 3 改”。

* 示范入口最直观，学习者只跑一个示例就能看到异构文档处理全链路。

***

### C. `retrieval.py`：文档级 ACL 过滤（检索强制）

#### 变更文件

* `phase4_projects/03_enterprise_rag/retrieval.py`

#### 改什么

* 为 `vector_search()`、`bm25_search()` 增加可选过滤参数（如 `acl_roles`、`domain`、`language`）。

* PostgreSQL 查询增加 metadata 过滤条件（JSONB）：

  * 先过滤业务域/ACL，再做向量排序或 BM25 排序。

* 保持默认兼容：

  * 不传 ACL 参数时，沿用当前行为，避免破坏既有示例 5/6/7。

#### 为什么这样做

* 你已明确要求“文档级 ACL”。

* 仅在 ingestion 写 ACL 不够，必须在 retrieval 阶段强制执行才有安全意义。

***

### D. `README.md`（可选但建议）

#### 变更文件

* `phase4_projects/03_enterprise_rag/README.md`

#### 改什么

* 更新示例 3 描述：从“PDF 多模态”升级为“多格式多模态 + 业务域集合 + ACL”。

* 补充字段约定：`domain`、`language`、`allowed_roles`。

#### 为什么这样做

* 文档需与代码一致，避免后续学习/维护误导。

***

## 4) Assumptions & Decisions

### 已锁定决策（来自本次确认）

* 支持范围：`PDF + DOCX + CSV`

* 默认切分：`表格整块 + 文本分层切分`

* 集合策略：`按业务域分集合`（不是按文件格式）

* 多语言：`统一多语 embedding`，并在 metadata 中写入语言信息

* 权限：`文档级 ACL`，在检索阶段强制过滤

* 本轮范围：`P0 + P1(文档版本管理)`

### 实现假设（执行前沿用）

* DOCX 解析优先使用现有依赖可达的 loader；若缺依赖则打印告警并回退，不中断批处理。

* CSV 以“行/小批量行”为表格块处理，不做字符级打散。

* ACL 字段暂采用 metadata 中统一键（如 `allowed_roles`），并以“请求方角色集合与文档角色集合有交集”作为命中规则。

* 版本管理优先采用 `metadata` 字段实现（`doc_id/version/is_active`），避免本轮引入数据库表结构迁移。

***

## 5) Verification Steps

执行阶段按以下顺序验证：

1. 运行示例 3，确认目录内 `pdf/docx/csv` 都被识别并进入处理流程（含失败统计）。
2. 检查切分结果统计：文本块数量 > 0；表格块保留整块（不被 child splitter 再切）。
3. 检查入库 metadata：每条记录含 `domain/language/file_type/allowed_roles`。
4. 用同一查询分别以不同 `acl_roles` 检索，验证结果集合变化符合权限预期（无权文档不返回）。
5. 运行现有示例 5/6 的基础路径，确认默认参数下无回归报错（兼容性验证）。

***

## 6) Implementation Order

1. `ingestion.py`：统一解析与批量入口 + 元数据标准化（含 `doc_id/version/is_active`）
2. `ingestion.py`：加入幂等去重、质量门控、失败记录与统计埋点（P0）
3. `main.py`：示例 3 改为目录级多格式演示 + 版本化入库示例
4. `retrieval.py`：ACL/业务域/语言过滤 + 仅检索 `is_active=true` 的最新有效版本
5. `README.md`：同步字段规范与版本管理规则
6. 运行验证脚本与示例流程，修正兼容问题

***

## 7) 企业级录入建议补充逻辑（建议分阶段）

### P0（建议本轮纳入）

1. 幂等与去重：基于 `source + content_hash + version` 防止重复入库；支持增量重跑。
2. 数据质量门控：空文档、超短文本、乱码页、低信息密度块直接丢弃或降权。
3. 失败重试与死信：解析失败记录错误码，支持重试队列，避免批任务整体失败。
4. 可观测性：记录每文件耗时、切块数、失败率、入库条数，便于线上排障。

### P1（建议下一阶段）

1. 文档版本管理：同一 `doc_id` 支持版本号与生效状态，便于回滚和审计。（本轮已选）
2. OCR 与扫描件兜底：对图片型 PDF/文档走 OCR 流程，避免“有文件无文本”。
3. 敏感信息治理：PII/密级识别并打标签，和 ACL 联动进行检索拦截。
4. 查询审计：保留“谁在什么时间以什么角色检索了什么域”的审计日志。

### P2（可选增强）

1. 语义缓存与冷热分层：高频文档优先索引，低频文档延迟加载。
2. 自动重切分策略：根据召回质量反馈动态调整 chunk 参数。
3. 近重复检测：跨文档 near-duplicate 合并，降低冗余与噪声召回。

### 本轮版本管理落地细则（已纳入）

1. 入库时为每个业务文档生成稳定 `doc_id`（不含版本号），并维护 `version`（递增）。
2. 新版本入库后，将同 `doc_id` 的旧版本标记 `is_active=false`，新版本 `is_active=true`。
3. 检索 SQL 默认附加 `is_active=true` 过滤，避免召回历史版本混淆答案。
4. 回滚通过切换 `is_active` 实现，不物理删除历史版本。

