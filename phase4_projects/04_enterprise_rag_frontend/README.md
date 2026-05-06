# Enterprise RAG Frontend

## 运行

```bash
npm install
npm run dev
```

## 接口地址配置

复制 `.env.example` 为 `.env`，按需修改：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8087
```

前端默认会调用：

- `POST /api/v1/ingestion/multimodal`
- `POST /api/v1/retrieval/full-pipeline/stream`

## 页面能力

- 文档上传入库
- 多轮问答
- 基于 SSE 的流式逐字输出
- 最近几轮对话自动回传给后端作为上下文
