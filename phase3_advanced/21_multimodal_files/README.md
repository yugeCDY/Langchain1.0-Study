# 多模态与文件处理 (Multimodal & File Handling)

## 快速开始

运行示例：

```bash
python phase3_advanced/21_multimodal_files/main.py
```

## 核心概念

### 为什么需要多模态？

现实世界的信息不只是文字：
- 用户上传一张截图问"这个报错什么意思"
- 产品说明书里有大量图表
- 发票、合同、证件需要 OCR 识别

多模态 = 让 LLM 同时理解**文本 + 图像 + 其他格式**。

### LangChain 多模态消息格式

```python
from langchain_core.messages import HumanMessage

# 单模态（纯文本）
msg_text = HumanMessage(content="你好")

# 多模态（文本 + 图片）
msg_multimodal = HumanMessage(content=[
    {"type": "text", "text": "描述这张图片"},
    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
])
```

`content` 可以是：
- `str` — 纯文本（向后兼容）
- `list[dict]` — 多模态内容块列表

### 内容块类型

| 类型 | 格式 | 说明 |
|------|------|------|
| `text` | `{"type": "text", "text": "..."}` | 文本内容 |
| `image_url` | `{"type": "image_url", "image_url": {"url": "..."}}` | 图片 URL 或 base64 |

### 图片来源

**方式一：在线 URL**
```python
{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
```

**方式二：Base64 Data URI（不依赖外部网络）**
```python
import base64

with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

url = f"data:image/jpeg;base64,{b64}"
{"type": "image_url", "image_url": {"url": url}}
```

### PDF 解析方案对比

| 方案 | 依赖 | 优势 | 劣势 |
|------|------|------|------|
| PyPDFLoader | `pypdf` | 轻量、速度快 | 只能提取纯文本 |
| UnstructuredPDFLoader | `unstructured[pdf]` | 识别表格、图片、标题 | 安装复杂、速度慢 |
| PDFPlumber | `pdfplumber` | 表格识别精准 | 仅文本+表格，无图片 |

## 关键代码片段

### 图像理解

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(content=[
    {"type": "text", "text": "这张图片里有什么？"},
    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
])

response = model.invoke([message])
print(response.content)
```

### PDF 解析

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()

for doc in documents:
    print(f"第 {doc.metadata['page']} 页: {doc.page_content[:100]}")
```

### 多模态消息 + System 角色

```python
from langchain_core.messages import HumanMessage, SystemMessage

system = SystemMessage(content="你是一个能看懂图片的助手。")
human = HumanMessage(content=[
    {"type": "text", "text": "分析这张图表的趋势"},
    {"type": "image_url", "image_url": {"url": "..."}},
])

response = model.invoke([system, human])
```

## 常见问题

**Q: 我的模型不支持图片输入怎么办？**

多模态需要模型本身支持（如 GPT-4o、Gemini Pro Vision、Qwen-VL）。如果模型不支持，会报错。可以在代码中捕获异常并提示用户。

**Q: 可以用本地图片文件吗？**

可以。读取文件为 base64，构造 `data:image/jpeg;base64,{base64_string}` 格式的 URL。

```python
import base64

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
url = f"data:image/jpeg;base64,{b64}"
```

**Q: PDF 解析后如何用于 RAG？**

解析得到 `Document` 列表后，用 `RecursiveCharacterTextSplitter` 切片，然后 `embed_and_store` 存入向量库。流程和文本 RAG 完全一样。

**Q: 多模态 RAG 是什么？**

传统 RAG 只检索文本。多模态 RAG 还会：
1. 用视觉模型提取图片中的文字/信息
2. 把图片也编码成向量参与检索
3. 生成回答时同时参考文本和图片内容

## 最佳实践

1. **优先用 base64**：避免外部 URL 失效或被墙
2. **图片大小要控制**：太大的图片会增加 token 消耗和延迟
3. **PDF 解析选合适的工具**：纯文本用 PyPDFLoader，复杂布局用 Unstructured
4. **多模态消息按顺序排列**：先放文本说明，再放图片，模型理解更顺畅
5. **记得捕获异常**：不是所有模型都支持多模态

## 下一步学习

- **phase4_projects**：把多模态能力整合到实际项目中
- 尝试用视觉模型提取图片中的表格数据
- 探索多模态向量检索（图片也能被语义搜索）
