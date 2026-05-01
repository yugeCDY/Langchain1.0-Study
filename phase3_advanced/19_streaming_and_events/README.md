# 流式输出与事件 (Streaming & Events)

## 快速开始

运行示例：

```bash
python phase3_advanced/19_streaming_and_events/main.py
```

## 核心概念

### 为什么需要流式输出？

| 场景 | invoke（同步） | stream（流式） |
|------|--------------|---------------|
| 聊天机器人 | 用户干等 5 秒，然后一次性弹出大段文字 | 像 ChatGPT 一样逐字输出，体验更好 |
| 多步骤工作流 | 黑盒执行，不知道进展到哪一步 | 每完成一个节点就推送更新，进度可见 |
| 耗时任务 | 用户不知道系统是否卡死 | 实时推送进度状态，减少焦虑 |

### 核心方法

```python
# 同步阻塞：等整个图执行完，返回最终结果
result = graph.invoke({"topic": "xxx"})

# 同步流式：每执行完一个节点，实时推送一次
for chunk in graph.stream({"topic": "xxx"}):
    print(chunk)

# 异步流式：async/await 版本
async for chunk in graph.astream({"topic": "xxx"}):
    print(chunk)
```

## stream_mode 详解

### `stream_mode="updates"`（增量模式）

只输出**节点返回的增量**——即节点 return 的那个 dict。

```python
for chunk in graph.stream(inputs, stream_mode="updates"):
    # chunk 格式：{node_name: {field: value}}
    for node_name, update in chunk.items():
        print(f"[{node_name}] 更新了: {update}")
```

**适合**：前端进度条、通知系统、只需要"变了什么"的场景。

### `stream_mode="values"`（全量模式）

输出**完整的 State 快照**——每次推送都是当时整个 State 的全貌。

```python
for chunk in graph.stream(inputs, stream_mode="values"):
    # chunk 格式：完整的 State 字典
    print(f"当前状态: {chunk}")
```

**适合**：需要渲染完整 UI、依赖全量状态做判断的场景。

### `stream_mode="messages"`（Token 流式）

捕获 LLM 调用产生的 token，**逐字实时输出**。

```python
for chunk in graph.stream(inputs, stream_mode="messages"):
    # chunk 格式：(message_chunk, metadata)
    msg_chunk, metadata = chunk
    print(msg_chunk.content, end="", flush=True)
```

**适合**：聊天 UI、内容生成界面，实现打字机效果。

**前提**：LLM 后端需要支持 SSE 流式传输。

### `stream_mode="custom"`（自定义事件）

节点内部通过 `get_stream_writer()` 发送任意自定义数据。

```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()
    writer({"step": 1, "status": "正在处理..."})  # ← 发送自定义事件
    # ... 做更多工作 ...
    writer({"step": 2, "status": "处理完成"})
    return {"result": "done"}
```

外部消费：

```python
for chunk in graph.stream(inputs, stream_mode="custom"):
    # chunk 就是 writer() 传入的数据
    print(f"进度: {chunk['step']} - {chunk['status']}")
```

**适合**：进度条、工具状态通知、中间结果展示。

## 关键代码片段

### invoke vs stream 对比

```python
# invoke：阻塞，返回最终结果
result = graph.invoke({"topic": "LangGraph"})
print(result)  # 一次性输出完整 State

# stream：实时，节点执行完就推送
for chunk in graph.stream({"topic": "LangGraph"}):
    # 每轮 chunk = {node_name: {field: value}}
    for node_name, update in chunk.items():
        print(f"[{node_name}] {update}")
```

输出对比：

```
【invoke 结果】
{'topic': 'LangGraph', 'plan': '计划：围绕「LangGraph」收集素材并撰写', 'result': 'LangGraph 是一个...'}

【stream 结果】
[plan] {'plan': '计划：围绕「LangGraph」收集素材并撰写'}
[generate] {'result': 'LangGraph 是一个...'}
```

### updates vs values 对比

假设 State 有 `topic`、`plan`、`result` 三个字段。

```python
# updates 模式：只看增量
for chunk in graph.stream(inputs, stream_mode="updates"):
    # chunk = {'plan': {'plan': '计划...'}}
    # chunk = {'generate': {'result': 'LangGraph 是...'}}
    pass

# values 模式：看全量 State
for chunk in graph.stream(inputs, stream_mode="values"):
    # chunk = {'topic': 'LangGraph', 'plan': '计划...', 'result': ''}
    # chunk = {'topic': 'LangGraph', 'plan': '计划...', 'result': 'LangGraph 是...'}
    pass
```

### messages 模式（打字机效果）

```python
from langchain_core.messages import HumanMessage
from typing import Annotated
import operator

class ChatState(TypedDict):
    messages: Annotated[list, operator.add]  # messages 字段必须存在

def chat_node(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(ChatState)
builder.add_node("assistant", chat_node)
builder.add_edge(START, "assistant")
graph = builder.compile()

# 流式输出 tokens
print("AI: ", end="", flush=True)
for chunk in graph.stream(
    {"messages": [HumanMessage(content="你好")]},
    stream_mode="messages",
):
    msg_chunk, metadata = chunk
    print(msg_chunk.content, end="", flush=True)
print()
```

### 自定义进度事件

```python
from langgraph.config import get_stream_writer

def long_task(state):
    writer = get_stream_writer()

    writer({"progress": 10, "msg": "正在连接数据库..."})
    # ... 执行操作 ...

    writer({"progress": 50, "msg": "正在分析数据..."})
    # ... 执行操作 ...

    writer({"progress": 100, "msg": "完成！"})
    return {"result": "分析完成"}

# 消费自定义事件
for event in graph.stream(inputs, stream_mode="custom"):
    print(f"[{event['progress']}%] {event['msg']}")
```

### 子图流式传播

```python
# 子图
subgraph = StateGraph(State)
subgraph.add_node("inner_a", node_a)
subgraph.add_node("inner_b", node_b)
sub = subgraph.compile()

# 父图
builder = StateGraph(State)
builder.add_node("sub", sub)
graph = builder.compile()

# 默认：子图是黑盒，只能看到 "sub" 节点的输出
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)  # 只有 {sub: {...}}

# subgraphs=True：子图内部节点的事件也可见
for chunk in graph.stream(inputs, stream_mode="updates", subgraphs=True):
    print(chunk)  # 包含 inner_a、inner_b 的事件
```

## 工作流程

### 基本流式输出流程

```
用户输入
   ↓
graph.stream(inputs, stream_mode="updates")
   ↓
节点 A 执行完 → 推送 {A: {field: value}}
   ↓
节点 B 执行完 → 推送 {B: {field: value}}
   ↓
节点 C 执行完 → 推送 {C: {field: value}}
   ↓
流结束
```

### 带自定义进度的事件流

```
用户输入
   ↓
graph.stream(inputs, stream_mode="custom")
   ↓
节点内 writer({"step": 1}) → 推送 {"step": 1}
   ↓
节点内 writer({"step": 2}) → 推送 {"step": 2}
   ↓
节点 return → 正常结束
```

## 常见问题

**Q: stream 和 invoke 返回的数据格式一样吗？**

不一样。invoke 返回完整的 State 字典；stream 的每个 chunk 是一个事件，格式取决于 stream_mode。

**Q: messages 模式为什么不输出 token？**

需要满足两个条件：
1. 图中节点调用了 LLM（model.invoke）
2. LLM 后端支持 SSE 流式传输

如果模型不支持流式，messages 模式可能只输出完整消息。

**Q: get_stream_writer() 可以在节点外面用吗？**

不可以。只能在节点函数（被 add_node 注册的函数）内部调用。

**Q: 可以同时使用多种 stream_mode 吗？**

可以。传入列表：`stream_mode=["updates", "custom"]`，每个 chunk 会带一个 mode 标记。

**Q: stream 支持异步吗？**

支持。用 `graph.astream()` + `async for`。

```python
async for chunk in graph.astream(inputs, stream_mode="updates"):
    print(chunk)
```

**Q: subgraphs=True 对性能有影响吗？**

有轻微开销，因为需要处理更多事件。只在需要调试子图时开启。

## 最佳实践

1. **聊天界面用 messages 模式**：实现打字机效果，用户体验最佳
2. **工作流进度用 custom 模式**：进度条、状态通知，让用户知道系统在忙什么
3. **后台任务用 invoke**：不需要实时反馈，直接等结果即可
4. **调试子图用 subgraphs=True**：开发阶段开启，生产环境关闭
5. **updates 模式省带宽**：前端只需要增量更新时，不要用 values 模式传全量 State
6. **异步场景用 astream**：Web 服务、高并发场景，避免阻塞事件循环

## 下一步学习

- **20_production_ready**：生产环境部署、LangSmith 监控、错误处理、成本控制
- 尝试把 custom 模式应用到实际项目，给耗时任务加进度条
- 用 messages 模式实现一个带打字机效果的聊天 UI
