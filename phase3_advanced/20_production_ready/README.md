# 生产级部署 (Production Ready)

## 快速开始

运行示例：

```bash
python phase3_advanced/20_production_ready/main.py
```

## 核心概念

### 为什么需要"生产级"考量？

学习阶段的代码跑通即可，生产环境的代码需要在**故障、压力、异常**下依然稳定运行。

| 问题 | 学习阶段 | 生产阶段 |
|------|---------|---------|
| 服务器重启 | 重新运行脚本 | 用户对话状态必须恢复 |
| API 超时 | 手动重跑 | 自动重试 + 兜底 |
| 异常抛出 | 脚本崩溃 | 捕获 → 记录 → 降级 |
| 逻辑死循环 | 手动 Ctrl+C | 步数限制自动终止 |
| 排查问题 | print 调试 | LangSmith 全链路追踪 |

### 四大支柱

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   持久化状态     │  │   容错与重试     │  │   错误处理       │  │   可观测性       │
│  Checkpoint     │  │  RetryPolicy    │  │  Error Routing  │  │  LangSmith      │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 关键 API

### Checkpoint 持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver  # 仅开发用

# SQLite 内存（演示）
checkpointer = SqliteSaver.from_conn_string(":memory:")

# SQLite 文件（单实例生产）
checkpointer = SqliteSaver.from_conn_string("app.db")

# 编译时传入
graph = builder.compile(checkpointer=checkpointer)

# 调用时必须指定 thread_id
graph.invoke(inputs, config={"configurable": {"thread_id": "user-123"}})
```

**为什么需要 thread_id？**

Checkpoint 按 `thread_id` 存储状态历史。同一个 thread_id 的调用共享同一条状态时间线——这是实现"对话恢复"的基础。

### RetryPolicy 自动重试

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(
        max_attempts=3,        # 总尝试次数（1次正常 + 2次重试）
        initial_interval=0.5,  # 首次重试前等待秒数
        backoff_factor=2.0,    # 间隔倍增系数
        max_interval=128.0,    # 最大等待间隔
        jitter=True,           # 随机抖动，避免共振
        retry_on=[ConnectionError, TimeoutError],  # 只重试这些异常
    ),
)
```

**RetryPolicy 的局限：**

- 重试耗尽后**直接抛异常**，没有"失败后去哪"的钩子
- 同一 superstep 中一个分支失败，其他分支的状态更新也会被回滚
- 适合：网络抖动、API 限流等**瞬态故障**

### 手动重试 + 兜底

```python
from langgraph.types import Command

def call_with_fallback(state):
    for i in range(3):
        try:
            return {"data": api.call()}
        except TimeoutError:
            time.sleep(0.5 * (2 ** i))
    # 耗尽后返回错误标记，由条件边路由到兜底
    return {"error": "exhausted"}

def router(state):
    if state.get("error"):
        return "fallback"
    return END
```

### 三层错误处理架构

```
应用层（App）          全局 try/except + 报警 + 熔断
    │
图  层（Graph）        条件边根据 state["error"] 路由
    │
节点层（Node）         try/except 捕获预期异常 → 写回 State
```

| 层级 | 职责 | 不做什么 |
|------|------|---------|
| 节点层 | 捕获**已知的、可恢复**的异常 | 不要 `except Exception` 吞掉所有错误 |
| 图层 | 根据错误状态选择重试/兜底/终止 | 不要在这里做业务逻辑 |
| 应用层 | 日志、监控、报警、熔断 | 不要替代图做路由决策 |

## 工作流程

### Checkpoint 持久化流程

```
1. 选择 checkpointer（SqliteSaver / PostgresSaver）
2. compile(checkpointer=checkpointer) 绑定到图
3. 每次 invoke 传入 thread_id
4. 状态自动保存到数据库
5. 进程重启后，相同 thread_id 可恢复状态
```

### 节点重试流程

```
1. 节点抛出可重试异常
2. LangGraph 捕获异常
3. 等待 initial_interval * backoff_factor^n
4. 重新执行节点
5. 成功 → 继续 / 耗尽 → 抛出异常
```

### 错误降级流程

```
1. 节点捕获异常 → 写 error 字段到 State
2. 条件边读取 error 字段
3. error_count < 阈值 → 路由回重试节点
4. error_count >= 阈值 → 路由到 fallback 节点
5. fallback 返回默认值/缓存/友好提示
```

## 关键代码片段

### 生产级 Checkpoint 配置

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

builder = StateGraph(State)
# ... 添加节点和边 ...

# 开发/测试：内存 SQLite
# checkpointer = SqliteSaver.from_conn_string(":memory:")

# 生产环境：文件持久化
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

graph = builder.compile(checkpointer=checkpointer)

# 必须为每次调用提供 thread_id
config = {"configurable": {"thread_id": "conversation-123"}}
result = graph.invoke({"query": "你好"}, config)
```

### 带步数限制的安全循环

```python
class State(TypedDict):
    step_count: int

def safe_node(state: State) -> dict:
    step = state.get("step_count", 0) + 1
    if step > 10:
        return {"step_count": step, "result": "[终止] 步数超限"}
    # ... 正常逻辑 ...
    return {"step_count": step}

def router(state: State) -> str:
    if state.get("step_count", 0) > 10:
        return END
    return "safe_node"
```

### LangSmith 接入（零代码改动）

```python
import os

# 在 .env 文件中配置：
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=ls-xxx
# LANGCHAIN_PROJECT=my-project

# 代码中无需任何修改，LangGraph 自动上报
graph.invoke(inputs, config)
```

## 常见问题

**Q: MemorySaver 和 SqliteSaver 有什么区别？**

- `MemorySaver`：纯内存，速度快，进程结束数据丢失。仅用于开发和单元测试。
- `SqliteSaver`：SQLite 持久化，单实例生产可用。多实例部署用 `PostgresSaver`。

**Q: RetryPolicy 的 max_attempts 是重试次数还是总次数？**

总次数。`max_attempts=3` 表示：1 次正常执行 + 最多 2 次重试。

**Q: RetryPolicy 和手动重试怎么选？**

| 场景 | 推荐方案 |
|------|---------|
| 简单的网络超时 | RetryPolicy |
| 需要根据响应动态调整策略 | 手动重试 |
| 重试耗尽后需要走 fallback | 手动重试 |
| 需要记录每次重试的详细日志 | 手动重试 |

**Q: 节点里 `except Exception` 有什么问题？**

会吞掉所有异常，包括编程错误（TypeError、KeyError）。这会导致：
1. Bug 被隐藏，LangSmith 显示节点"成功"
2. 错误状态被写入 State，图可能进入错误的降级路径

**正确做法**：只 catch 你知道怎么处理的特定异常。

```python
# 错误
try:
    result = api.call()
except Exception as e:  # 吞掉所有错误
    return {"error": str(e)}

# 正确
try:
    result = api.call()
except (ConnectionError, TimeoutError) as e:  # 只 catch 网络问题
    return {"error": str(e)}
```

**Q: 为什么需要步数限制？**

条件边的路由逻辑如果有 bug（比如忘记更新某个字段），可能导致无限循环。步数限制是最后一道保险。

## 最佳实践

1. **生产环境绝不使用 MemorySaver**：负载均衡下每个实例的内存独立，用户请求可能落到任意实例
2. **始终传入 thread_id**：没有 thread_id 的调用不会被持久化
3. **RetryPolicy 配在"外部调用"节点上**：LLM API、数据库、第三方服务
4. **错误状态要结构化**：不要只存字符串，存 `{"type": "timeout", "count": 2}` 便于路由判断
5. **步数限制 + Token 监控**：防止逻辑死循环和上下文膨胀拖垮成本
6. **LangSmith 是必备项**：生产排障没有追踪数据等于盲飞

## 下一步学习

- **phase4_projects**：将 checkpoint、重试、错误处理、子图等知识整合到完整项目
- 推荐阅读：[LangGraph Error Handling Patterns for Production](https://focused.io/lab/langgraph-agent-error-handling-production)
