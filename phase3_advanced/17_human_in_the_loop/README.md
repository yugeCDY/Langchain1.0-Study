# 人在回路 (Human-in-the-Loop)

## 快速开始

运行示例：

```bash
python phase3_advanced/17_human_in_the_loop/main.py
```

## 核心概念

### 什么是 HITL？

Human-in-the-Loop = 在 AI 工作流执行过程中，**暂停并等待人类输入**，然后根据人类决策继续执行。

**为什么需要 HITL？**

| 场景 | 没有 HITL | 有 HITL |
|------|----------|---------|
| 生成内容直接发布 | 可能出错，后果严重 | 人工审核后再发布 |
| Agent 调用敏感工具 | 可能误操作（发邮件、转账） | 用户确认后再执行 |
| 长流程自动执行 | 中间出错难以发现 | 每步人工确认 |
| AI 决策不确定 | 全自动化风险高 | 关键节点人工介入 |

### HITL 的核心机制

LangGraph 的 HITL 基于两个 API：

```
节点执行中:
  1. interrupt(payload)         -- 暂停，把 payload 展示给用户
  2. 等待用户响应...
  3. Command(resume=value)      -- 恢复，value 传给 interrupt() 作为返回值
  4. 节点继续执行后面的代码
```

**关键理解：** `interrupt()` 不是打印或日志，它真的会**停止图执行**。图停在当前节点，状态被 checkpointer 保存。等 `Command(resume=...)` 后，同一个节点**重新执行**，但 `interrupt()` 这次直接返回 resume 的值。

## 关键 API

### interrupt()

```python
from langgraph.types import interrupt

def review_node(state: State):
    # interrupt 暂停执行，payload 可以是任意可序列化数据
    human_input = interrupt({
        "question": "是否批准？",
        "content": state["content"],
        "options": ["approve", "reject"],
    })

    # 恢复后，human_input = resume 传的值
    if human_input == "approve":
        return {"status": "approved"}
    return {"status": "rejected"}
```

### Command(resume=...)

```python
from langgraph.types import Command

# 简单恢复
graph.invoke(Command(resume="approve"), config)

# 同时修改 state 再恢复
graph.invoke(
    Command(update={"draft": "用户修改后的内容"}, resume="edit"),
    config,
)
```

### graph.get_state()

```python
# 查看当前中断状态
state = graph.get_state(config)
print(state.interrupts[0].value)  # interrupt() 传的 payload
```

### checkpointer（必备）

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 内存版：演示用，进程结束丢失
graph = builder.compile(checkpointer=InMemorySaver())

# 文件版：持久化，进程重启可恢复
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))
```

**没有 checkpointer，interrupt 后无法 resume。**

## 工作流程

### 最简单的 HITL 流程

```
1. graph.invoke(initial_state, config)   -- 启动图
2. 图执行到 interrupt() 处暂停
3. graph.get_state(config)               -- 查看 interrupt payload
4. graph.invoke(Command(resume=...), config)  -- 恢复执行
5. 图继续执行到 END
```

### 带状态修改的 HITL 流程

```
1. graph.invoke(initial_state, config)
2. 图执行到 interrupt() 处暂停
3. 用户修改内容（如编辑草稿）
4. graph.invoke(Command(update={"draft": "新内容"}, resume="edit"), config)
5. 图恢复，state["draft"] 已更新
6. 图继续执行到 END
```

## 关键代码片段

### 审批流程

```python
def review_node(state: State):
    human_input = interrupt({
        "question": "是否批准？",
        "content": state["content"],
        "options": ["approve", "reject"],
    })
    return {"status": "approved" if human_input == "approve" else "rejected"}

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "demo"}}

# 执行到 interrupt
graph.invoke({"content": "..."}, config)

# 恢复
graph.invoke(Command(resume="approve"), config)
```

### 循环中多次 interrupt

```python
def confirm(state: State):
    human_input = interrupt({
        "question": f"第 {state['step']} 步完成，是否继续？",
        "options": ["continue", "stop"],
    })
    return {"done": human_input == "stop"}

def route_loop(state: State):
    if state["done"]:
        return END
    return "process_step"

builder.add_conditional_edges("confirm", route_loop)
```

### 敏感操作前审批

```python
def plan_action(state: State):
    if is_sensitive(state["query"]):
        return {"action": "send_email"}
    return {"action": "none"}

def request_approval(state: State):
    if state["action"] == "none":
        return {}  # 不触发 interrupt
    human_input = interrupt({
        "question": f"请求执行: {state['action']}",
        "options": ["approve", "deny"],
    })
    return {"approved": human_input == "approve"}
```

## 常见问题

**Q: interrupt() 会抛出异常吗？**

不会。`interrupt()` 是 LangGraph 内部的控制流机制，它会优雅地暂停图执行。`graph.invoke()` 正常返回，不会抛异常。中断信息通过 `graph.get_state(config).interrupts` 获取。

**Q: 同一个节点里可以有多个 interrupt() 吗？**

可以。但每执行到一个 `interrupt()` 就会暂停一次，需要分别用 `Command(resume=...)` 恢复。

**Q: Command(update=...) 修改的 state 会触发节点重新执行吗？**

不会。`update` 直接修改 state，然后从当前节点继续执行。不会从头重新跑整个图。

**Q: 实际 Web 项目中怎么用 HITL？**

```
POST /start  → graph.invoke(state, config) → 返回 thread_id + interrupt payload
POST /resume → graph.invoke(Command(resume=...), config) → 返回结果或下一个 interrupt
```

前端收到 interrupt payload 后展示 UI，用户操作后调用 /resume。

**Q: InMemorySaver 和 SqliteSaver 的区别？**

| | InMemorySaver | SqliteSaver |
|---|---|---|
| 存储位置 | 内存 | SQLite 文件 |
| 进程重启 | 丢失 | 保留 |
| 适用场景 | 演示、测试 | 生产环境 |
| 用法 | `InMemorySaver()` | `SqliteSaver.from_conn_string("db.sqlite")` |

## 最佳实践

1. **所有 HITL 图都必须配 checkpointer**，否则中断后无法恢复
2. **thread_id 是恢复的唯一标识**，每次启动用新的，恢复用旧的
3. **interrupt payload 要包含足够信息**，让前端/用户知道在等什么、有哪些选项
4. **敏感操作前主动 interrupt**，不要等出错了再补救
5. **非敏感操作不要滥用 interrupt**，否则用户体验差

## 下一步学习

- **18_subgraphs**：把 HITL 审批节点封装成子图，复用到多个工作流
