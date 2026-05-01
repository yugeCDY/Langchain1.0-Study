# 多 Agent 协作 (Multi-Agent Collaboration)

## 快速开始

运行示例：

```bash
python phase3_advanced/16_multi_agent/main.py
```

## 核心概念

### 什么是多 Agent 系统？

单 Agent = 一个 LLM + 若干工具，所有任务自己处理。

多 Agent = 多个专门的 Agent，每个负责一个领域，通过协调机制分工协作。

**什么时候需要多 Agent？**

| 场景 | 单 Agent | 多 Agent |
|------|---------|---------|
| 任务类型单一 | 够用 | 不需要 |
| 需要代码+写作+搜索 | 容易混淆 | 每个 Agent 专精一项 |
| 需要审核/校验 | 自己检查自己，容易漏 | 专门放一个 Review Agent |
| 多轮复杂流程 | prompt 越来越长 | 拆成多个节点，各自简洁 |

### 两种核心架构

**1. Supervisor 模式（中心化）**

```
用户 -> Supervisor -> 判断类型 -> Agent A / Agent B / Agent C
                        ^
                        |
              收集结果后汇总返回
```

- 一个中央协调者（Supervisor）接收所有请求
- Supervisor 决定分配给哪个子 Agent
- 子 Agent 执行后结果返回给 Supervisor 汇总
- 适合：任务分类明确、需要统一入口的场景

**2. Handoff 网络模式（去中心化）**

```
用户 -> Agent A -> 判断需要 B -> Agent B -> 判断需要 C -> Agent C -> 返回
```

- 没有中央协调者
- 每个 Agent 执行完后自己决定下一步交给谁
- 适合：工单流转、客服转接、流水线处理

## 关键 API

### create_agent

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,          # 聊天模型
    tools=[tool1, tool2], # 工具列表
)

# 调用
result = agent.invoke({
    "messages": [{"role": "user", "content": "..."}]
})
```

**内部机制**（不用你写，但要知道）：

```
用户输入 -> LLM 思考 -> 想调用工具？
    ├── 是 -> 执行工具 -> 观察结果 -> 回到 LLM 思考
    └── 否 -> 直接返回答案 -> END
```

`create_agent` 自动帮你建了上面这个循环图。

### 把 Agent 嵌入 StateGraph

```python
from langgraph.graph import StateGraph, START, END

# Agent 本身就是一张图，可以直接当作 StateGraph 的一个节点
builder = StateGraph(State)
builder.add_node("research_agent", research_agent)  # research_agent 是 create_agent 创建的
builder.add_node("write_agent", write_agent)
builder.add_edge(START, "research_agent")
builder.add_edge("research_agent", "write_agent")
builder.add_edge("write_agent", END)
```

## 工作流程

### Supervisor 模式的工作流程

```
1. 用户输入 -> Supervisor 节点
2. Supervisor 分析意图 -> 决定调用哪个 Agent
3. 条件边路由到对应 Agent
4. Agent 执行（内部可能有多轮工具调用）
5. Agent 返回结果 -> Supervisor 汇总
6. Supervisor 返回最终答案给用户
```

### Handoff 模式的工作流程

```
1. 用户输入 -> 入口 Agent
2. 入口 Agent 处理后更新 state["next_agent"]
3. 条件边读取 next_agent，跳转到下一个 Agent
4. 重复直到某个 Agent 把 next_agent 设为 END
```

## 关键代码片段

### 定义带工具的 Agent

```python
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """搜索网页"""
    return "搜索结果..."

agent = create_agent(model=model, tools=[search])
```

### Supervisor 路由函数

```python
# Supervisor 节点：决定走哪个分支
def supervisor(state: State):
    query = state["messages"][-1]["content"]
    if "天气" in query:
        return {"current_agent": "weather"}
    return {"current_agent": "general"}

# 路由函数：根据 state 中的标记跳转
def route_by_agent(state: State):
    return state["current_agent"]  # "weather" 或 "general"

builder.add_conditional_edges(
    "supervisor",
    route_by_agent,
    {"weather": "weather_agent", "general": "general_agent"}
)
```

### Handoff 接力

```python
def agent_a(state: State):
    # 处理完后决定下一步
    if needs_b:
        return {"next_agent": "agent_b", "notes": ["A 处理完成"]}
    return {"next_agent": END, "notes": ["A 直接完成"]}

def route_handoff(state: State):
    return state["next_agent"]  # 可以是节点名或 END

builder.add_conditional_edges("agent_a", route_handoff)
```

### 消息历史的 reducer

```python
from typing import Annotated
import operator

class State(TypedDict):
    # ★ 多个 Agent 依次对话时，messages 需要追加而不是覆盖
    messages: Annotated[list, operator.add]
```

不用 `operator.add` 的话，后一个 Agent 的回复会把前一个的覆盖掉。

## 常见问题

**Q: create_agent 和手动 StateGraph 有什么区别？**

| | create_agent | 手动 StateGraph |
|---|---|---|
| 用途 | 快速搭建单 Agent | 灵活控制整个流程 |
| 代码量 | 1 行 | 多行 |
| 可控性 | 低（内部封装） | 高（每个节点自己定义） |
| 适用场景 | 简单 Agent | 多 Agent 协作、复杂路由 |

**Q: 一个 Agent 节点里可以调用 create_agent 吗？**

可以。`create_agent` 返回的是一个图（Runnable），你可以把它整体注册为一个节点：

```python
research_agent = create_agent(model, [search_tool])
builder.add_node("research", research_agent)
```

**Q: 多 Agent 之间如何共享上下文？**

通过 State 中的共享字段传递，常见做法：
- `messages`: 对话历史（用 reducer 追加）
- `notes`: 各 Agent 收集的信息（用 reducer 追加）
- `query`: 用户原始问题（只读）
- `final_answer`: 最终答案（覆盖更新）

**Q: 为什么 messages 一定要用 reducer？**

因为多个 Agent 节点都会向 messages 里追加内容。没有 reducer 时，后执行的 Agent 会覆盖前一个 Agent 的 messages，导致对话历史丢失。

## 最佳实践

1. **Agent 要专不要杂**：每个 Agent 只负责一个明确任务，prompt 中写清楚角色边界
2. **Supervisor 判断逻辑要可靠**：简单的关键词匹配容易出错，实际项目建议用 LLM 做意图分类
3. **共享状态设计要清晰**：哪些字段是"累积型"（用 reducer）、哪些是"覆盖型"，一开始就定义好
4. **用工厂函数创建相似 Agent**：避免复制粘贴节点函数
5. **先单 Agent 跑通，再组合**：确保每个子 Agent 独立工作正常，再连到多 Agent 图中

## 下一步学习

- **17_human_in_the_loop**：人工审批、打断、编辑（LangGraph 独有核心能力）
- 尝试把本模块的"硬编码 Agent"替换成真实的 `create_agent` + LLM + 工具
