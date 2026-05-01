# 15 - LangGraph Low Level (LangGraph 底层 API)

## 快速开始

```bash
# 1. 确认已安装依赖
pip install -r requirements.txt

# 2. 运行完整示例
cd phase3_advanced/15_langgraph_low_level
python main.py

# 3. 运行测试（无需 API key）
python test.py
```

## 重要提示（LangGraph 1.x）

本模块已按 LangGraph 1.x 官方文档核实语法：

- 基础导入：`from langgraph.graph import StateGraph, START, END`
- 图构建：`builder = StateGraph(State)`
- 添加节点：`builder.add_node("node_name", node_func)`
- 普通边：`builder.add_edge("node_a", "node_b")`
- 条件边：`builder.add_conditional_edges("node_a", route_func, path_map)`
- 编译图：`graph = builder.compile()`
- 调用图：`graph.invoke(input_state)`

LangGraph 是低层编排框架，适合需要显式控制状态、分支、循环、人机协作和长时间运行流程的场景。如果只是普通 Agent，可以优先使用 LangChain 的 `create_agent`。

## 它和 Dify 工作流像吗？

是的，可以把 LangGraph 理解成 **Dify 工作流的代码版**。

Dify 用可视化画布拖节点、连线、配置变量；LangGraph 用 Python 代码定义节点、边和状态。两者解决的是同一类问题：把一个复杂 AI 应用拆成多个步骤，并控制每一步怎么执行、数据怎么传递、什么时候分支、什么时候结束。

对应关系：

| Dify 工作流 | LangGraph |
|------------|-----------|
| User Input 节点 | `graph.invoke(input_state)` 传入初始状态 |
| 节点 | Python 函数，也就是 `add_node()` 添加的 node |
| 变量 | `State`，在节点之间传递的数据 |
| 连线 | `add_edge()` |
| 条件分支 | `add_conditional_edges()` |
| 工具节点 | 普通节点里调用工具、API、数据库或检索器 |
| Output / Answer 节点 | 最后一个输出节点，或连接到 `END` 前的节点 |

例如一个 Dify 风格流程：

```text
用户输入
  -> 问题分类器
  -> 概念解释分支 / 排错分支
  -> 输出节点
```

用 LangGraph 表达就是：

```text
START -> classify_input -> explain_node/debug_node -> output_node -> END
```

也可以画成简单 ASCII 流程图：

```text
+-------+
| START |
+-------+
    |
    v
+----------------+
| classify_input |
+----------------+
    |
    +-------------------+
    |                   |
    v                   v
+--------------+   +------------+
| explain_node |   | debug_node |
+--------------+   +------------+
    |                   |
    +---------+---------+
              |
              v
      +---------------+
      |  output_node  |
      +---------------+
              |
              v
          +------+
          | END  |
          +------+
```

什么时候更适合用 LangGraph：

- 你想把工作流写进代码仓库，方便测试和版本管理
- 分支、循环、状态更新比较复杂
- 想和自己的 Python 代码、数据库、工具链深度集成
- 后续要做多 Agent、人工审核、持久化恢复等复杂流程

什么时候 Dify 更合适：

- 想快速搭建和试验
- 更喜欢可视化编排
- 团队里有非工程成员一起调整流程

## 核心概念

## 先看一段最小语法

先不要管复杂 Agent，LangGraph 最小写法就是下面 8 步：

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    answer: str


def answer_node(state: State):
    return {"answer": f"你问的是：{state['question']}"}


builder = StateGraph(State)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile()
result = graph.invoke({"question": "什么是 LangGraph？", "answer": ""})
```

逐句解释：

| 代码 | 意思 |
|------|------|
| `class State(TypedDict)` | 定义工作流里有哪些变量 |
| `question: str` | 有一个字符串变量叫 `question` |
| `answer: str` | 有一个字符串变量叫 `answer` |
| `def answer_node(state)` | 定义一个节点，节点会收到当前所有变量 |
| `return {"answer": ...}` | 节点返回要更新的变量 |
| `StateGraph(State)` | 创建一个图，并告诉它变量结构 |
| `add_node("answer", answer_node)` | 把 Python 函数注册成图里的节点 |
| `add_edge(START, "answer")` | 设置入口：从 START 走到 answer 节点 |
| `add_edge("answer", END)` | 设置出口：answer 执行后结束 |
| `compile()` | 把图编译成可运行对象 |
| `invoke({...})` | 运行工作流，并传入初始变量 |

注意：

```python
result = graph.invoke({...})
```

这里的 `result` 不是“最后一个节点单独返回的结果”，而是**整个图执行完成后的最终完整 state**。

也就是说，如果你的 `State` 里定义了：

```python
class State(TypedDict):
    question: str
    answer: str
```

那么 `result` 里通常会同时包含：

```python
{
    "question": "...",
    "answer": "..."
}
```

如果图里有更多字段，例如 `plan`、`practice`、`final_answer`，那么 `result` 也会把这些字段一起带回来。

可以把这段代码想成 Dify 里的：

```text
User Input -> Answer Node -> Output
```

### 1. State：工作流变量

State 是图中所有节点共享的数据结构。它可以用 `TypedDict`、Pydantic 模型或 dataclass 定义。

```python
from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    answer: str
```

这表示这个工作流有两个变量：

- `question`：用户问题
- `answer`：最终答案

为什么要写 `TypedDict`？

因为 LangGraph 需要知道 state 里有哪些字段。它就像 Dify 里提前定义变量名，后面的节点才能引用这些变量。

节点读取当前 state，并返回要更新的字段：

```python
def answer_node(state: State):
    return {"answer": f"问题是: {state['question']}"}
```

注意：节点不需要返回完整 state，只需要返回自己要更新的字段。

```python
return {"answer": "..."}
```

这句意思是：只更新 `answer`，其他字段保持不变。

### 2. Nodes：工作流节点

节点是普通 Python 函数。每个节点负责一个明确步骤，例如分类、检索、调用工具、生成答案、评估结果。

```python
builder.add_node("answer", answer_node)
```

这句有两个参数：

```python
builder.add_node("answer", answer_node)
                 # 节点名   # 节点函数
```

- `"answer"`：图里的节点名字，后面连线会用到
- `answer_node`：真正执行逻辑的 Python 函数

节点的设计建议：

- 只做一件事
- 输入来自 state
- 输出为 dict，表示 state 更新
- 不在节点里写复杂路由逻辑，路由交给条件边

### 3. Edges：工作流连线

边决定下一个节点。

普通边适合固定流程：

```python
builder.add_edge(START, "classify")
builder.add_edge("classify", "answer")
builder.add_edge("answer", END)
```

这三句表示：

```text
START -> classify -> answer -> END
```

`START` 和 `END` 不是你自己写的函数，而是 LangGraph 提供的虚拟节点。

条件边适合运行时分支：

```python
def route(state: State):
    if state["category"] == "debug":
        return "debug"
    return "concept"


builder.add_conditional_edges(
    "classify",
    route,
    {
        "debug": "debug_answer",
        "concept": "concept_answer",
    },
)
```

这段表示：

```text
classify
  -> debug_answer   如果 route 返回 "debug"
  -> concept_answer 如果 route 返回 "concept"
```

`route` 函数只负责决定走哪条路：

```python
def route(state: State):
    if state["category"] == "debug":
        return "debug"
    return "concept"
```

它返回的 `"debug"` 和 `"concept"` 会去映射表里查：

```python
{
    "debug": "debug_answer",
    "concept": "concept_answer",
}
```

### 4. START 和 END：入口和出口

`START` 是虚拟入口节点，用来指定图从哪里开始。

`END` 是虚拟终止节点，用来表示流程结束。

```python
from langgraph.graph import START, END

builder.add_edge(START, "first_node")
builder.add_edge("last_node", END)
```

### 5. Reducer：变量是覆盖还是追加

默认情况下，同一个字段的新值会覆盖旧值。如果希望列表字段不断追加，可以使用 reducer。

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict


class State(TypedDict):
    steps: Annotated[list[str], operator.add]
```

当多个节点返回 `{"steps": ["xxx"]}` 时，LangGraph 会把列表拼接起来，而不是覆盖。

不使用 reducer：

```python
return {"steps": ["第一步"]}
return {"steps": ["第二步"]}
# 最后可能只剩 ["第二步"]
```

使用 reducer：

```python
steps: Annotated[list[str], operator.add]

return {"steps": ["第一步"]}
return {"steps": ["第二步"]}
# 最后是 ["第一步", "第二步"]
```

所以 reducer 很像 Dify 里的“数组变量追加”。

## 常用语法速查

### 创建图

```python
builder = StateGraph(State)
```

意思：创建一个工作流图，变量结构由 `State` 决定。

### 添加节点

```python
builder.add_node("classify", classify_node)
```

意思：添加一个名叫 `classify` 的节点，它执行 `classify_node` 函数。

### 添加固定连线

```python
builder.add_edge("classify", "answer")
```

意思：`classify` 执行完后，固定进入 `answer`。

### 添加条件连线

```python
builder.add_conditional_edges(
    "classify",
    route_func,
    {"a": "node_a", "b": "node_b"},
)
```

意思：

- `classify` 执行完后调用 `route_func(state)`
- 如果返回 `"a"`，去 `node_a`
- 如果返回 `"b"`，去 `node_b`

### 编译图

```python
graph = builder.compile()
```

意思：把定义好的节点和边变成可运行工作流。

### 运行图

```python
result = graph.invoke({"question": "你好", "answer": ""})
```

意思：传入初始变量，运行整个工作流，拿到最终 state。

## 本模块示例

### 示例 1：像 Dify 一样编排工作流

演示一个最直观的 Dify 风格工作流：

```text
START -> classify_input -> explain_node/debug_node -> output_node -> END
```

ASCII 图：

```text
+-------+
| START |
+-------+
    |
    v
+----------------+
| classify_input |
+----------------+
    |
    +-------------------+
    |                   |
    v                   v
+--------------+   +------------+
| explain_node |   | debug_node |
+--------------+   +------------+
    |                   |
    +---------+---------+
              |
              v
      +---------------+
      |  output_node  |
      +---------------+
              |
              v
          +------+
          | END  |
          +------+
```

学习重点：

- Dify 节点和 LangGraph node 的对应关系
- Dify 变量和 LangGraph State 的对应关系
- Dify 条件分支和 `add_conditional_edges()` 的对应关系
- Output/Answer 节点如何对应最后的输出节点

注意：这个示例没有接入大模型，分支判断来自普通 Python 规则。

```python
def classify_input(state: State):
    if "报错" in state["user_input"] or "错误" in state["user_input"]:
        return {"intent": "debug"}
    return {"intent": "explain"}
```

也就是说：

```text
用户输入包含“报错”或“错误” -> intent = debug
否则                         -> intent = explain
```

LangGraph 本身不负责“智能判断”，它负责根据 state 和路由函数组织流程。判断逻辑可以是：

- 普通 Python `if` 规则
- 关键词匹配
- 数据库查询结果
- 工具调用结果
- 大模型分类结果

如果接入大模型，只需要把 `classify_input` 节点改成“调用 LLM 判断问题类型”，后面的 `add_conditional_edges()` 路由逻辑仍然可以保持不变。

### 示例 2：多节点顺序执行

演示固定流程：

```text
START -> make_plan -> write_draft -> polish_answer -> END
```

ASCII 图：

```text
+-------+
| START |
+-------+
    |
    v
+-----------+
| make_plan |
+-----------+
    |
    v
+-------------+
| write_draft |
+-------------+
    |
    v
+---------------+
| polish_answer |
+---------------+
    |
    v
+------+
| END  |
+------+
```

适合确定性任务，例如数据清洗、报告生成、审批流程。

### 示例 3：条件路由

根据问题类型选择不同回答路径：

```text
classify -> answer_concept
         -> answer_debug
```

ASCII 图：

```text
+-------+
| START |
+-------+
    |
    v
+----------+
| classify |
+----------+
    |
    +---------------------+
    |                     |
    v                     v
+----------------+   +--------------+
| answer_concept |   | answer_debug |
+----------------+   +--------------+
    |                     |
    v                     v
+------+              +------+
| END  |              | END  |
+------+              +------+
```

适合分类、风控、客服分流、工具选择。

### 示例 4：使用 reducer 追加状态

用 `Annotated[list[str], operator.add]` 记录执行轨迹。

先说明一件很重要的事：

前面几个例子里，大多数节点写的是**不同字段**，例如：

- 一个节点写 `plan`
- 下一个节点写 `practice`
- 最后一个节点写 `final_answer`

这种情况下，字段之间不会冲突，当然也就**不需要 reducer**。

所以如果你是第一次接触 LangGraph，更自然的理解顺序应该是：

1. 先学“不同节点写不同字段”
2. 再学“多个节点都写同一个字段时会发生什么”
3. 最后再学 reducer 为什么有用

这个示例是**故意制造“多个节点都写同一个字段”**，专门演示 LangGraph 的默认合并规则，不是说你平时一定要这么写。

最关键的语法就是这一行：

```python
steps: Annotated[list[str], operator.add]
```

拆开看：

- `list[str]`：`steps` 是一个字符串列表
- `Annotated[..., operator.add]`：告诉 LangGraph，这个字段多次更新时不要覆盖，而是用 `operator.add` 合并
- 对列表来说，`operator.add` 的效果就是“拼接列表”

对应效果：

```python
def collect_input(state: State):
    return {"steps": ["收集输入"]}


def process_data(state: State):
    return {"steps": ["处理数据"]}
```

如果没有 reducer，后一次更新可能把前一次覆盖掉；有了 reducer，最终会变成：

```python
["收集输入", "处理数据", "生成总结"]
```

为什么会覆盖？

因为 LangGraph 的节点不是直接修改原对象，而是：

1. 节点收到当前 `state`
2. 节点返回一个更新字典，例如 `{"steps": ["收集输入"]}`
3. LangGraph 把这个更新字典合并回总 `state`

如果两个节点都更新 `steps`，默认规则就是“后者覆盖前者”。

也就是说，问题不在于节点不是顺序执行，而在于：

```text
顺序执行 + 同名字段默认覆盖
```

reducer 的作用就是把这个默认规则改掉。

更真实的使用场景通常不是 `steps` 这种教学字段，而是这些：

- `messages`：多轮对话里不断追加消息历史
- `notes`：多个节点都在收集资料，最后汇总到一个列表
- `logs` / `trace`：记录流程走过哪些步骤
- `errors` / `warnings`：不同节点发现的问题统一收集

所以 reducer 的真实意义是：

```text
多个节点共同积累同一类信息
```

适合保存：

- 消息历史
- 工具调用记录
- 工作流步骤
- 中间观察结果

### 示例 5：循环和终止条件

演示生成、评估、重试：

```text
draft -> evaluate -> draft
                 -> END
```

ASCII 图：

```text
+-------+
| START |
+-------+
    |
    v
+-------+
| draft |
+-------+
    |
    v
+----------+
| evaluate |
+----------+
    |
    +------------------+
    |                  |
    v                  v
+--------+         +------+
| revise |         | END  |
+--------+         +------+
    |
    +-------> back to draft
```

注意：循环必须有明确终止条件。`recursion_limit` 是保护网，不应替代业务逻辑。

### 示例 6：小型研究工作流

综合使用状态、条件边和 reducer，构建一个简化研究流程：

```text
classify_task -> collect_facts -> compare_options -> write_report
                              -> explain_topic    -> write_report
```

ASCII 图：

```text
+-------+
| START |
+-------+
    |
    v
+---------------+
| classify_task |
+---------------+
    |
    v
+---------------+
| collect_facts |
+---------------+
    |
    +-----------------------+
    |                       |
    v                       v
+-----------------+   +---------------+
| compare_options |   | explain_topic |
+-----------------+   +---------------+
    |                       |
    +-----------+-----------+
                |
                v
        +--------------+
        | write_report |
        +--------------+
                |
                v
            +------+
            | END  |
            +------+
```

## 常见问题

### Q1: LangGraph 和 LangChain Agent 有什么区别？

LangChain Agent 是更高层抽象，适合快速创建常见工具调用 Agent。

LangGraph 是更底层的编排框架，适合你需要显式控制状态、分支、循环、持久化、人机协作或多 Agent 流程时使用。

### Q2: 节点必须调用 LLM 吗？

不必须。节点就是普通 Python 函数，可以是规则逻辑、数据库查询、工具调用、LLM 调用或任意业务代码。

本模块故意不调用 LLM，是为了先把 LangGraph 的图结构和状态更新学清楚。

### Q3: 条件边返回什么？

可以直接返回目标节点名，也可以返回映射表里的 key。

推荐在教学和生产代码中使用映射表：

```python
builder.add_conditional_edges(
    "classify",
    route,
    {"debug": "debug_answer", "concept": "concept_answer"},
)
```

这样路由函数返回值可以更语义化，不必和节点名强绑定。

### Q4: 什么时候需要 reducer？

当多个节点会更新同一个字段，并且你希望合并而不是覆盖时，就需要 reducer。

常见例子是消息列表：

```python
messages: Annotated[list, operator.add]
```

### Q5: 循环为什么需要 `recursion_limit`？

如果业务终止条件写错，图可能一直循环。`recursion_limit` 可以限制最大执行步数，避免无限运行。

但正确做法仍然是在条件边里写清楚终止条件。

## 最佳实践

1. **先画流程，再写图**
   先写出节点和边，再实现每个节点函数。

2. **节点职责单一**
   一个节点只做一个明确动作，例如 classify、retrieve、answer、evaluate。

3. **路由函数只做决策**
   条件边的 route 函数只决定下一步，不混入复杂业务处理。

4. **循环必须有终止条件**
   常见终止条件包括评分达标、次数上限、状态完成、用户确认。

5. **状态字段保持清晰**
   不要把所有东西塞进一个 dict 字符串里。明确字段会让调试容易很多。

6. **需要追加历史时使用 reducer**
   消息、事件、步骤日志都适合 reducer。

## 进一步学习

下一章建议学习：

- `16_multi_agent`：多个专业 Agent 的协作
- supervisor 模式：一个协调者分配任务
- handoff 模式：Agent 之间移交控制权
- 图持久化：用 checkpointer 保存长流程状态

## 参考资料

- LangGraph 概览: https://docs.langchain.com/oss/python/langgraph
- Graph API 概览: https://docs.langchain.com/oss/python/langgraph/graph-api
- 使用 Graph API: https://docs.langchain.com/oss/python/langgraph/use-graph-api
- Recursion Limit: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT

## 核心要点总结

1. **LangGraph 很像 Dify 工作流的代码版**
2. **StateGraph = 状态 + 节点 + 边**
3. **节点负责工作，边负责下一步**
4. **条件边让流程动态变化**
5. **reducer 决定状态字段如何合并**
6. **循环必须有明确终止条件**
7. **LangGraph 适合复杂可控工作流**
