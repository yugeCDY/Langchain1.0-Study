# 子图嵌套 (Subgraphs)

## 快速开始

运行示例：

```bash
python phase3_advanced/18_subgraphs/main.py
```

## 核心概念

### 什么是子图？

子图 = 一个独立的 StateGraph，编译后可以嵌入到更大的图中作为一个节点。

**为什么需要子图？**

| 场景 | 不用子图 | 用子图 |
|------|---------|--------|
| 一个流程在多个地方复用 | 复制粘贴节点代码 | 编译一次，多处嵌入 |
| 复杂流程节点太多 | 一张大图画满节点 | 拆成子图，主图清晰 |
| 团队协作 | 所有人改同一个文件 | 每人维护自己的子图 |
| 模块化管理 | 节点和边混在一起 | 功能模块用子图封装 |

### 两种嵌入模式

**模式一：直接嵌入（父子共享 State）**

```python
# 子图
subgraph = StateGraph(State)
subgraph.add_node("a", node_a)
subgraph.add_edge(START, "a")
subgraph = subgraph.compile()

# 父图直接嵌入
builder = StateGraph(State)
builder.add_node("process", subgraph)  # ← compile 后的子图直接传入
builder.add_edge(START, "process")
```

父子 State 完全一样，子图的 return 自动合并到父图 State。

**模式二：包装器模式（父子 State 不同）**

```python
# 子图有自己的 State
class SubState(TypedDict):
    input_text: str
    output_text: str

subgraph = StateGraph(SubState)
subgraph.add_node("process", sub_process)
subgraph = subgraph.compile()

# 父图 State 不同
class ParentState(TypedDict):
    user_request: str
    result: str

# 包装函数做状态转换
def wrapper(state: ParentState):
    # 父 State -> 子图输入
    sub_result = subgraph.invoke({
        "input_text": state["user_request"],
        "output_text": "",
    })
    # 子图输出 -> 父 State
    return {"result": sub_result["output_text"]}

builder = StateGraph(ParentState)
builder.add_node("process", wrapper)
```

## 关键 API

### add_node("name", subgraph)

```python
from langgraph.graph import StateGraph

# 子图编译
subgraph_builder = StateGraph(State)
subgraph_builder.add_node("a", node_a)
subgraph_builder.add_edge(START, "a")
subgraph = subgraph_builder.compile()

# 父图嵌入
builder = StateGraph(State)
builder.add_node("process", subgraph)  # ← 直接传入编译后的子图
```

### 包装器模式

```python
def wrapper(state: ParentState):
    # 提取 -> 调用 -> 转换
    sub_input = {"field": state["parent_field"]}
    sub_output = subgraph.invoke(sub_input)
    return {"parent_field": sub_output["field"]}

builder.add_node("process", wrapper)
```

### Interrupt 传播

```python
# 子图内部
def sub_review(state):
    human_input = interrupt({"question": "审批？", "options": ["pass", "fail"]})
    return {"status": human_input}

# 父图不需要特殊处理
# 子图的 interrupt 会自动冒泡到父图
# 父图 Command(resume=...) 恢复后，子图自动继续
```

### 查看子图状态

```python
# 获取父图状态（包含子图）
state = graph.get_state(config, subgraphs=True)
```

## 工作流程

### 直接嵌入模式

```
1. 定义子图的 State、节点、边
2. 子图.compile() 得到编译后的 Runnable
3. 父图 add_node("name", subgraph) 嵌入
4. 父图 compile()
5. 调用父图，子图作为普通节点执行
```

### 包装器模式

```
1. 定义子图（独立 State）
2. 子图.compile()
3. 父图写包装函数：提取字段 -> 调用子图 -> 转换回父字段
4. 父图 add_node("name", wrapper)
5. 调用父图
```

## 关键代码片段

### 最简单的子图

```python
class State(TypedDict):
    text: str      # 输入字段：父图传入，子图读取
    result: str    # 输出字段：子图和父图依次写入（后覆盖前）

# 子图第一个节点：读取 text，写入 result（转大写）
def sub_uppercase(state: State):
    return {"result": state["text"].upper()}

# 子图第二个节点：读取 result（已大写），再次写入 result（加前缀）
# 同一字段多次写入时，后写的覆盖先写的（last-write-wins）
def sub_add_prefix(state: State):
    return {"result": f"[PROCESSED] {state['result']}"}

subgraph = StateGraph(State)
subgraph.add_node("uppercase", sub_uppercase)
subgraph.add_node("add_prefix", sub_add_prefix)
subgraph.add_edge(START, "uppercase")
subgraph.add_edge("uppercase", "add_prefix")
subgraph.add_edge("add_prefix", END)
sub = subgraph.compile()

# 父图节点：读取子图处理完的 result，继续追加标记
def parent_append(state: State):
    return {"result": f"{state['result']} <<父图追加>>"}

builder = StateGraph(State)
builder.add_node("process", sub)           # ← compile 后的子图直接嵌入
builder.add_node("finalize", parent_append)
builder.add_edge(START, "process")
builder.add_edge("process", "finalize")
builder.add_edge("finalize", END)
```

**子图内部的数据流：**

```
输入: text = "hello world", result = ""

子图 uppercase 节点:
  读取 state["text"] = "hello world"
  返回 {"result": "HELLO WORLD"}     ← 写入 result

子图 add_prefix 节点:
  读取 state["result"] = "HELLO WORLD"
  返回 {"result": "[PROCESSED] HELLO WORLD"}  ← 覆盖 result

父图 finalize 节点:
  读取 state["result"] = "[PROCESSED] HELLO WORLD"
  返回 {"result": "[PROCESSED] HELLO WORLD <<父图追加>>"}  ← 再次覆盖

输出: result = "[PROCESSED] HELLO WORLD <<父图追加>>"
```

关键点：**父子共享 State 时，子图中的节点和父图中的节点操作的是同一张状态表**。同一字段被多次写入时，后写的覆盖先写的（last-write-wins）。这和普通函数里的变量赋值一样自然。

### 多层嵌套

```python
# 孙图
grandchild = StateGraph(State)
grandchild.add_node("g", grandchild_node)
gc = grandchild.compile()

# 子图嵌套孙图
child = StateGraph(State)
child.add_node("deep", gc)
c = child.compile()

# 父图嵌套子图
parent = StateGraph(State)
parent.add_node("pipeline", c)
```

### 子图复用

```python
# 创建子图模板
def make_subgraph(prefix: str):
    def step1(state):
        return {"result": f"[{prefix}] {state['input']}"}
    builder = StateGraph(State)
    builder.add_node("process", step1)
    return builder.compile()

# 复用
builder = StateGraph(State)
builder.add_node("branch_a", make_subgraph("A"))
builder.add_node("branch_b", make_subgraph("B"))
```

## 常见问题

**Q: 子图和父图的 State 可以不同吗？**

可以。如果不同，用包装器模式手动转换。如果相同，直接嵌入更简洁。

**Q: 子图可以修改父图没有的字段吗？**

不可以。子图只能读写自己 State 中声明的字段。如果子图 return 了父图 State 中没有的字段，会被忽略。

**Q: 子图的 interrupt 会影响父图吗？**

会。子图的 interrupt 会冒泡到父图，父图层面用 `Command(resume=...)` 恢复即可。不需要在父图做任何特殊配置。

**Q: 子图需要单独配 checkpointer 吗？**

不需要。父图的 checkpointer 会自动传播给子图。一个 checkpointer 管全部。

**Q: 子图内部可以嵌套子图吗？**

可以。LangGraph 支持任意深度的嵌套。编译顺序：先编译最底层，再逐层向上。

## 最佳实践

1. **先 compile 子图，再 add_node**：`add_node("name", subgraph)` 传入的必须是编译后的对象
2. **子图负责一个独立功能**：比如"文本处理子图"、"审批子图"、"RAG 检索子图"
3. **共享 State 优先**：如果父子 State 可以设计得一致，代码最简洁
4. **包装函数保持薄**：只做"提取-调用-转换"，不要把业务逻辑塞进去
5. **用工厂函数创建相似子图**：避免复制粘贴子图定义

## 下一步学习

- **19_streaming_and_events**：流式输出时如何同时看到子图和父图的事件
- 把 17_human_in_the_loop 的审批流程封装成"审批子图"，复用到多个工作流
