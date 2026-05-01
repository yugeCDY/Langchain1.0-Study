"""
简单测试：验证 LangGraph 底层 API 示例（无需 API key）
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


print("=" * 70)
print("测试：LangGraph Low Level")
print("=" * 70)


# ============================================================================
# 测试 1：最小图
# ============================================================================
print("\n--- 测试 1: 最小 StateGraph ---")


class MinimalState(TypedDict):
    text: str
    output: str


def echo_node(state: MinimalState):
    return {"output": f"收到: {state['text']}"}


builder = StateGraph(MinimalState)
builder.add_node("echo", echo_node)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile()

result = graph.invoke({"text": "hello", "output": ""})
assert result["output"] == "收到: hello"
print("[OK] 最小图运行成功")


# ============================================================================
# 测试 2：条件路由
# ============================================================================
print("\n--- 测试 2: 条件路由 ---")


class RouteState(TypedDict):
    value: int
    path: str


def start_node(state: RouteState):
    return {}


def even_node(state: RouteState):
    return {"path": "even"}


def odd_node(state: RouteState):
    return {"path": "odd"}


def route_by_value(state: RouteState):
    if state["value"] % 2 == 0:
        return "even"
    return "odd"


builder = StateGraph(RouteState)
builder.add_node("start", start_node)
builder.add_node("even_node", even_node)
builder.add_node("odd_node", odd_node)
builder.add_edge(START, "start")
builder.add_conditional_edges(
    "start",
    route_by_value,
    {
        "even": "even_node",
        "odd": "odd_node",
    },
)
builder.add_edge("even_node", END)
builder.add_edge("odd_node", END)
graph = builder.compile()

even_result = graph.invoke({"value": 2, "path": ""})
odd_result = graph.invoke({"value": 3, "path": ""})

assert even_result["path"] == "even"
assert odd_result["path"] == "odd"
print("[OK] 条件路由运行成功")


# ============================================================================
# 测试 3：reducer 追加状态
# ============================================================================
print("\n--- 测试 3: reducer 追加状态 ---")


class ReducerState(TypedDict):
    events: Annotated[list[str], operator.add]


def node_a(state: ReducerState):
    return {"events": ["A"]}


def node_b(state: ReducerState):
    return {"events": ["B"]}


builder = StateGraph(ReducerState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)
graph = builder.compile()

result = graph.invoke({"events": []})
assert result["events"] == ["A", "B"]
print("[OK] reducer 追加状态成功")


# ============================================================================
# 测试 4：循环终止
# ============================================================================
print("\n--- 测试 4: 循环终止 ---")


class LoopState(TypedDict):
    count: int
    trace: Annotated[list[str], operator.add]


def increment(state: LoopState):
    next_count = state["count"] + 1
    return {
        "count": next_count,
        "trace": [f"count={next_count}"],
    }


def should_continue(state: LoopState):
    if state["count"] >= 3:
        return "end"
    return "continue"


builder = StateGraph(LoopState)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_conditional_edges(
    "increment",
    should_continue,
    {
        "continue": "increment",
        "end": END,
    },
)
graph = builder.compile()

result = graph.invoke({"count": 0, "trace": []}, {"recursion_limit": 10})
assert result["count"] == 3
assert result["trace"] == ["count=1", "count=2", "count=3"]
print("[OK] 循环和终止条件成功")


# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("LangGraph Low Level 测试完成！")
print("=" * 70)

print("\n已验证:")
print("  [OK] StateGraph / START / END")
print("  [OK] add_node / add_edge")
print("  [OK] add_conditional_edges")
print("  [OK] reducer 状态合并")
print("  [OK] 循环终止条件")

print("\n运行完整示例:")
print("  python main.py")
