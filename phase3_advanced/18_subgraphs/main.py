"""
LangChain 1.0 - 子图嵌套 (Subgraphs)
======================================

本模块重点讲解：
1. 什么是子图，为什么需要它
2. 直接嵌入：父子共享 State，compile 后直接 add_node
3. 包装器模式：父子 State 不同，手动转换
4. 子图中的 interrupt 如何传播到父图
5. 多层嵌套：子图里面还可以有子图

说明：
- 详细知识点和逐句语法解释见 README.md
- main.py 只保留可运行演示和关键结果输出
"""

import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# ============================================================================
# 环境配置
# ============================================================================

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")

if not API_KEY:
    raise ValueError("请先设置 OPENAI_API_KEY")

model = init_chat_model(
    "openai:glm-5.1",
    api_key=API_KEY,
    base_url=BASE_URL,
)

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
from typing import Annotated
import operator


# ============================================================================
# 示例 1：最简单的子图嵌入（父子共享 State）
# ============================================================================
def example_1_subgraph_basic():
    """
    示例1：最简单的子图嵌入

    父图和子图使用同一个 State 定义，子图 compile 后直接传给 add_node。
    """
    print("\n" + "=" * 70)
    print("示例 1：最简单的子图嵌入（父子共享 State）")
    print("=" * 70)

    # 父图和子图共享同一个 State 定义。
    # 这意味着子图中的节点可以直接读写父图的字段，反之亦然。
    class State(TypedDict):
        text: str      # 输入字段：由父图传入，子图读取
        result: str    # 输出字段：子图和父图都会写入（依次覆盖）

    # ★ 子图：负责"文本处理"这一独立功能。
    #   子图有自己的节点和边，compile 后对外只暴露一个入口。
    #
    # 子图内部的数据流：
    #   uppercase 节点读取 text → 写入 result（转大写）
    #   add_prefix 节点读取 result（已大写）→ 再次写入 result（加前缀）
    def sub_uppercase(state: State):
        # 子图第一个节点：将输入 text 转为大写，写入 result
        return {"result": state["text"].upper()}

    def sub_add_prefix(state: State):
        # 子图第二个节点：读取上一个节点写入的 result，在前面加标记
        # 同一字段多次写入时，后写的覆盖先写的（last-write-wins）
        return {"result": f"[PROCESSED] {state['result']}"}

    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node("uppercase", sub_uppercase)
    subgraph_builder.add_node("add_prefix", sub_add_prefix)
    subgraph_builder.add_edge(START, "uppercase")
    subgraph_builder.add_edge("uppercase", "add_prefix")
    subgraph_builder.add_edge("add_prefix", END)

    # compile() 返回的是一个 Runnable，可以直接当作节点函数使用。
    subgraph = subgraph_builder.compile()

    # ★ 父图：把编译后的子图当作一个普通节点注册进来。
    #   add_node("process", subgraph) 的意思是：
    #   当执行到 "process" 节点时，运行整个子图（uppercase → add_prefix）。
    def parent_append(state: State):
        # 父图节点：读取子图处理完的 result，追加自己的标记
        # 子图结束后，result 的值是 "[PROCESSED] HELLO WORLD"
        # 父图继续覆盖 result，最终变成 "[PROCESSED] HELLO WORLD <<父图追加>>"
        return {"result": f"{state['result']} <<父图追加>>"}

    builder = StateGraph(State)
    builder.add_node("process", subgraph)   # ← 子图直接嵌入
    builder.add_node("finalize", parent_append)
    builder.add_edge(START, "process")
    builder.add_edge("process", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile()
    result = graph.invoke({"text": "hello world", "result": ""})

    print(f"输入: hello world")
    print(f"输出: {result['result']}")
    print("\n执行流程:")
    print("  START -> process(子图: uppercase -> add_prefix) -> finalize -> END")

    print("\n关键点:")
    print("  - 子图 compile() 后可以直接 add_node()")
    print("  - 父子共享 State，子图的 return 直接合并到父图 State")
    print("  - 子图内部结构对外部是黑盒")


# ============================================================================
# 示例 2：子图作为可复用组件
# ============================================================================
def example_2_reusable_subgraph():
    """
    示例2：子图作为可复用组件

    同一个子图被父图的多个节点复用，每个节点传入不同的初始数据。
    """
    print("\n" + "=" * 70)
    print("示例 2：子图作为可复用组件")
    print("=" * 70)

    # 父图 State：同时包含两路数据。
    class State(TypedDict):
        query_a: str
        query_b: str
        result_a: str
        result_b: str

    # ★ 子图 A：处理 query_a / result_a。
    #   子图 State 只声明自己关心的字段，其他字段父图传入但子图无视。
    class SubStateA(TypedDict):
        query_a: str
        result_a: str

    def sub_a_step1(state: SubStateA):
        return {"result_a": f"[A处理] {state['query_a']}"}

    def sub_a_step2(state: SubStateA):
        return {"result_a": f"{state['result_a']} [A完成]"}

    subgraph_a = StateGraph(SubStateA)
    subgraph_a.add_node("step1", sub_a_step1)
    subgraph_a.add_node("step2", sub_a_step2)
    subgraph_a.add_edge(START, "step1")
    subgraph_a.add_edge("step1", "step2")
    subgraph_a.add_edge("step2", END)
    compiled_a = subgraph_a.compile()

    # ★ 子图 B：结构和 A 完全一样，只是字段名换成 query_b / result_b。
    #   这就是"复用"的含义：相同的处理逻辑，不同的数据字段。
    class SubStateB(TypedDict):
        query_b: str
        result_b: str

    def sub_b_step1(state: SubStateB):
        return {"result_b": f"[B处理] {state['query_b']}"}

    def sub_b_step2(state: SubStateB):
        return {"result_b": f"{state['result_b']} [B完成]"}

    subgraph_b = StateGraph(SubStateB)
    subgraph_b.add_node("step1", sub_b_step1)
    subgraph_b.add_node("step2", sub_b_step2)
    subgraph_b.add_edge(START, "step1")
    subgraph_b.add_edge("step1", "step2")
    subgraph_b.add_edge("step2", END)
    compiled_b = subgraph_b.compile()

    # ★ 关键：子图只更新自己 State 中声明的字段。
    #   compiled_a 只动 result_a，compiled_b 只动 result_b，互不干扰。

    def parent_combine(state):
        return {
            "result_a": f"{state['result_a']} <<父图合并>>",
            "result_b": f"{state['result_b']} <<父图合并>>",
        }

    builder = StateGraph(State)
    builder.add_node("branch_a", compiled_a)
    builder.add_node("branch_b", compiled_b)
    builder.add_node("combine", parent_combine)

    # 两个子图并行执行（都从 START 出发）。
    builder.add_edge(START, "branch_a")
    builder.add_edge(START, "branch_b")
    builder.add_edge("branch_a", "combine")
    builder.add_edge("branch_b", "combine")
    builder.add_edge("combine", END)

    graph = builder.compile()
    result = graph.invoke({"query_a": "任务A", "query_b": "任务B", "result_a": "", "result_b": ""})

    print(f"query_a: 任务A -> {result['result_a']}")
    print(f"query_b: 任务B -> {result['result_b']}")
    print("\n执行流程:")
    print("  START -> branch_a(子图A) -----")
    print("       -> branch_b(子图B) -----> combine -> END")

    print("\n关键点:")
    print("  - 同一个子图结构可以创建多个实例处理不同数据")
    print("  - 子图只读写自己声明的字段，其他字段保持不动")
    print("  - 多个子图可以并行执行")


# ============================================================================
# 示例 3：包装器模式（父子 State 不同）
# ============================================================================
def example_3_wrapper_pattern():
    """
    示例3：包装器模式

    父图和子图的 State 完全不同，不能直接嵌入。
    写一个包装函数做"父 State -> 子图输入 -> 子图输出 -> 父 State"的转换。
    """
    print("\n" + "=" * 70)
    print("示例 3：包装器模式（父子 State 不同）")
    print("=" * 70)

    # 父图的 State。
    class ParentState(TypedDict):
        user_request: str
        summary: str
        enriched: str

    # 子图的 State（和父图完全不同）。
    class SubState(TypedDict):
        input_text: str
        word_count: int
        output_text: str

    # 子图：专门做"文本统计+摘要"。
    def sub_count_words(state: SubState):
        words = len(state["input_text"].split())
        return {"word_count": words}

    def sub_summarize(state: SubState):
        return {"output_text": f"摘要({state['word_count']}字): {state['input_text'][:10]}..."}

    subgraph_builder = StateGraph(SubState)
    subgraph_builder.add_node("count", sub_count_words)
    subgraph_builder.add_node("summarize", sub_summarize)
    subgraph_builder.add_edge(START, "count")
    subgraph_builder.add_edge("count", "summarize")
    subgraph_builder.add_edge("summarize", END)
    subgraph = subgraph_builder.compile()

    # ★ 包装函数：负责父 State 和子图 State 之间的转换。
    def enrich_with_subgraph(state: ParentState):
        # 1. 从父 State 中提取子图需要的字段。
        subgraph_input = {"input_text": state["user_request"], "word_count": 0, "output_text": ""}

        # 2. 调用子图。
        subgraph_result = subgraph.invoke(subgraph_input)

        # 3. 把子图输出转换回父 State。
        return {
            "enriched": f"{subgraph_result['output_text']} (原文: {state['user_request']})",
        }

    def parent_finalize(state: ParentState):
        return {"summary": f"最终报告: {state['enriched']}"}

    builder = StateGraph(ParentState)
    builder.add_node("enrich", enrich_with_subgraph)
    builder.add_node("finalize", parent_finalize)
    builder.add_edge(START, "enrich")
    builder.add_edge("enrich", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile()
    result = graph.invoke({"user_request": "帮我分析这段长文本的核心观点", "summary": "", "enriched": ""})

    print(f"输入: 帮我分析这段长文本的核心观点")
    print(f"enriched: {result['enriched']}")
    print(f"summary: {result['summary']}")
    print("\n数据流:")
    print("  父图 user_request")
    print("    -> 包装函数提取 input_text")
    print("    -> 子图(count -> summarize)")
    print("    -> 包装函数转换回 enriched")
    print("    -> 父图 finalize")

    print("\n关键点:")
    print("  - 父子 State 不同时，不能直接把子图 add_node")
    print("  - 用包装函数手动做状态转换")
    print("  - 子图.invoke() 内部完整执行，对外部是同步调用")


# ============================================================================
# 示例 4：子图的 interrupt 传播到父图
# ============================================================================
def example_4_subgraph_interrupt():
    """
    示例4：子图的 interrupt 传播到父图

    子图内部用 interrupt() 暂停，中断会冒泡到父图，
    父图用 Command(resume=...) 恢复后，子图继续执行。
    """
    print("\n" + "=" * 70)
    print("示例 4：子图的 interrupt 传播到父图")
    print("=" * 70)

    class State(TypedDict):
        content: str
        reviewed: str
        final: str

    # 子图：包含 interrupt，用于内容审批。
    def sub_generate(state: State):
        return {"content": f"草稿: {state['content']}"}

    def sub_review(state: State):
        human_input = interrupt({
            "question": "子图请求审批：是否通过此内容？",
            "content": state["content"],
            "options": ["pass", "fail"],
        })
        return {"reviewed": "通过" if human_input == "pass" else "不通过"}

    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node("generate", sub_generate)
    subgraph_builder.add_node("review", sub_review)
    subgraph_builder.add_edge(START, "generate")
    subgraph_builder.add_edge("generate", "review")
    subgraph_builder.add_edge("review", END)
    subgraph = subgraph_builder.compile()

    # 父图：把含 interrupt 的子图嵌入。
    def parent_postprocess(state: State):
        return {"final": f"[{state['reviewed']}] {state['content']}"}

    builder = StateGraph(State)
    builder.add_node("draft", subgraph)
    builder.add_node("postprocess", parent_postprocess)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "postprocess")
    builder.add_edge("postprocess", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "subgraph-interrupt"}}

    # 启动图，子图内的 interrupt 会让整个父图暂停。
    print("[1] 启动父图，子图执行到 interrupt 处，父图也暂停...")
    graph.invoke({"content": "关于LangChain的说明", "reviewed": "", "final": ""}, config)

    current = graph.get_state(config)
    print(f"    父图 interrupts: {len(current.interrupts)}")
    print(f"    中断来自子图: {current.interrupts[0].value['question']}")

    # 用户在父图层面恢复，子图自动继续。
    user_input = input("\n请输入 (pass/fail): ").strip().lower()
    while user_input not in ("pass", "fail"):
        user_input = input("请输入 pass 或 fail: ").strip().lower()

    print(f"[2] 用户输入 '{user_input}'，恢复父图（子图自动继续）...")
    result = graph.invoke(Command(resume=user_input), config)
    print(f"    reviewed: {result['reviewed']}")
    print(f"    final: {result['final']}")

    print("\n关键点:")
    print("  - 子图的 interrupt 会冒泡到父图")
    print("  - 父图用 Command(resume) 恢复，子图内部自动继续")
    print("  - 只需要一个 checkpointer，父子自动共享")


# ============================================================================
# 示例 5：多层嵌套子图
# ============================================================================
def example_5_nested_subgraphs():
    """
    示例5：多层嵌套子图

    子图里面还可以有子图，形成多层嵌套。
    LangGraph 支持任意深度的嵌套。
    """
    print("\n" + "=" * 70)
    print("示例 5：多层嵌套子图")
    print("=" * 70)

    class State(TypedDict):
        data: str
        level1_result: str
        level2_result: str
        final: str

    # 最底层子图（孙图）。
    def grandchild_process(state: State):
        return {"level2_result": f"[孙图] 处理 {state['data']}"}

    grandchild = StateGraph(State)
    grandchild.add_node("process", grandchild_process)
    grandchild.add_edge(START, "process")
    grandchild.add_edge("process", END)
    compiled_grandchild = grandchild.compile()

    # 中间层子图（子图），内部嵌套孙图。
    def child_preprocess(state: State):
        return {"level1_result": f"[子图预处理] {state['data']}"}

    child = StateGraph(State)
    child.add_node("preprocess", child_preprocess)
    child.add_node("deep_process", compiled_grandchild)  # ← 孙图嵌套
    child.add_edge(START, "preprocess")
    child.add_edge("preprocess", "deep_process")
    child.add_edge("deep_process", END)
    compiled_child = child.compile()

    # 父图，嵌套子图。
    def parent_wrap(state: State):
        return {"final": f"[父图汇总] {state['level1_result']} | {state['level2_result']}"}

    builder = StateGraph(State)
    builder.add_node("pipeline", compiled_child)
    builder.add_node("wrap", parent_wrap)
    builder.add_edge(START, "pipeline")
    builder.add_edge("pipeline", "wrap")
    builder.add_edge("wrap", END)

    graph = builder.compile()
    result = graph.invoke({"data": "测试数据", "level1_result": "", "level2_result": "", "final": ""})

    print(f"data: 测试数据")
    print(f"level1_result: {result['level1_result']}")
    print(f"level2_result: {result['level2_result']}")
    print(f"final: {result['final']}")
    print("\n嵌套结构:")
    print("  父图 -> pipeline(子图)")
    print("            -> preprocess")
    print("            -> deep_process(孙图)")
    print("                 -> process")
    print("            -> END")
    print("  父图 -> wrap -> END")

    print("\n关键点:")
    print("  - 子图可以嵌套子图，层数不限")
    print("  - 每一层只关心自己的 State 字段")
    print("  - 编译顺序：先编译最底层，再逐层向上")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - 子图嵌套 (Subgraphs)")
    print("=" * 70)

    try:
        example_1_subgraph_basic()
        input("\n按 Enter 继续...")

        example_2_reusable_subgraph()
        input("\n按 Enter 继续...")

        example_3_wrapper_pattern()
        input("\n按 Enter 继续...")

        example_4_subgraph_interrupt()
        input("\n按 Enter 继续...")

        example_5_nested_subgraphs()

        print("\n" + "=" * 70)
        print(" 完成！")
        print("=" * 70)
        print("\n核心要点:")
        print("  1. 子图 compile 后可以直接 add_node() 嵌入父图")
        print("  2. 父子共享 State 时最简洁")
        print("  3. State 不同时用包装函数做转换")
        print("  4. 子图的 interrupt 会冒泡到父图")
        print("  5. 子图可以无限嵌套")
        print("\n下一步:")
        print("  19_streaming_and_events - 学习流式输出和事件过滤")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
