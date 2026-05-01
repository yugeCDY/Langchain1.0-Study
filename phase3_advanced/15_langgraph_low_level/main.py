"""
LangChain 1.0 - LangGraph Low Level (LangGraph 底层 API)
=======================================================

本模块重点讲解：
1. LangGraph 和 Dify 工作流的对应关系
2. StateGraph、START、END 的基础用法
3. 节点、普通边、条件边的工作方式
4. 使用 reducer 合并状态
5. 使用条件路由实现循环和终止

说明：
- 详细知识点和逐句语法解释见 README.md
- main.py 只保留可运行演示和关键结果输出
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict

# StateGraph 是 LangGraph 的核心：用 State + 节点 + 边定义一个可运行流程。
# START / END 是固定的虚拟入口和出口，不需要自己实现函数。
#
# 可以先把 LangGraph 想成：
# 1. 先定义一份共享状态 State
# 2. 再定义多个节点函数
# 3. 每个节点读取 State，并返回"要更新哪些字段"
# 4. LangGraph 按边的定义依次执行节点，并把更新合并回 State
#
# ★ 节点 return 的规则（最重要的概念）：
#
# 节点函数的 return 值是一个 dict，LangGraph 会把它**合并（merge）**回当前 State。
# 关键是：return 的不是"新的完整 State"，而是"我要更新哪些字段"。
#
#   def my_node(state: State):
#       return {"intent": "debug"}          # 只更新 intent，其他字段不动
#
# 合并规则：
#   - 不同节点写不同字段 → 互不影响，各自更新各自的
#   - 多个节点写同一字段 → 默认后写覆盖前写（last-write-wins）
#   - 想要"追加"而不是"覆盖" → 用 reducer：Annotated[list, operator.add]
#
# 这就是示例 1-3 用普通字段（覆盖无妨），
# 示例 4 起要用 reducer（需要追加）的原因。
from langgraph.graph import END, START, StateGraph


def print_title(title: str):
    """打印示例标题。"""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def print_points(*points: str):
    """打印少量关键点。"""
    print("\n关键点:")
    for point in points:
        print(f"  - {point}")


# ============================================================================
# 示例 1：像 Dify 一样编排工作流
# ============================================================================
def example_1_dify_style_workflow():
    """
    示例1：像 Dify 一样编排工作流

    Dify: 输入节点 -> 分类节点 -> 分支节点 -> 输出节点
    LangGraph: START -> classify -> explain/debug -> output -> END
    """
    print_title("示例 1：像 Dify 一样编排工作流")

    # State 定义工作流中会流动的变量，类似 Dify 里的变量面板。
    # 每个节点都能读取这些字段，并返回 dict 更新其中一部分字段。
    #
    # 这里 4 个字段分别代表：
    # - user_input: 用户输入
    # - intent: 分类结果
    # - processed_text: 分支节点处理后的中间结果
    # - output: 最终输出给用户的内容
    class State(TypedDict):
        user_input: str
        intent: str
        processed_text: str
        output: str

    # ★ 节点函数的 return 就是"告诉 LangGraph 要更新哪些字段"。
    #
    # 这个例子里 4 个节点各写不同字段：
    #   classify_input  → 写 intent
    #   explain_node    → 写 processed_text
    #   debug_node      → 写 processed_text
    #   output_node     → 写 output
    #
    # 因为不同节点写不同字段，互不冲突，所以不需要 reducer。
    # LangGraph 会把每个节点的 return dict 合并进 State，未提到的字段保持不变。
    def classify_input(state: State):
        # 只返回 {"intent": ...}，user_input / processed_text / output 都不动。
        if "报错" in state["user_input"] or "错误" in state["user_input"]:
            return {"intent": "debug"}
        return {"intent": "explain"}

    def explain_node(state: State):
        # 只返回 {"processed_text": ...}，其他字段不动。
        return {"processed_text": "概念解释流程：先讲定义，再给最小示例。"}

    def debug_node(state: State):
        return {"processed_text": "排错流程：先复现问题，再检查依赖、版本和参数。"}

    def output_node(state: State):
        # 读取 state['processed_text']（上一个节点写入的），生成最终 output。
        return {"output": f"处理结果: {state['processed_text']}"}

    # 路由函数只决定走哪条分支。
    # 返回值会在 add_conditional_edges 的映射表中查找目标节点。
    #
    # 这里的意思是：
    # - 如果 state["intent"] == "debug"，就走 debug_node
    # - 否则走 explain_node
    def route_by_intent(state: State):
        if state["intent"] == "debug":
            return "debug"
        return "explain"

    # 创建图构建器。此时只是定义流程，还没有真正运行。
    builder = StateGraph(State)

    # add_node("节点名", 函数) 把 Python 函数注册成图中的节点。
    builder.add_node("classify_input", classify_input)
    builder.add_node("explain_node", explain_node)
    builder.add_node("debug_node", debug_node)
    builder.add_node("output_node", output_node)

    # 普通边：START 后固定进入 classify_input。
    builder.add_edge(START, "classify_input")

    # 条件边：classify_input 执行后调用 route_by_intent 决定下一步。
    builder.add_conditional_edges(
        "classify_input",
        route_by_intent,
        {
            "explain": "explain_node",
            "debug": "debug_node",
        },
    )
    builder.add_edge("explain_node", "output_node")
    builder.add_edge("debug_node", "output_node")
    builder.add_edge("output_node", END)

    # compile() 把定义好的图编译成可 invoke 的 Runnable。
    # compile 之前是"流程定义"，compile 之后才是"可执行对象"。
    graph = builder.compile()

    print("流程: START -> classify_input -> explain/debug -> output_node -> END")
    for user_input in ["什么是 LangGraph？", "LangGraph 运行时报错怎么办？"]:
        # invoke 传入初始 state，返回执行完整个图后的最终 state。
        # result 不是"最后一个节点单独的返回值"，而是整个图跑完后的完整状态。
        result = graph.invoke({
            "user_input": user_input,
            "intent": "",
            "processed_text": "",
            "output": "",
        })
        print(f"\n输入: {user_input}")
        print(f"分支: {result['intent']}")
        print(f"输出: {result['output']}")

    print_points(
        "LangGraph 可以理解为 Dify 工作流的代码版",
        "State 是变量，node 是节点，edge 是连线",
        "条件边用于实现分类、分流和动态流程",
    )


# ============================================================================
# 示例 2：多节点顺序执行
# ============================================================================
def example_2_sequential_nodes():
    """
    示例2：多节点顺序执行

    普通边表示固定流程：make_plan -> write_draft -> polish_answer
    """
    print_title("示例 2：多节点顺序执行")

    class State(TypedDict):
        user_request: str
        plan: str
        practice: str
        final_answer: str

    # 三个节点各写一个不同字段，是最安全的写法：
    #   make_plan   → 写 plan
    #   write_draft → 写 practice
    #   polish_answer → 读 plan + practice，写 final_answer
    #
    # 因为每个节点只动自己负责的字段，合并时不会互相覆盖，不需要 reducer。
    # 执行完之后，result 里同时包含 plan / practice / final_answer 的最终值。
    def make_plan(state: State):
        return {"plan": "概念理解 -> 语法拆解 -> 动手实践"}

    def write_draft(state: State):
        return {"practice": "每一步配套 1 个可运行小例子"}

    def polish_answer(state: State):
        return {
            "final_answer": (
                f"围绕「{state['user_request']}」学习：{state['plan']}；"
                f"{state['practice']}。"
            )
        }

    builder = StateGraph(State)
    builder.add_node("make_plan", make_plan)
    builder.add_node("write_draft", write_draft)
    builder.add_node("polish_answer", polish_answer)

    # 普通边适合表达固定顺序的流水线。
    builder.add_edge(START, "make_plan")
    builder.add_edge("make_plan", "write_draft")
    builder.add_edge("write_draft", "polish_answer")
    builder.add_edge("polish_answer", END)

    graph = builder.compile()
    initial_state = {
        "user_request": "学习 LangGraph",
        "plan": "",
        "practice": "",
        "final_answer": "",
    }
    # 这里 result 会同时包含：
    # user_request / plan / practice / final_answer
    result = graph.invoke(initial_state)

    print("流程: START -> make_plan -> write_draft -> polish_answer -> END")
    print(f"输入: {initial_state['user_request']}")
    print(f"make_plan -> {result['plan']}")
    print(f"write_draft -> {result['practice']}")
    print(f"polish_answer -> {result['final_answer']}")
    print_points(
        "普通边适合固定顺序流程",
        "后一个节点读取前一个节点写入的 state 字段",
    )


# ============================================================================
# 示例 3：条件路由
# ============================================================================
def example_3_conditional_routing():
    """
    示例3：条件路由

    add_conditional_edges 根据 state 选择下一节点。
    """
    print_title("示例 3：条件路由")

    class State(TypedDict):
        question: str
        category: str
        answer: str

    # 这个节点先写 category，后面路由函数再根据 category 决定走哪条边。
    def classify(state: State):
        if "报错" in state["question"] or "错误" in state["question"]:
            return {"category": "debug"}
        return {"category": "concept"}

    def answer_concept(state: State):
        return {"answer": "概念类问题：先解释核心定义，再给最小示例。"}

    def answer_debug(state: State):
        return {"answer": "排错类问题：先复现错误，再检查依赖、版本和参数。"}

    def route_by_category(state: State):
        if state["category"] == "debug":
            return "debug"
        return "concept"

    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("answer_concept", answer_concept)
    builder.add_node("answer_debug", answer_debug)

    builder.add_edge(START, "classify")

    # route_by_category 返回 "concept" 或 "debug"，
    # LangGraph 根据下面的映射表跳转到不同节点。
    #
    # 这个"先写 category，再按 category 路由"的模式很常见：
    # 先分类，再分配不同处理分支。
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "concept": "answer_concept",
            "debug": "answer_debug",
        },
    )
    builder.add_edge("answer_concept", END)
    builder.add_edge("answer_debug", END)

    graph = builder.compile()

    for question in ["什么是 StateGraph？", "运行时报错怎么排查？"]:
        result = graph.invoke({"question": question, "category": "", "answer": ""})
        print(f"\n问题: {question}")
        print(f"分类: {result['category']}")
        print(f"回答: {result['answer']}")

    print_points(
        "条件边适合分类、分支和是否继续",
        "路由函数只负责决定下一步",
    )


# ============================================================================
# 示例 4：使用 reducer 追加状态
# ============================================================================
def example_4_state_reducer():
    """
    示例4：使用 reducer 追加状态

    Annotated[list[str], operator.add] 表示列表更新时追加而不是覆盖。
    这个例子是故意让多个节点都写同一个字段，用来演示 reducer。
    """
    print_title("示例 4：使用 reducer 追加状态")

    class State(TypedDict):
        # ★ 前面示例 1-3 的节点各写不同字段，合并时互不干扰。
        # ★ 本例故意让多个节点都写 steps，演示"同字段多次更新"时的合并规则。
        #
        # 默认行为（无 reducer）：后写的覆盖前写的。
        #   collect_input  return {"steps": ["收集输入"]}   → steps = ["收集输入"]
        #   process_data   return {"steps": ["处理数据"]}   → steps = ["处理数据"]  ← 覆盖了！
        #
        # 用了 reducer（Annotated[list[str], operator.add]）：追加而不是覆盖。
        #   collect_input  return {"steps": ["收集输入"]}   → steps = ["收集输入"]
        #   process_data   return {"steps": ["处理数据"]}   → steps = ["收集输入", "处理数据"]  ← 追加！
        #
        # 原理：operator.add 就是 Python 的 +，对列表来说就是拼接：
        #   ["收集输入"] + ["处理数据"] = ["收集输入", "处理数据"]
        #
        # 真实项目里，messages、notes、logs、errors 等累积型字段都需要 reducer。
        steps: Annotated[list[str], operator.add]
        summary: str

    # 三个节点都 return {"steps": [...]}，写同一个字段。
    # 因为 steps 声明了 reducer，所以每次 return 的列表会被追加，不会丢失。
    def collect_input(state: State):
        return {"steps": ["收集输入"]}          # state["steps"] 变成 ["收集输入"]

    def process_data(state: State):
        return {"steps": ["处理数据"]}          # 追加 → ["收集输入", "处理数据"]

    def summarize(state: State):
        # state["steps"] 此时有 ["收集输入", "处理数据"]，再加一条"生成总结"。
        # return 了两字段：steps 追加，summary 覆盖（summary 没有 reducer）。
        steps_text = " -> ".join(state["steps"] + ["生成总结"])
        return {
            "steps": ["生成总结"],              # 追加 → ["收集输入", "处理数据", "生成总结"]
            "summary": f"执行路径: {steps_text}",  # 普通字段，直接赋值
        }

    builder = StateGraph(State)
    builder.add_node("collect_input", collect_input)
    builder.add_node("process_data", process_data)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "collect_input")
    builder.add_edge("collect_input", "process_data")
    builder.add_edge("process_data", "summarize")
    builder.add_edge("summarize", END)

    graph = builder.compile()
    # 初始时给 steps 一个空列表，后续节点会不断往这个列表里追加。
    result = graph.invoke({"steps": [], "summary": ""})

    print("语法: steps: Annotated[list[str], operator.add]")
    print("场景: 多个节点共同往同一个列表字段里追加内容")
    print(f"steps: {result['steps']}")
    print(f"summary: {result['summary']}")
    print_points(
        "没有 reducer 时，同字段更新默认覆盖",
        "reducer 适合 messages、notes、logs 这类累积型字段",
    )


# ============================================================================
# 示例 5：循环和终止条件
# ============================================================================
def example_5_loop_with_stop_condition():
    """
    示例5：循环和终止条件

    条件边可以返回前面的节点形成循环，但必须有终止条件。
    """
    print_title("示例 5：循环和终止条件")

    class State(TypedDict):
        task: str
        score: int          # 普通字段，每轮覆盖——只关心最新分数
        attempts: int       # 普通字段，每轮覆盖——只关心最新次数
        # history 记录多轮循环的轨迹，需要 reducer 追加保存。
        # 没有 reducer 的话，每轮循环都会覆盖 history，只能看到最后一轮。
        history: Annotated[list[str], operator.add]
        final_answer: str   # 普通字段，每轮覆盖——只保留最终版本的答案

    # draft 节点同时更新 4 个字段，演示同一个 return 里混合使用 reducer 和普通字段：
    #   attempts / final_answer / score → 普通字段，直接覆盖
    #   history → 有 reducer，追加到列表
    def draft(state: State):
        next_attempt = state["attempts"] + 1
        return {
            "attempts": next_attempt,                          # 覆盖：1 → 2 → 3
            "history": [f"草稿{next_attempt}"],                # 追加：["草稿1", "草稿2", ...]
            "final_answer": f"{state['task']} 的第 {next_attempt} 版答案",  # 覆盖：只保留最新版
        }

    def evaluate(state: State):
        score = min(100, state["score"] + 35)
        return {
            "score": score,                                   # 覆盖：35 → 70 → 100
            "history": [f"评分{score}"],                       # 追加：["草稿1", "评分35", ...]
        }

    # route_after_evaluate 决定：
    # - 继续改写一轮
    # - 还是结束整个流程
    def route_after_evaluate(state: State):
        if state["score"] >= 80 or state["attempts"] >= 3:
            return "finish"
        return "revise"

    builder = StateGraph(State)
    builder.add_node("draft", draft)
    builder.add_node("evaluate", evaluate)

    builder.add_edge(START, "draft")
    builder.add_edge("draft", "evaluate")

    # 条件边可以返回前面的节点形成循环。
    # 这里 "revise" 回到 draft，"finish" 结束到 END。
    #
    # 这就是 LangGraph 和普通线性流程的一个重要区别：
    # 图不一定只往前走，也可以回到前面的节点重复执行。
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "revise": "draft",
            "finish": END,
        },
    )

    graph = builder.compile()
    result = graph.invoke(
        {
            "task": "解释 LangGraph 循环",
            "score": 0,
            "attempts": 0,
            "history": [],
            "final_answer": "",
        },
        # recursion_limit 是保护网，防止条件写错导致无限循环。
        {"recursion_limit": 10},
    )

    print(f"attempts: {result['attempts']}")
    print(f"score: {result['score']}")
    print(f"history: {result['history']}")
    print(f"final_answer: {result['final_answer']}")
    print_points(
        "循环通过条件边返回前面的节点实现",
        "业务终止条件比 recursion_limit 更重要",
    )


# ============================================================================
# 示例 6：小型研究工作流
# ============================================================================
def example_6_research_workflow():
    """
    示例6：小型研究工作流

    综合使用状态、条件边和 reducer，构建一个简化研究流程。
    """
    print_title("示例 6：小型研究工作流")

    class State(TypedDict):
        query: str
        task_type: str
        # notes 汇总多个节点收集到的信息，适合用 reducer。
        notes: Annotated[list[str], operator.add]
        result: str

    # 这里的 task_type 和前面示例 3 的 category 是同一种思路：
    # 先分类，再根据分类结果选择不同处理分支。
    def classify_task(state: State):
        task_type = "compare" if "对比" in state["query"] else "explain"
        return {
            "task_type": task_type,
            "notes": [f"任务类型: {task_type}"],
        }

    def collect_facts(state: State):
        return {
            "notes": [
                "LangGraph 使用图组织工作流",
                "节点负责执行，边负责路由",
            ]
        }

    def compare_options(state: State):
        return {"notes": ["LangChain Agent 更高层，LangGraph 更底层可控"]}

    def explain_topic(state: State):
        return {"notes": ["适合需要状态、分支、循环的复杂应用"]}

    def write_report(state: State):
        return {
            "result": "\n".join(f"- {note}" for note in state["notes"])
        }

    def route_by_task_type(state: State):
        if state["task_type"] == "compare":
            return "compare"
        return "explain"

    builder = StateGraph(State)
    builder.add_node("classify_task", classify_task)
    builder.add_node("collect_facts", collect_facts)
    builder.add_node("compare_options", compare_options)
    builder.add_node("explain_topic", explain_topic)
    builder.add_node("write_report", write_report)

    builder.add_edge(START, "classify_task")
    builder.add_edge("classify_task", "collect_facts")

    # 根据 task_type 选择不同处理分支，然后汇合到 write_report。
    # 这是一个"分开处理，最后汇总"的常见工作流结构。
    builder.add_conditional_edges(
        "collect_facts",
        route_by_task_type,
        {
            "compare": "compare_options",
            "explain": "explain_topic",
        },
    )
    builder.add_edge("compare_options", "write_report")
    builder.add_edge("explain_topic", "write_report")
    builder.add_edge("write_report", END)

    graph = builder.compile()

    for query in ["解释 LangGraph 的适用场景", "对比 LangChain Agent 和 LangGraph"]:
        result = graph.invoke({
            "query": query,
            "task_type": "",
            "notes": [],
            "result": "",
        })
        print(f"\n查询: {query}")
        print(result["result"])

    print_points(
        "复杂流程可以拆成多个清晰节点",
        "条件边让工作流根据 state 动态变化",
    )


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - LangGraph Low Level (LangGraph 底层 API)")
    print("="*70)

    try:
        # example_1_dify_style_workflow()
        # input("\n按 Enter 继续...")

        # example_2_sequential_nodes()
        # input("\n按 Enter 继续...")

        # example_3_conditional_routing()
        # input("\n按 Enter 继续...")

        example_4_state_reducer()
        input("\n按 Enter 继续...")

        example_5_loop_with_stop_condition()
        input("\n按 Enter 继续...")

        example_6_research_workflow()

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点:")
        print("  1. LangGraph 可以理解为 Dify 工作流的代码版")
        print("  2. StateGraph 用共享 state 串联节点")
        print("  3. 条件边适合分支和循环")
        print("  4. reducer 控制多次状态更新如何合并")
        print("\n下一步:")
        print("  16_multi_agent - 学习多 Agent 协作")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
