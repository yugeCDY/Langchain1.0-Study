"""
LangChain 1.0 - 人在回路 (Human-in-the-Loop)
=============================================

本模块重点讲解：
1. 什么是 HITL，为什么实际产品需要人工介入
2. 用 interrupt() 在节点执行中暂停，等待用户输入
3. 用 Command(resume=...) 恢复执行，把用户决策传回节点
4. 用 Command(update=..., resume=...) 同时修改状态并恢复
5. 结合 checkpointer + thread_id 实现跨会话恢复

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
# 示例 1：审批流程（approve / reject）
# ============================================================================
def example_1_approval_flow():
    """
    示例1：审批流程

    最简单的 HITL 场景：节点生成内容后，用 interrupt() 暂停，
    等待用户 approve 或 reject，然后再决定下一步。
    """
    print("\n" + "=" * 70)
    print("示例 1：审批流程（approve / reject）")
    print("=" * 70)

    class State(TypedDict):
        topic: str
        content: str
        status: str

    def generate(state: State):
        return {"content": f"关于「{state['topic']}」的生成内容：这是一段示例文本。"}

    # ★ interrupt() 在节点执行到此处时暂停整个图的运行。
    #   它会抛出一个包含 payload 的 GraphInterrupt，图停止执行。
    #   等用户用 Command(resume=...) 恢复后，interrupt() 返回 resume 的值，
    #   节点继续执行后面的代码。
    def review(state: State):
        # interrupt 的第一个参数是 payload，可以传任何可序列化的数据。
        # 这里传一个 dict，告诉前端/用户：需要审批什么内容、有哪些选项。
        human_input = interrupt({
            "question": "是否批准以下内容？",
            "content": state["content"],
            "options": ["approve", "reject"],
        })

        # ★ 上面 interrupt() 被恢复后，human_input = Command(resume=...) 传的值。
        if human_input == "approve":
            return {"status": "approved"}
        return {"status": "rejected"}

    builder = StateGraph(State)
    builder.add_node("generate", generate)
    builder.add_node("review", review)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "review")
    builder.add_edge("review", END)

    # ★ HITL 必须搭配 checkpointer 使用。
    #   checkpointer 负责保存图的完整状态，这样中断后才能从断点恢复。
    #   没有 checkpointer，interrupt 后无法 resume。
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "example-1"}}

    # 第一次 invoke：图执行到 review 节点的 interrupt() 处暂停。
    print("[1] 启动图...")
    result = graph.invoke({"topic": "LangChain", "content": "", "status": ""}, config)

    # ★ graph.invoke() 遇到 interrupt() 后**正常返回**，不会抛异常。
    #   但 review 节点只执行到 interrupt() 之前，后面的代码还没跑。
    #   所以 status 字段是空的——这就是"暂停"的证据。
    print(f"    生成内容: {result.get('content', 'N/A')}")
    print(f"    审批状态: '{result.get('status', 'N/A')}' ← 空！因为 review 节点在 interrupt 处停了")

    # 查看 interrupt 信息。
    current_state = graph.get_state(config)
    payload = current_state.interrupts[0].value
    print(f"\n    [interrupt] {payload['question']}")
    print(f"    内容: {payload['content']}")
    print(f"    选项: {payload['options']}")

    # ★ 这里程序**真的停下来了**，用 input() 等待用户输入。
    #   在 Web/API 场景里，这里对应前端展示弹窗、等用户点击按钮。
    #   在命令行里，我们用 input() 模拟这个等待过程。
    user_input = input("\n请输入你的决策 (approve/reject): ").strip().lower()
    while user_input not in ("approve", "reject"):
        user_input = input("请输入 approve 或 reject: ").strip().lower()

    # ★ 用 Command(resume=user_input) 恢复执行。
    #    user_input 会作为 interrupt() 的返回值，review 节点继续执行后面的代码。
    print(f"[2] 用户输入 '{user_input}'，恢复执行...")
    result = graph.invoke(Command(resume=user_input), config)
    print(f"    最终 status: {result['status']}")

    print("\n关键点:")
    print("  - interrupt() 暂停后，status 字段为空（review 没跑完）")
    print("  - 程序用 input() 等待用户输入，模拟真实的审批等待")
    print("  - Command(resume=...) 把用户决策传回中断的节点")
    print("  - 必须用 checkpointer，否则中断后无法恢复")


# ============================================================================
# 示例 2：编辑内容（Command(update + resume)）
# ============================================================================
def example_2_edit_content():
    """
    示例2：编辑内容

    用户不仅可以 approve/reject，还可以修改内容。
    用 Command(update=..., resume=...) 同时修改 state 并恢复执行。
    """
    print("\n" + "=" * 70)
    print("示例 2：编辑内容（Command(update + resume)）")
    print("=" * 70)

    class State(TypedDict):
        draft: str
        final: str
        status: str

    def generate_draft(state: State):
        return {"draft": "初稿：今天天气不错，适合出门散步。"}

    def review_draft(state: State):
        human_input = interrupt({
            "question": "请审核草稿，选择操作：",
            "draft": state["draft"],
            "options": ["approve", "edit"],
        })

        if human_input == "approve":
            return {"status": "approved", "final": state["draft"]}
        # human_input == "edit" 的情况：
        # 理论上用户应该在 interrupt 期间修改内容，
        # 修改后的内容通过 Command(update=...) 写回 state。
        # 这里 review_draft 返回的 draft 就是用户编辑后的版本。
        return {"status": "edited"}

    builder = StateGraph(State)
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("review_draft", review_draft)
    builder.add_edge(START, "generate_draft")
    builder.add_edge("generate_draft", "review_draft")
    builder.add_edge("review_draft", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "edit-demo"}}
    print("[1] 生成草稿...")
    result = graph.invoke({"draft": "", "final": "", "status": ""}, config)
    print(f"    草稿: {result['draft']}")

    # 等待用户输入。
    user_input = input("\n请输入操作 (approve/edit): ").strip().lower()
    while user_input not in ("approve", "edit"):
        user_input = input("请输入 approve 或 edit: ").strip().lower()

    if user_input == "approve":
        print(f"[2] 用户 approve，恢复执行...")
        result = graph.invoke(Command(resume="approve"), config)
        print(f"    status: {result['status']}, final: {result['final']}")
    else:
        # 用户选择 edit，让用户输入修改后的内容。
        new_draft = input("请输入修改后的内容: ").strip()
        print(f"[2] 用户编辑后恢复...")
        # ★ Command(update={"draft": new_draft}, resume="edit") 同时做两件事：
        #   1. update: 把 draft 字段更新为用户编辑后的内容
        #   2. resume: 让 interrupt() 返回 "edit"，review_draft 继续执行
        result = graph.invoke(
            Command(update={"draft": new_draft}, resume="edit"),
            config,
        )
        print(f"    status: {result['status']}")
        print(f"    draft(已被 update 修改): {result['draft']}")

    print("\n关键点:")
    print("  - Command(update=...) 可以在恢复时修改任意 state 字段")
    print("  - 用户编辑内容后，不需要重新执行 generate 节点")
    print("  - update + resume 组合是 HITL 中最强大的模式")


# ============================================================================
# 示例 3：循环中的 interrupt
# ============================================================================
def example_3_interrupt_in_loop():
    """
    示例3：循环中的 interrupt

    interrupt 可以在循环中多次触发，每轮迭代都等待用户确认是否继续。
    """
    print("\n" + "=" * 70)
    print("示例 3：循环中的 interrupt")
    print("=" * 70)

    class State(TypedDict):
        step: int
        max_steps: int
        # history 记录每轮迭代的结果，需要 reducer 追加。
        history: Annotated[list[str], operator.add]
        done: bool

    def process_step(state: State):
        current = state["step"] + 1
        return {
            "step": current,
            "history": [f"第 {current} 步处理完成"],
        }

    def confirm(state: State):
        # 每轮处理完后询问用户是否继续下一轮。
        human_input = interrupt({
            "question": f"第 {state['step']} 步已完成，是否继续下一步？",
            "current_step": state["step"],
            "options": ["continue", "stop"],
        })

        if human_input == "stop":
            return {"done": True}
        return {"done": False}

    def route_loop(state: State):
        if state["done"]:
            return END
        if state["step"] >= state["max_steps"]:
            return END
        return "process_step"

    builder = StateGraph(State)
    builder.add_node("process_step", process_step)
    builder.add_node("confirm", confirm)

    builder.add_edge(START, "process_step")
    builder.add_edge("process_step", "confirm")
    builder.add_conditional_edges("confirm", route_loop)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "loop-demo"}}

    # 启动图，第一轮执行到 confirm 的 interrupt 处。
    print("[启动] 图开始执行...")
    graph.invoke({"step": 0, "max_steps": 5, "history": [], "done": False}, config)

    # 循环：每轮等待用户输入，直到用户输入 stop 或达到 max_steps。
    while True:
        current = graph.get_state(config)
        if not current.interrupts:
            break  # 没有中断说明已经结束了

        payload = current.interrupts[0].value
        print(f"\n    [interrupt] {payload['question']}")
        print(f"    已执行: {payload['current_step']} 步")
        print(f"    选项: {payload['options']}")

        user_input = input("\n请输入 (continue/stop): ").strip().lower()
        while user_input not in ("continue", "stop"):
            user_input = input("请输入 continue 或 stop: ").strip().lower()

        if user_input == "stop":
            print("[用户] stop -> 结束")
            result = graph.invoke(Command(resume="stop"), config)
            break

        print("[用户] continue -> 下一轮")
        graph.invoke(Command(resume="continue"), config)

    # 循环结束后，图要么被 stop 终止，要么自然到达 max_steps。
    final_state = graph.get_state(config)
    print(f"\n执行历史: {final_state.values.get('history', [])}")
    print(f"最终 step: {final_state.values.get('step', 0)}")

    print("\n关键点:")
    print("  - 循环中每轮都可以触发 interrupt，用户实时决定下一步")
    print("  - 用 while current.interrupts 检测是否还有未处理的中断")
    print("  - 适合需要人工逐步确认的长流程")


# ============================================================================
# 示例 4：持久化 + 跨会话恢复
# ============================================================================
def example_4_persistence_resume():
    """
    示例4：持久化 + 跨会话恢复

    用 SqliteSaver 把状态持久化到磁盘。
    程序关闭后重新打开，只要用同一个 thread_id 就能从断点恢复。
    """
    print("\n" + "=" * 70)
    print("示例 4：持久化 + 跨会话恢复")
    print("=" * 70)

    # ★ 实际项目中，InMemorySaver 只在演示用（程序关闭就丢了）。
    #   产品环境用 SqliteSaver 或 PostgresSaver，状态持久化到数据库。
    #   这里用 InMemorySaver 演示概念，代码结构和 SqliteSaver 完全一致。

    class State(TypedDict):
        task: str
        result: str
        status: str

    def do_work(state: State):
        return {"result": f"已完成: {state['task']}"}

    def wait_confirm(state: State):
        human_input = interrupt({
            "question": "任务已完成，请确认",
            "result": state["result"],
            "options": ["ok"],
        })
        return {"status": f"confirmed_by_{human_input}"}

    builder = StateGraph(State)
    builder.add_node("do_work", do_work)
    builder.add_node("wait_confirm", wait_confirm)
    builder.add_edge(START, "do_work")
    builder.add_edge("do_work", "wait_confirm")
    builder.add_edge("wait_confirm", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # 模拟"程序第一次运行"：执行到 interrupt 处。
    print("[程序第一次运行] 执行到 interrupt，准备关闭...")
    config = {"configurable": {"thread_id": "persistent-demo"}}
    graph.invoke({"task": "数据分析", "result": "", "status": ""}, config)

    state_before = graph.get_state(config)
    print(f"    thread_id: {config['configurable']['thread_id']}")
    print(f"    interrupts: {len(state_before.interrupts)} 个未处理")

    # 模拟"程序重新启动"：只要 thread_id 相同，就能恢复。
    # 实际项目中，这里可能是：
    #   1. Web 服务重启后，用户再次发起 /resume 请求
    #   2. 另一个进程读取 SQLite 中的 checkpoint，继续执行
    print("\n[程序重新启动] 用同一个 thread_id 恢复...")
    # 注意：这里用同一个 checkpointer 和 graph 对象演示。
    # 实际跨进程恢复时，需要重新创建 graph + checkpointer，但 thread_id 不变。
    result = graph.invoke(Command(resume="ok"), config)
    print(f"    恢复后 status: {result['status']}")

    print("\n关键点:")
    print("  - InMemorySaver: 内存存储，进程结束就丢失（演示用）")
    print("  - SqliteSaver: 存到 SQLite 文件，进程重启可恢复")
    print("  - thread_id 是恢复的唯一标识，必须保持一致")


# ============================================================================
# 示例 5：工具调用前的人工审批
# ============================================================================
def example_5_tool_call_approval():
    """
    示例5：工具调用前的人工审批

    最接近实际产品的场景：Agent 想执行敏感操作（如发送邮件、转账）前，
    先用 interrupt 请求用户确认。
    """
    print("\n" + "=" * 70)
    print("示例 5：工具调用前的人工审批")
    print("=" * 70)

    class State(TypedDict):
        query: str
        action: str
        approved: bool
        result: str

    def plan_action(state: State):
        # 分析用户意图，决定要执行什么操作。
        if "邮件" in state["query"] or "发送" in state["query"]:
            return {"action": "send_email"}
        return {"action": "none"}

    def request_approval(state: State):
        # ★ 敏感操作前触发 interrupt，等待用户审批。
        human_input = interrupt({
            "question": f"Agent 请求执行敏感操作：{state['action']}",
            "details": f"原始请求: {state['query']}",
            "options": ["approve", "deny"],
        })

        if human_input == "approve":
            return {"approved": True}
        return {"approved": False, "result": "操作已被用户拒绝"}

    def execute_action(state: State):
        if not state["approved"]:
            return {}
        return {"result": f"已执行 {state['action']}：操作成功完成"}

    def route_after_approval(state: State):
        if state["action"] == "none":
            return END
        if state["approved"]:
            return "execute_action"
        return END

    builder = StateGraph(State)
    builder.add_node("plan_action", plan_action)
    builder.add_node("request_approval", request_approval)
    builder.add_node("execute_action", execute_action)

    builder.add_edge(START, "plan_action")
    builder.add_conditional_edges("plan_action", route_after_approval)
    builder.add_edge("request_approval", "execute_action")
    builder.add_edge("execute_action", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # 场景 A：敏感操作，需要用户审批。
    print("[场景A] 用户请求：帮我发送一封邮件")
    config = {"configurable": {"thread_id": "tool-approval"}}
    result = graph.invoke({"query": "帮我发送一封邮件给老板", "action": "", "approved": False, "result": ""}, config)

    # 检查是否触发了 interrupt（敏感操作会触发，非敏感操作不会）。
    current = graph.get_state(config)
    if current.interrupts:
        payload = current.interrupts[0].value
        print(f"\n    [interrupt] {payload['question']}")
        print(f"    详情: {payload['details']}")
        print(f"    选项: {payload['options']}")

        user_input = input("\n请输入 (approve/deny): ").strip().lower()
        while user_input not in ("approve", "deny"):
            user_input = input("请输入 approve 或 deny: ").strip().lower()

        result = graph.invoke(Command(resume=user_input), config)
        print(f"    result: {result.get('result', '操作被拒绝，无结果')}")
    else:
        print(f"    action: {result['action']}（非敏感操作，无需审批）")

    # 场景 B：非敏感操作，直接通过。
    print("\n[场景B] 用户请求：今天天气怎么样")
    config2 = {"configurable": {"thread_id": "tool-weather"}}
    result2 = graph.invoke({"query": "今天天气怎么样", "action": "", "approved": False, "result": ""}, config2)
    print(f"    action: {result2['action']}（非敏感操作，无需审批，直接结束）")

    print("\n关键点:")
    print("  - 敏感操作前用 interrupt 阻断，等用户确认")
    print("  - 非敏感操作可以直接走，不需要 interrupt")
    print("  - 用 graph.get_state(config).interrupts 判断是否触发了中断")
    print("  - 这是 HITL 在实际产品中最常见的用途")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - 人在回路 (Human-in-the-Loop)")
    print("=" * 70)

    try:
        example_1_approval_flow()
        input("\n按 Enter 继续...")

        example_2_edit_content()
        input("\n按 Enter 继续...")

        example_3_interrupt_in_loop()
        input("\n按 Enter 继续...")

        example_4_persistence_resume()
        input("\n按 Enter 继续...")

        example_5_tool_call_approval()

        print("\n" + "=" * 70)
        print(" 完成！")
        print("=" * 70)
        print("\n核心要点:")
        print("  1. interrupt() 在节点内暂停，等待外部输入")
        print("  2. Command(resume=...) 恢复并传回用户决策")
        print("  3. Command(update=..., resume=...) 同时修改 state")
        print("  4. checkpointer + thread_id 实现持久化和跨会话恢复")
        print("  5. HITL 适合审批、编辑、敏感操作确认等场景")
        print("\n下一步:")
        print("  18_subgraphs - 学习子图嵌套和图组合")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
