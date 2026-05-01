"""
LangChain 1.0 - 多 Agent 协作 (Multi-Agent Collaboration)
=========================================================

本模块重点讲解：
1. 什么是多 Agent 系统，什么时候需要它
2. 用 create_agent 快速创建单个 Agent（LangChain 1.0 统一 API）
3. 手动实现 Supervisor 模式（协调者 + 多个专业 Agent）
4. Agent 之间直接 Handoff 的网络模式
5. 多 Agent 共享状态与最佳实践

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

from langchain_core.tools import tool

# ★ create_agent 是 LangChain 1.0 的统一 API：
#    传入 model + tools，它自动帮你建好 ReAct 循环图（思考→行动→观察→...）。
#    返回的是一个编译好的图，可以直接 invoke，也可以当作子图嵌入更大的工作流。
from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from typing import Annotated
import operator


# ============================================================================
# 示例 1：多 Agent 协作的概念（纯逻辑演示）
# ============================================================================
def example_1_multi_agent_concept():
    """
    示例1：多 Agent 协作的概念

    用纯 Python 逻辑演示"一个协调者把任务分给不同专家"的核心思想。
    不涉及 LLM，先理解 wiring 模式。
    """
    print("\n" + "=" * 70)
    print("示例 1：多 Agent 协作的概念（纯逻辑演示）")
    print("=" * 70)

    class State(TypedDict):
        query: str
        task_type: str
        result: str

    # 协调者：分析需求，决定交给哪个专家。
    def coordinator(state: State):
        if "代码" in state["query"] or "bug" in state["query"]:
            return {"task_type": "code"}
        return {"task_type": "general"}

    # 代码专家：处理编程相关问题。
    def code_expert(state: State):
        return {"result": f"[代码专家] 分析 '{state['query']}'：检查语法、依赖和运行环境。"}

    # 通用专家：处理其他问题。
    def general_expert(state: State):
        return {"result": f"[通用专家] 回答 '{state['query']}'：提供概念解释和操作步骤。"}

    # 路由：根据 task_type 决定调用哪个专家。
    def route_to_expert(state: State):
        if state["task_type"] == "code":
            return "code"
        return "general"

    builder = StateGraph(State)
    builder.add_node("coordinator", coordinator)
    builder.add_node("code_expert", code_expert)
    builder.add_node("general_expert", general_expert)

    builder.add_edge(START, "coordinator")
    builder.add_conditional_edges(
        "coordinator",
        route_to_expert,
        {"code": "code_expert", "general": "general_expert"},
    )
    builder.add_edge("code_expert", END)
    builder.add_edge("general_expert", END)

    graph = builder.compile()

    for query in ["这段代码有 bug 怎么修？", "介绍一下 LangChain 的用途"]:
        result = graph.invoke({"query": query, "task_type": "", "result": ""})
        print(f"\n输入: {query}")
        print(f"路由: {result['task_type']} -> {result['result']}")

    print("\n关键点:")
    print("  - 多 Agent = 多个节点，每个节点负责一个专业领域")
    print("  - 协调者决定任务分配给哪个 Agent（Supervisor 模式）")
    print("  - 实际项目中，每个节点内部是一个完整的 LLM Agent")


# ============================================================================
# 示例 2：用 create_agent 创建单个 Agent
# ============================================================================
def example_2_create_agent():
    """
    示例2：用 create_agent 创建单个 Agent

    create_agent(model, tools) 自动搭建 ReAct 循环：
    思考 -> 决定调用哪个工具 -> 执行工具 -> 观察结果 -> 继续思考...
    """
    print("\n" + "=" * 70)
    print("示例 2：用 create_agent 创建单个 Agent")
    print("=" * 70)

    # 定义两个简单工具。实际项目中可以是查数据库、调 API、搜向量库等。
    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气"""
        return f"{city} 今天晴朗，气温 25°C。"

    @tool
    def get_news(topic: str) -> str:
        """查询指定话题的最新资讯"""
        return f"关于「{topic}」的最新消息：LangGraph 1.0 正式发布。"

    # ★ create_agent 返回的是一个编译好的图。
    #    它内部已经包含了：
    #    - 一个 "agent" 节点（调用 LLM 做推理）
    #    - 一个 "tools" 节点（执行工具）
    #    - 条件边：LLM 想调用工具 -> 走 tools；不想调用 -> 结束
    #    你不需要手动写这些节点和边。
    agent = create_agent(
        model=model,
        tools=[get_weather, get_news],
    )

    # 调用方式：传入 messages，agent 会自动决定是否需要调用工具。
    # 返回的 result 包含完整的对话历史 messages。
    print("问题: 北京今天天气怎么样？")
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
        )
        # result["messages"] 是完整的对话历史，最后一条是 Agent 的最终回复。
        final_msg = result["messages"][-1]
        print(f"Agent 回复: {final_msg.content}")
    except Exception as e:
        print(f"⚠️ 调用失败（可能模型不支持 tool calling）: {e}")

    print("\n问题: 最近有什么 AI 新闻？")
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "最近有什么 AI 新闻？"}]}
        )
        final_msg = result["messages"][-1]
        print(f"Agent 回复: {final_msg.content}")
    except Exception as e:
        print(f"⚠️ 调用失败（可能模型不支持 tool calling）: {e}")

    print("\n关键点:")
    print("  - create_agent = 一行代码搭好 ReAct Agent")
    print("  - 内部自动处理思考→工具→观察的循环")
    print("  - 返回的是可 invoke 的图，也可以嵌入更大的多 Agent 工作流")


# ============================================================================
# 示例 3：手动实现 Supervisor 模式
# ============================================================================
def example_3_supervisor_pattern():
    """
    示例3：手动实现 Supervisor 模式

    Supervisor 不是"路由一次就结束"，而是持续协调：
    收到任务 -> 决定调用 Agent A -> A 返回结果 -> Supervisor 再决定 ->
    调用 Agent B -> B 返回 -> Supervisor 汇总 -> 结束
    """
    print("\n" + "=" * 70)
    print("示例 3：手动实现 Supervisor 模式")
    print("=" * 70)

    class State(TypedDict):
        query: str
        # ★ step 字段记录当前阶段，Supervisor 靠它判断"现在该做什么"。
        #   "start"      = 刚收到用户问题，还没开始处理
        #   "got_weather" = weather_agent 已返回，有了天气信息
        #   "got_fashion" = fashion_agent 已返回，有了穿衣建议
        #   "done"        = Supervisor 汇总完成，可以结束
        step: str
        weather_info: str     # weather_agent 写入
        fashion_advice: str   # fashion_agent 写入
        # notes 记录完整执行轨迹，需要 reducer 追加。
        notes: Annotated[list[str], operator.add]
        final_answer: str

    # ★ Supervisor 是核心协调者。每次 Agent 执行完都会回到这里，
    #   Supervisor 根据当前 step 决定下一步干什么。
    def supervisor(state: State):
        step = state["step"]
        query = state["query"]

        if step == "" or step == "start":
            # 刚收到问题，先查天气。
            return {
                "step": "need_weather",
                "notes": [f"[Supervisor] 收到问题: {query}，决定先查天气"],
            }

        elif step == "got_weather":
            # 已有天气，继续获取穿衣建议。
            return {
                "step": "need_fashion",
                "notes": [f"[Supervisor] 天气已获取({state['weather_info']})，决定获取穿衣建议"],
            }

        elif step == "got_fashion":
            # 所有信息收集完毕，汇总生成最终答案。
            return {
                "step": "done",
                "notes": ["[Supervisor] 所有信息收集完毕，汇总生成最终答案"],
                "final_answer": (
                    f"天气: {state['weather_info']} | 穿衣: {state['fashion_advice']}"
                ),
            }

        return {"step": "done", "notes": ["[Supervisor] 未知状态，直接结束"]}

    # 天气 Agent：专门查询天气信息。
    # 返回后 step 被更新为 "got_weather"，Supervisor 下一轮会进入对应分支。
    def weather_agent(state: State):
        return {
            "weather_info": "北京明天多云，18~26°C，可能有短时阵雨",
            "step": "got_weather",
            "notes": ["[天气Agent] 查询完成: 北京明天多云，18~26°C"],
        }

    # 穿搭 Agent：根据天气给穿衣建议。
    # 返回后 step 被更新为 "got_fashion"，Supervisor 下一轮会汇总结束。
    def fashion_agent(state: State):
        weather = state["weather_info"]
        return {
            "fashion_advice": f"根据天气({weather})，建议带薄外套+短袖，备一把伞",
            "step": "got_fashion",
            "notes": ["[穿搭Agent] 建议完成: 薄外套+短袖，备伞"],
        }

    # ★ 路由函数：根据 step 的值决定下一步走哪个节点。
    #   "need_weather" -> weather_agent
    #   "need_fashion" -> fashion_agent
    #   "done"         -> END
    def route_by_step(state: State):
        step = state["step"]
        if step == "need_weather":
            return "weather_agent"
        elif step == "need_fashion":
            return "fashion_agent"
        return END  # "done" 或其他状态直接结束

    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor)
    builder.add_node("weather_agent", weather_agent)
    builder.add_node("fashion_agent", fashion_agent)

    builder.add_edge(START, "supervisor")

    # Supervisor 执行完，根据 step 的值路由到不同节点。
    builder.add_conditional_edges(
        "supervisor",
        route_by_step,
        {
            "weather_agent": "weather_agent",
            "fashion_agent": "fashion_agent",
            END: END,
        },
    )

    # ★ 关键：Agent 执行完后**回到 Supervisor**，不是直接结束。
    #   这就是 Supervisor 模式的循环本质。
    builder.add_edge("weather_agent", "supervisor")
    builder.add_edge("fashion_agent", "supervisor")

    graph = builder.compile()

    result = graph.invoke(
        {
            "query": "明天去北京出差，帮我看看天气和穿什么",
            "step": "",
            "weather_info": "",
            "fashion_advice": "",
            "notes": [],
            "final_answer": "",
        }
    )

    print("问题: 明天去北京出差，帮我看看天气和穿什么")
    print(f"\n执行轨迹:")
    for note in result["notes"]:
        print(f"  {note}")
    print(f"\n最终回答: {result['final_answer']}")

    print("\n关键点:")
    print("  - Supervisor 不是一次路由，而是持续协调的循环")
    print("  - Agent 执行完回到 Supervisor，由 Supervisor 决定下一步")
    print("  - step 字段记录阶段状态，驱动整个流程推进")


# ============================================================================
# 示例 4：Agent Handoff 网络模式
# ============================================================================
def example_4_agent_handoff():
    """
    示例4：Agent Handoff 网络模式

    与 Supervisor 不同，Handoff 模式没有中央协调者。
    Agent 执行完自己的任务后，可以直接决定把控制权转交给另一个 Agent。
    """
    print("\n" + "=" * 70)
    print("示例 4：Agent Handoff 网络模式")
    print("=" * 70)

    class State(TypedDict):
        query: str
        # ★ next_agent 字段决定下一个要执行的 Agent。
        #   每个 Agent 节点执行完后更新 next_agent，实现"接力"效果。
        next_agent: str
        notes: Annotated[list[str], operator.add]
        result: str

    # 接待 Agent：识别用户需求，决定转交给谁。
    def reception_agent(state: State):
        query = state["query"]
        if "退款" in query or "订单" in query:
            next_agent = "order_agent"
        elif "技术" in query or "bug" in query:
            next_agent = "tech_agent"
        else:
            next_agent = "END"
        return {
            "next_agent": next_agent,
            "notes": [f"[接待] 识别到需求，转交 {next_agent}"],
        }

    # 订单 Agent：处理订单相关问题。
    def order_agent(state: State):
        return {
            "next_agent": "END",
            "notes": ["[订单] 已查询订单状态，处理退款申请"],
            "result": "您的退款申请已提交，预计 3 个工作日内到账。",
        }

    # 技术 Agent：处理技术问题。
    def tech_agent(state: State):
        return {
            "next_agent": "END",
            "notes": ["[技术] 已排查问题，提供解决方案"],
            "result": "请更新到最新版本，该 bug 已在 v2.1 中修复。",
        }

    # 路由：根据 next_agent 字段跳转到对应节点，或结束。
    def route_handoff(state: State):
        next_agent = state.get("next_agent", "END")
        if next_agent == "END":
            return END
        return next_agent

    builder = StateGraph(State)
    builder.add_node("reception", reception_agent)
    builder.add_node("order_agent", order_agent)
    builder.add_node("tech_agent", tech_agent)

    builder.add_edge(START, "reception")

    # ★ Handoff 的核心：每个节点都可以决定下一个节点是谁。
    #   reception 之后由 route_handoff 决定走 order_agent、tech_agent 还是 END。
    builder.add_conditional_edges("reception", route_handoff)
    builder.add_conditional_edges("order_agent", route_handoff)
    builder.add_conditional_edges("tech_agent", route_handoff)

    graph = builder.compile()

    for query in ["我想申请退款", "登录时出现 bug"]:
        result = graph.invoke(
            {"query": query, "next_agent": "", "notes": [], "result": ""}
        )
        print(f"\n问题: {query}")
        print(f"执行路径: {' -> '.join(result['notes'])}")
        print(f"结果: {result['result']}")

    print("\n关键点:")
    print("  - Handoff 没有中央协调者，Agent 自己决定下一步")
    print("  - 适合客服、工单流转等场景")
    print("  - 每个节点都有可能指向任意其他节点（包括自己）")


# ============================================================================
# 示例 5：最佳实践 - 可复用的 Agent 节点工厂
# ============================================================================
def example_5_agent_factory_pattern():
    """
    示例5：最佳实践 - 可复用的 Agent 节点工厂

    实际项目中，不要用一堆重复代码定义 Agent 节点。
    用一个工厂函数统一创建，然后注册到 StateGraph 中。
    """
    print("\n" + "=" * 70)
    print("示例 5：最佳实践 - 可复用的 Agent 节点工厂")
    print("=" * 70)

    class State(TypedDict):
        query: str
        agent_outputs: Annotated[list[str], operator.add]
        final_result: str

    # ★ Agent 节点工厂：传入 name 和 logic，返回一个节点函数。
    #   这样新增 Agent 时只需改配置，不用复制粘贴节点函数。
    def make_agent_node(name: str, logic):
        def agent_node(state: State):
            result = logic(state["query"])
            return {
                "agent_outputs": [f"[{name}] {result}"],
            }

        return agent_node

    # 用工厂创建三个 Agent 节点。
    search_agent = make_agent_node(
        "搜索", lambda q: f"搜索到 3 条关于「{q}」的结果"
    )
    summarize_agent = make_agent_node(
        "总结", lambda q: f"总结：「{q}」的核心要点如下..."
    )
    format_agent = make_agent_node(
        "格式化", lambda q: "已按 Markdown 格式排版完成"
    )

    # 顺序执行：搜索 -> 总结 -> 格式化
    builder = StateGraph(State)
    builder.add_node("search", search_agent)
    builder.add_node("summarize", summarize_agent)
    builder.add_node("format", format_agent)

    builder.add_edge(START, "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "format")
    builder.add_edge("format", END)

    graph = builder.compile()

    result = graph.invoke(
        {"query": "LangGraph 多 Agent", "agent_outputs": [], "final_result": ""}
    )

    print("流程: START -> 搜索 -> 总结 -> 格式化 -> END")
    print(f"query: {result['query']}")
    print("各 Agent 输出:")
    for out in result["agent_outputs"]:
        print(f"  {out}")

    print("\n关键点:")
    print("  - 用工厂函数避免重复代码")
    print("  - Agent 逻辑与图结构分离，便于维护")
    print("  - 新增 Agent 只需一行配置，不用改图结构")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - 多 Agent 协作 (Multi-Agent Collaboration)")
    print("=" * 70)

    try:
        example_1_multi_agent_concept()
        input("\n按 Enter 继续...")

        example_2_create_agent()
        input("\n按 Enter 继续...")

        example_3_supervisor_pattern()
        input("\n按 Enter 继续...")

        example_4_agent_handoff()
        input("\n按 Enter 继续...")

        example_5_agent_factory_pattern()

        print("\n" + "=" * 70)
        print(" 完成！")
        print("=" * 70)
        print("\n核心要点:")
        print("  1. 多 Agent = 多个专业节点 + 路由机制")
        print("  2. Supervisor 模式：中央协调者分配任务")
        print("  3. Handoff 模式：Agent 之间直接交接")
        print("  4. create_agent 一行代码搭好单 Agent")
        print("  5. 共享状态用 reducer 追加，避免覆盖")
        print("\n下一步:")
        print("  17_human_in_the_loop - 学习人工审批与打断")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
