"""
LangChain 1.0 - 流式输出与事件 (Streaming & Events)
=====================================================

本模块重点讲解：
1. invoke vs stream：同步阻塞 vs 实时流式
2. stream_mode="updates"：只输出节点增量
3. stream_mode="values"：输出完整 State
4. stream_mode="messages"：流式输出 LLM tokens
5. stream_mode="custom" + get_stream_writer()：自定义进度事件
6. subgraphs=True：子图事件也流出来

说明：
- 详细知识点和逐句语法解释见 README.md
- 本模块涉及实时输出，建议在终端直接运行观察效果
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
from langgraph.config import get_stream_writer
from typing_extensions import TypedDict
from typing import Annotated
import operator
import time


# ============================================================================
# 公共 State 定义
# ============================================================================

class State(TypedDict):
    """工作流状态：topic 输入 -> 多步处理 -> result 输出"""
    topic: str
    plan: str
    result: str


class ChatState(TypedDict):
    """聊天状态：用于 messages 流式模式，必须有 messages 字段"""
    messages: Annotated[list, operator.add]


# ============================================================================
# 示例 1：invoke 阻塞调用 vs stream 实时流式
# ============================================================================
def example_1_invoke_vs_stream():
    """
    示例1：对比同步 invoke 和流式 stream 的区别

    invoke()：阻塞等待整个图执行完，返回最终结果。
    stream()：图每执行完一个节点，就实时输出一次，不阻塞。
    """
    print("\n" + "=" * 70)
    print("示例 1：invoke 阻塞调用 vs stream 实时流式")
    print("=" * 70)

    # 节点1：规划阶段（模拟思考）
    def plan_node(state: State):
        return {"plan": f"计划：围绕「{state['topic']}」收集素材并撰写"}

    # 节点2：生成阶段（调用 LLM）
    def generate_node(state: State):
        prompt = f"请用一句话介绍「{state['topic']}」"
        response = model.invoke(prompt)
        return {"result": response.content}

    # 构建图
    builder = StateGraph(State)
    builder.add_node("plan", plan_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "generate")
    builder.add_edge("generate", END)
    graph = builder.compile()

    # --- 方式 A：invoke（阻塞）---
    print("\n【方式 A】graph.invoke() —— 阻塞调用，一次性返回完整结果：")
    result = graph.invoke({"topic": "LangGraph"})
    print(f"  完整结果: {result}")

    # --- 方式 B：stream（实时）---
    print("\n【方式 B】graph.stream() —— 实时流式，节点执行完就推送：")
    for chunk in graph.stream({"topic": "LangGraph"}):
        # chunk 的格式：{node_name: {field: value}}
        for node_name, update in chunk.items():
            print(f"  [{node_name}] 输出: {update}")

    print("\n关键点：")
    print("  - invoke：适合后台任务、批量处理，不关心中间过程")
    print("  - stream：适合前端 UI、聊天界面，需要实时反馈")


# ============================================================================
# 示例 2：stream_mode="updates" vs "values"
# ============================================================================
def example_2_stream_modes():
    """
    示例2：两种核心 stream_mode 的区别

    updates 模式：只输出节点返回的增量（增量更新）
    values 模式：输出完整的 State 快照（全量状态）
    """
    print("\n" + "=" * 70)
    print("示例 2：stream_mode=\"updates\" vs \"values\"")
    print("=" * 70)

    def step1(state: State):
        return {"plan": "步骤1完成"}

    def step2(state: State):
        return {"result": "步骤2完成"}

    builder = StateGraph(State)
    builder.add_node("step1", step1)
    builder.add_node("step2", step2)
    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", END)
    graph = builder.compile()

    # --- updates 模式 ---
    print("\n【updates 模式】只输出节点返回的增量：")
    for chunk in graph.stream(
        {"topic": "test"},
        stream_mode="updates",
    ):
        # chunk 格式：{node_name: {field: value}}
        print(f"  {chunk}")

    # --- values 模式 ---
    print("\n【values 模式】输出完整 State（包含之前所有字段）：")
    for chunk in graph.stream(
        {"topic": "test"},
        stream_mode="values",
    ):
        # chunk 格式：完整的 State 字典
        print(f"  {chunk}")

    print("\n关键点：")
    print("  - updates：省带宽，前端只接收变化的部分")
    print("  - values：全量快照，方便 UI 直接渲染完整状态")


# ============================================================================
# 示例 3：messages 模式 —— 流式输出 LLM tokens
# ============================================================================
def example_3_messages_streaming():
    """
    示例3：stream_mode="messages" 逐字流式输出

    messages 模式会捕获节点内 LLM 调用产生的 token，
    像 ChatGPT 打字机效果一样逐个字符输出。

    注意：需要 LLM 后端支持流式输出（SSE）。
    """
    print("\n" + "=" * 70)
    print("示例 3：messages 模式 —— 流式输出 LLM tokens")
    print("=" * 70)

    from langchain_core.messages import HumanMessage

    # 聊天节点：接收 messages，调用 LLM 回复
    def chat_node(state: ChatState):
        messages = state["messages"]
        response = model.invoke(messages)
        # 返回的 messages 会被 operator.add 追加到列表
        return {"messages": [response]}

    builder = StateGraph(ChatState)
    builder.add_node("assistant", chat_node)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    graph = builder.compile()

    user_msg = HumanMessage(content="用一句话介绍 LangGraph")

    print("\n【invoke 方式】一次性返回完整回复：")
    result = graph.invoke({"messages": [user_msg]})
    for msg in result["messages"]:
        print(f"  {msg.type}: {msg.content}")

    print("\n【方式 A】直接 model.stream() —— 最原始的 token 流式：")
    print("  回复: ", end="", flush=True)
    # model.stream() 是底层 API，直接产生 token 流
    # 这是所有"打字机效果"的本质，不依赖 LangGraph
    full_text = ""
    for chunk in model.stream([user_msg]):
        text = chunk.content
        full_text += text
        print(text, end="", flush=True)
    print(f"\n  (共 {len(full_text)} 字符)")

    print("\n【方式 B】graph.stream() + messages 模式 —— LangGraph 封装：")
    print("  回复: ", end="", flush=True)
    token_count = 0
    try:
        for chunk in graph.stream(
            {"messages": [user_msg]},
            stream_mode="messages",
        ):
            # chunk 格式：(message_chunk, metadata)
            # 注意：节点内用的是 model.invoke()，LangGraph 可能无法拆出 token
            # 部分模型/后端支持自动拦截，部分不支持
            msg_chunk, metadata = chunk
            if hasattr(msg_chunk, "content") and msg_chunk.content:
                print(msg_chunk.content, end="", flush=True)
                token_count += 1
        print()
        if token_count == 0:
            print("  (messages 模式未产生 token 级输出，模型可能不支持拦截)")
        else:
            print(f"  共输出 {token_count} 个 token")
    except Exception as e:
        print(f"\n  ⚠️ messages 模式失败: {e}")

    print("\n关键点：")
    print("  - messages 模式 = ChatGPT 打字机效果的核心实现")
    print("  - 适合聊天 UI、实时内容生成场景")
    print("  - 需要模型后端支持 SSE 流式传输")


# ============================================================================
# 示例 4：custom 模式 —— 节点内自定义进度事件
# ============================================================================
def example_4_custom_events():
    """
    示例4：stream_mode="custom" + get_stream_writer() 自定义事件

    在节点内部用 get_stream_writer() 发送任意自定义数据，
    适合进度条、工具状态通知、中间结果展示等场景。
    """
    print("\n" + "=" * 70)
    print("示例 4：custom 模式 —— 节点内自定义进度事件")
    print("=" * 70)

    # 模拟一个多步骤的耗时任务
    def long_task_node(state: State):
        writer = get_stream_writer()
        # 发送自定义事件到 stream_mode="custom"
        writer({"step": 1, "status": "开始分析需求..."})
        time.sleep(0.3)  # 模拟耗时

        writer({"step": 2, "status": "正在检索相关文档..."})
        time.sleep(0.3)

        writer({"step": 3, "status": "调用 LLM 生成内容..."})
        time.sleep(0.3)

        writer({"step": 4, "status": "正在格式化输出..."})
        time.sleep(0.2)

        return {"result": f"关于「{state['topic']}」的分析已完成"}

    builder = StateGraph(State)
    builder.add_node("task", long_task_node)
    builder.add_edge(START, "task")
    builder.add_edge("task", END)
    graph = builder.compile()

    print("\n【custom 模式】实时接收节点内部发送的进度事件：")
    for chunk in graph.stream(
        {"topic": "LangGraph 流式 API"},
        stream_mode="custom",
    ):
        # chunk 就是 writer() 传入的任意数据
        progress = chunk
        print(f"  [进度 {progress['step']}/4] {progress['status']}")

    print("\n关键点：")
    print("  - get_stream_writer() 只能在节点函数内部调用")
    print("  - 可以发送任意 JSON 可序列化的数据（字符串、字典等）")
    print("  - 适合进度条、状态通知、中间结果等场景")


# ============================================================================
# 示例 5：子图流式传播 —— subgraphs=True
# ============================================================================
def example_5_subgraph_streaming():
    """
    示例5：子图的事件也能流出来

    设置 subgraphs=True 后，父图 stream 时会同时输出子图的事件。
    结合 18_subgraphs 的知识，子图不是黑盒。
    """
    print("\n" + "=" * 70)
    print("示例 5：子图流式传播 —— subgraphs=True")
    print("=" * 70)

    # 子图：内部有两个节点
    def sub_step1(state: State):
        return {"plan": f"子图步骤1：处理 {state['topic']}"}

    def sub_step2(state: State):
        return {"result": "子图处理完成"}

    subgraph = StateGraph(State)
    subgraph.add_node("sub1", sub_step1)
    subgraph.add_node("sub2", sub_step2)
    subgraph.add_edge(START, "sub1")
    subgraph.add_edge("sub1", "sub2")
    subgraph.add_edge("sub2", END)
    sub = subgraph.compile()

    # 父图：嵌入子图
    def parent_finalize(state: State):
        return {"result": f"{state['result']} ← 父图收尾"}

    builder = StateGraph(State)
    builder.add_node("sub_process", sub)  # ← compile 后的子图直接嵌入
    builder.add_node("finalize", parent_finalize)
    builder.add_edge(START, "sub_process")
    builder.add_edge("sub_process", "finalize")
    builder.add_edge("finalize", END)
    graph = builder.compile()

    print("\n【默认 stream】只能看到父图节点的事件：")
    for chunk in graph.stream({"topic": "测试"}, stream_mode="updates"):
        for node_name, update in chunk.items():
            print(f"  [{node_name}] {update}")

    print("\n【subgraphs=True】子图内部节点的事件也可见：")
    for chunk in graph.stream(
        {"topic": "测试"},
        stream_mode="updates",
        subgraphs=True,
    ):
        # subgraphs=True 时，chunk 格式为 (namespace, {node_name: update})
        # namespace: () 表示根图，("sub_process:xxx",) 表示子图内的事件
        namespace, update_dict = chunk
        source = "根图" if not namespace else f"子图{namespace}"
        for node_name, update in update_dict.items():
            print(f"  [{source} -> {node_name}] {update}")

    print("\n关键点：")
    print("  - 默认 stream 子图是黑盒，只能看到子图整体节点的输出")
    print("  - subgraphs=True 让子图内部事件透明可见")
    print("  - 调试复杂工作流时非常有用")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - 流式输出与事件 (Streaming & Events)")
    print("=" * 70)

    try:
        example_1_invoke_vs_stream()
        input("\n按 Enter 继续...")

        example_2_stream_modes()
        input("\n按 Enter 继续...")

        example_3_messages_streaming()
        input("\n按 Enter 继续...")

        example_4_custom_events()
        input("\n按 Enter 继续...")

        example_5_subgraph_streaming()

        print("\n" + "=" * 70)
        print(" 完成！")
        print("=" * 70)
        print("\n核心要点：")
        print("  ✅ invoke 阻塞等待结果，stream 实时推送节点输出")
        print("  ✅ updates 模式输出增量，values 模式输出全量 State")
        print("  ✅ messages 模式实现打字机效果的 token 级流式")
        print("  ✅ custom 模式 + get_stream_writer() 发送自定义进度")
        print("  ✅ subgraphs=True 让子图内部事件也透明可见")
        print("\n下一步：")
        print("  20_production_ready - 生产环境部署与监控")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
