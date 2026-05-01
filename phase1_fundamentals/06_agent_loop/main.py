"""
LangChain 1.0 - Agent 执行循环
==============================

本模块重点讲解：
1. Agent 执行循环的详细过程
2. 流式输出（streaming）
3. 查看中间步骤
4. 理解消息流转
"""

import os
import sys

# 添加工具目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, '04_custom_tools', 'tools'))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from calculator import calculator
from weather import get_weather

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here_replace_this":
    raise ValueError("请先设置 GROQ_API_KEY")

model = init_chat_model(
        "openai:deepseek-v4-flash",  # Groq 提供的 Llama 3.3 模型
        api_key=GROQ_API_KEY,
        base_url="https://api.deepseek.com",
        extra_body={"thinking": {"type": "disabled"}},
    )
# model = init_chat_model("groq:llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


# ============================================================================
# 示例 1：理解执行循环 - 查看完整消息历史
# ============================================================================
def example_1_understand_loop():
    """
    示例1：查看 Agent 执行循环的每一步

    关键：response['messages'] 包含完整的对话历史
    """
    print("\n" + "="*70)
    print("示例 1：Agent 执行循环详解")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[calculator]
    )

    print("\n问题：25 乘以 8 等于多少？")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "25 乘以 8 等于多少？"}]
    })

    print("\n完整消息历史：")
    for i, msg in enumerate(response['messages'], 1):
        print(f"\n{'='*60}")
        print(f"消息 {i}: {msg.__class__.__name__}")
        print(f"{'='*60}")

        if hasattr(msg, 'content') and msg.content:
            print(f"内容: {msg.content}")

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"工具调用:")
            for tc in msg.tool_calls:
                print(f"  - 工具: {tc['name']}")
                print(f"  - 参数: {tc['args']}")

        if hasattr(msg, 'name'):
            print(f"工具名: {msg.name}")

    print("\n\n执行流程：")
    print("""
    1. HumanMessage    → 用户问题
    2. AIMessage       → AI 决定调用工具（包含 tool_calls）
    3. ToolMessage     → 工具执行结果
    4. AIMessage       → AI 基于结果生成最终答案
    """)

    print("\n关键点：")
    print("  - Agent 自动完成这个循环")
    print("  - 所有步骤都记录在 messages 中")
    print("  - 最后一条消息是最终答案")


# ============================================================================
# 示例 2：流式输出（Streaming）
# ============================================================================
def example_2_streaming():
    """
    示例2：实时查看 Agent 的输出

    两种流式模式：
    1. stream_mode="updates" - 按步骤输出（默认）
    2. stream_mode="messages" - 逐 token 输出
    """
    print("\n" + "="*70)
    print("示例 2：流式输出")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[calculator, get_weather]
    )

    # ========== 方式 1：按步骤输出 (stream_mode="updates") ==========
    print("\n【方式 1：按步骤输出】")
    print("问题：北京天气如何？")
    print("-" * 70)

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "北京天气如何？"}]},
        stream_mode="updates"  # 默认模式，每个节点执行完返回一次
    ):
        for node_name, node_output in chunk.items():
            if 'messages' in node_output:
                latest_msg = node_output['messages'][-1]
                msg_type = latest_msg.__class__.__name__

                if msg_type == "AIMessage":
                    if hasattr(latest_msg, 'tool_calls') and latest_msg.tool_calls:
                        print(f"[{node_name}] AI 调用工具: {latest_msg.tool_calls[0]['name']}")
                    elif latest_msg.content:
                        print(f"[{node_name}] AI 最终回答: {latest_msg.content}")
                elif msg_type == "ToolMessage":
                    print(f"[{node_name}] 工具返回: {latest_msg.content[:50]}...")

    # ========== 方式 2：逐 token 输出 (stream_mode="messages") ==========
    print("\n\n【方式 2：逐 token 输出】")
    print("问题：介绍langchain 500字左右")
    print("-" * 70)
    print("实时输出: ", end="", flush=True)

    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": "介绍langchain 500字左右"}]},
        stream_mode="messages"  # 逐 token 输出
    ):
        # 只显示 model 节点的输出（过滤工具调用等）
        if metadata.get("langgraph_node") == "model":
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)

    print("\n")  # 换行

    print("\n关键点：")
    print("  - stream_mode='updates': 按步骤输出，适合显示进度")
    print("  - stream_mode='messages': 逐 token 输出，适合打字机效果")
    print("  - messages 模式返回 (chunk, metadata) 元组")


# ============================================================================
# 示例 3：多步骤执行
# ============================================================================
def example_3_multi_step():
    """
    示例3：Agent 执行多个工具调用

    理解复杂任务的执行过程
    """
    print("\n" + "="*70)
    print("示例 3：多步骤执行")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt="你是一个数学助手。当遇到复杂计算时，分步骤计算。"
    )

    print("\n问题：先算 10 加 20，然后把结果乘以 3")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "先算 10 加 20，然后把结果乘以 3"}]
    })

    # 统计工具调用次数
    tool_calls_count = 0
    for msg in response['messages']:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_count += len(msg.tool_calls)

    print(f"\n工具调用次数: {tool_calls_count}")
    print(f"最终答案: {response['messages'][-1].content}")

    print("\n关键点：")
    print("  - Agent 可以多次调用工具")
    print("  - 每次调用的结果会影响下一步")
    print("  - 直到得到最终答案")


# ============================================================================
# 示例 4：查看中间状态
# ============================================================================
def example_4_inspect_state():
    """
    示例4：在执行过程中查看状态

    使用 stream 并检查每个 chunk
    """
    print("\n" + "="*70)
    print("示例 4：查看中间状态")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[calculator]
    )

    print("\n问题：100 除以 5 等于多少？")
    print("\n执行步骤：")

    step = 0
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "100 除以 5 等于多少？"}]
    }):
        step += 1
        print(f"\n步骤 {step}:")

        # chunk 格式: {'model': {'messages': [...]}} 或 {'tools': {'messages': [...]}}
        for node_name, node_output in chunk.items():
            if 'messages' in node_output:
                latest = node_output['messages'][-1]
                msg_type = latest.__class__.__name__
                print(f"  节点: {node_name}")
                print(f"  类型: {msg_type}")

                if hasattr(latest, 'tool_calls') and latest.tool_calls:
                    print(f"  工具调用: {latest.tool_calls[0]['name']}")
                elif hasattr(latest, 'content') and latest.content:
                    print(f"  内容: {latest.content[:50]}...")  # 只显示前50个字符

    print("\n关键点：")
    print("  - stream 让你看到每个步骤")
    print("  - 可以用于调试")
    print("  - 可以用于进度显示")


# ============================================================================
# 示例 5：理解消息类型
# ============================================================================
def example_5_message_types():
    """
    示例5：详解各种消息类型

    Agent 执行循环中的消息类型
    """
    print("\n" + "="*70)
    print("示例 5：消息类型详解")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[get_weather]
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "上海天气如何？"}]
    })

    print("\n消息类型分析：")
    for msg in response['messages']:
        msg_type = msg.__class__.__name__

        if msg_type == "HumanMessage":
            print(f"\n[HumanMessage] 用户输入")
            print(f"  内容: {msg.content}")

        elif msg_type == "AIMessage":
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"\n[AIMessage] AI 决定调用工具")
                print(f"  工具: {msg.tool_calls[0]['name']}")
                print(f"  参数: {msg.tool_calls[0]['args']}")
            else:
                print(f"\n[AIMessage] AI 的最终回答")
                print(f"  内容: {msg.content}")

        elif msg_type == "ToolMessage":
            print(f"\n[ToolMessage] 工具执行结果")
            print(f"  工具: {msg.name}")
            print(f"  结果: {msg.content}")

    print("\n\n消息类型总结：")
    print("""
    HumanMessage  → 用户的输入
    AIMessage     → AI 的输出（可能包含 tool_calls 或最终答案）
    ToolMessage   → 工具的执行结果
    SystemMessage → 系统指令（通过 system_prompt 设置）
    """)


# ============================================================================
# 示例 6：执行循环最佳实践
# ============================================================================
def example_6_best_practices():
    """
    示例6：使用执行循环的最佳实践
    """
    print("\n" + "="*70)
    print("示例 6：执行循环最佳实践")
    print("="*70)

    print("""
最佳实践：

1. 获取最终答案
   final_answer = response['messages'][-1].content

2. 查看是否使用了工具
   used_tools = [
       msg.tool_calls[0]['name']
       for msg in response['messages']
       if hasattr(msg, 'tool_calls') and msg.tool_calls
   ]

3. 流式输出用于用户体验
   for chunk in agent.stream(input):
       # 实时显示给用户

4. 调试时查看完整历史
   for msg in response['messages']:
       print(msg)

5. 错误处理
   try:
       response = agent.invoke(input)
   except Exception as e:
       print(f"Agent 执行错误: {e}")
    """)

    # 实际示例
    print("\n实际示例：")
    agent = create_agent(model=model, tools=[calculator])

    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": "5 加 3"}]
        })

        # 获取最终答案
        final_answer = response['messages'][-1].content
        print(f"最终答案: {final_answer}")

        # 查看使用的工具
        used_tools = [
            msg.tool_calls[0]['name']
            for msg in response['messages']
            if hasattr(msg, 'tool_calls') and msg.tool_calls
        ]
        print(f"使用的工具: {used_tools}")

    except Exception as e:
        print(f"错误: {e}")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - Agent 执行循环")
    print("="*70)

    try:
        example_1_understand_loop()
        input("\n按 Enter 继续...")

        example_2_streaming()
        input("\n按 Enter 继续...")

        example_3_multi_step()
        input("\n按 Enter 继续...")

        example_4_inspect_state()
        input("\n按 Enter 继续...")

        example_5_message_types()
        input("\n按 Enter 继续...")

        example_6_best_practices()

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  Agent 执行循环：问题 → 工具调用 → 结果 → 答案")
        print("  messages 记录完整历史")
        print("  stream() 用于实时输出")
        print("  理解 HumanMessage、AIMessage、ToolMessage")
        print("\n阶段一（基础）完成！")
        print("  已学习：模型调用、提示词、消息、工具、Agent")
        print("\n下一阶段：")
        print("  phase2_intermediate - 内存、中间件、结构化输出")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
