"""
LangChain 1.0 - 生产级部署 (Production Ready)
================================================

本模块重点讲解：
1. Checkpoint 持久化 —— 状态不丢失，支持故障恢复
2. RetryPolicy —— 节点级自动重试与退避
3. 错误处理模式 —— 节点级、图级、应用级三层防护
4. 手动重试与兜底 —— 精细化控制重试耗尽后的行为
5. 生产最佳实践 —— 状态设计、步数限制、LangSmith 接入
"""

import os
import sqlite3
import sys
import time
import random
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# ============================================================================
# 环境配置
# ============================================================================

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")

# Windows 控制台 UTF-8 支持
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# 依赖检查
# ============================================================================

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import RetryPolicy, Command
except ImportError as e:
    print(f"依赖缺失: {e}")
    print("请运行: pip install langgraph langgraph-checkpoint-sqlite")
    sys.exit(1)


# ============================================================================
# 示例 1：Checkpoint 持久化 —— 状态保存与恢复
# ============================================================================

class CheckpointState(TypedDict):
    """极简状态，用于演示 checkpoint"""
    value: int


def increment_node(state: CheckpointState) -> dict:
    """节点：value + 1"""
    return {"value": state["value"] + 1}


def double_node(state: CheckpointState) -> dict:
    """节点：value * 2"""
    return {"value": state["value"] * 2}


def example_1_checkpoint_basics():
    """
    示例1：Checkpoint 持久化

    关键概念：
    - MemorySaver：内存 checkpoint，进程结束就丢失（仅开发/测试）
    - SqliteSaver：SQLite 持久化，进程重启也能恢复（生产推荐）
    - thread_id：同一 ID 的调用共享 checkpoint 历史
    """
    print("\n" + "=" * 70)
    print("示例 1：Checkpoint 持久化")
    print("=" * 70)

    # ---- 1. MemorySaver（内存，不持久）----
    print("\n[1] MemorySaver（内存 checkpoint）")
    memory = MemorySaver()
    builder = StateGraph(CheckpointState)
    builder.add_node("increment", increment_node)
    builder.add_node("double", double_node)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", "double")
    builder.add_edge("double", END)
    graph_mem = builder.compile(checkpointer=memory)

    config_mem = {"configurable": {"thread_id": "mem-demo"}}
    result = graph_mem.invoke({"value": 1}, config_mem)
    print(f"  输入: value=1")
    print(f"  流程: 1 + 1 = 2, 2 * 2 = 4")
    print(f"  结果: value={result['value']}")

    # 检查 checkpoint 中保存的状态历史
    state_mem = graph_mem.get_state(config_mem)
    print(f"  checkpoint 记录数: {len(state_mem.tasks)}")
    print(f"  说明: 内存 checkpoint，进程结束即丢失")

    # ---- 2. SqliteSaver（持久化）----
    print("\n[2] SqliteSaver（SQLite 持久化）")
    # :memory: 表示内存中的 SQLite（演示用），生产环境用文件路径如 "app.db"
    # SqliteSaver 需要传入 sqlite3.Connection 对象
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    sqlite = SqliteSaver(conn)
    graph_sql = builder.compile(checkpointer=sqlite)

    config_sql = {"configurable": {"thread_id": "sqlite-demo"}}
    result2 = graph_sql.invoke({"value": 3}, config_sql)
    print(f"  输入: value=3")
    print(f"  流程: 3 + 1 = 4, 4 * 2 = 8")
    print(f"  结果: value={result2['value']}")

    # 验证 checkpoint 是否保存
    state_sql = graph_sql.get_state(config_sql)
    print(f"  checkpoint 已保存到 SQLite")
    print(f"  最终状态: value={state_sql.values['value']}")

    # ---- 3. 生产场景说明 ----
    print("\n[3] 为什么生产环境必须持久化 checkpoint？")
    print("  - 服务器重启/崩溃 → 用户对话状态不丢失")
    print("  - 负载均衡多实例 → 请求路由到任意实例都能恢复")
    print("  - HITL 中断后 → 用户过一会儿回来还能继续")
    print("\n  生产建议: 用 PostgresSaver 替代 SqliteSaver")


# ============================================================================
# 示例 2：RetryPolicy —— 节点级自动重试
# ============================================================================

class RetryState(TypedDict):
    result: str


# 全局计数器，模拟不稳定 API
_api_call_count = 0


def flaky_api_node(state: RetryState) -> dict:
    """模拟不稳定的外部 API：前 2 次调用失败，第 3 次成功"""
    global _api_call_count
    _api_call_count += 1
    if _api_call_count < 3:
        # ConnectionError 属于 RetryPolicy 默认会重试的异常类型
        raise ConnectionError(f"模拟网络错误 (第 {_api_call_count} 次)")
    return {"result": f"API 调用成功（尝试了 {_api_call_count} 次）"}


def example_2_retry_policy():
    """
    示例2：RetryPolicy 自动重试

    关键概念：
    - RetryPolicy 附加在节点上，节点失败时自动重试
    - max_attempts：总尝试次数（含第一次），不是重试次数
    - backoff_factor：指数退避乘数，避免雪崩
    - retry_on：指定哪些异常才重试（不配置则使用默认规则）
    """
    print("\n" + "=" * 70)
    print("示例 2：RetryPolicy 自动重试")
    print("=" * 70)

    global _api_call_count
    _api_call_count = 0

    builder = StateGraph(RetryState)
    # retry_policy 参数：节点失败时的自动重试策略
    builder.add_node(
        "api",
        flaky_api_node,
        retry_policy=RetryPolicy(
            max_attempts=5,           # 最多尝试 5 次（1次正常 + 4次重试）
            initial_interval=0.1,     # 首次重试前等待 0.1 秒
            backoff_factor=2.0,       # 每次等待时间翻倍（0.1, 0.2, 0.4...）
            max_interval=1.0,         # 最长等待不超过 1 秒
            jitter=True,              # 加入随机抖动，避免所有节点同时重试
        ),
    )
    builder.add_edge(START, "api")
    builder.add_edge("api", END)

    graph = builder.compile()

    print("\n  模拟: API 节点前 2 次调用失败，第 3 次成功")
    print(f"  重试策略: max_attempts=5, backoff_factor=2.0")

    result = graph.invoke({"result": ""})

    print(f"\n  结果: {result['result']}")
    print("\n关键点：")
    print("  - RetryPolicy 只负责'自动重试'，不负责'重试耗尽后去哪'")
    print("  - 如果 5 次都失败，异常会继续抛出，需要外层捕获")
    print("  - 适合处理: 网络抖动、API 限流、临时不可用")


# ============================================================================
# 示例 3：节点级错误处理 + 图级条件路由
# ============================================================================

class ErrorState(TypedDict):
    value: int
    error: str
    error_count: int


def risky_compute(state: ErrorState) -> dict:
    """
    节点级错误处理：捕获预期异常，返回错误状态。
    不把错误抛出去——而是写进 State，让图来决定下一步。
    """
    try:
        # 模拟：value 为偶数时成功，奇数时失败
        if state["value"] % 2 != 0:
            raise ValueError("奇数不允许执行")
        return {
            "value": state["value"] * 10,
            "error": "",
        }
    except ValueError as e:
        # 不抛异常，而是把错误信息写回 State
        # 这样图可以继续执行，而不是直接崩溃
        return {
            "error": str(e),
            "error_count": state.get("error_count", 0) + 1,
        }


def fallback_node(state: ErrorState) -> dict:
    """兜底节点：当主逻辑反复失败时执行"""
    return {
        "value": 0,
        "error": f"已兜底处理（累计错误 {state.get('error_count', 0)} 次）",
    }


def route_by_error(state: ErrorState) -> str:
    """路由函数：根据错误状态决定下一步"""
    if state.get("error_count", 0) >= 2:
        return "fallback"  # 错误太多，走兜底
    if state.get("error"):
        return "risky"     # 还有重试机会，回去重试
    return "success"       # 没有错误，流程结束


def example_3_error_handling():
    """
    示例3：节点级错误处理 + 图级路由

    三层防护理念：
    1. 节点层：捕获已知异常，转为状态字段（不让图崩溃）
    2. 图层：条件边根据错误状态选择重试或兜底
    3. 应用层：全局捕获未预期异常，记录日志/报警
    """
    print("\n" + "=" * 70)
    print("示例 3：节点级错误处理 + 图级条件路由")
    print("=" * 70)

    builder = StateGraph(ErrorState)
    builder.add_node("risky", risky_compute)
    builder.add_node("fallback", fallback_node)
    builder.add_node("success", lambda s: {"value": s["value"], "error": ""})

    builder.add_edge(START, "risky")
    # 条件路由：根据错误状态决定走向
    builder.add_conditional_edges(
        "risky",
        route_by_error,
        {"risky": "risky", "fallback": "fallback", "success": "success"},
    )
    builder.add_edge("fallback", END)
    builder.add_edge("success", END)

    graph = builder.compile()

    # 场景 A：偶数，直接成功
    print("\n[场景 A] 输入 value=2（偶数，直接成功）")
    result_a = graph.invoke({"value": 2, "error": "", "error_count": 0})
    print(f"  结果: value={result_a['value']}, error='{result_a['error']}'")

    # 场景 B：奇数，触发错误，但由于 error_count < 2 会循环回 risky
    # 但 value 没变，所以还是会失败。演示需要改 value...
    # 为了演示循环，我们做一个会自我修复的节点
    print("\n[场景 B] 输入 value=3（奇数，触发错误处理）")
    # 由于路由会回到 risky，而 value 不变，会死循环。
    # 实际生产中会结合 input() 让用户修改，或自动修正。
    # 这里演示单次错误即 fallback 的情况
    result_b = graph.invoke({"value": 3, "error": "", "error_count": 2})
    print(f"  结果: value={result_b['value']}, error='{result_b['error']}'")

    print("\n关键点：")
    print("  - 节点内只 catch 你'知道怎么处理'的异常")
    print("  - 未知异常（TypeError/KeyError）应直接抛出，便于排查")
    print("  - 用 State 传递错误信息，条件边做路由决策")


# ============================================================================
# 示例 4：手动重试 + 指数退避 + 耗尽兜底
# ============================================================================

class ManualRetryState(TypedDict):
    data: str
    attempts: int
    error: str


def api_with_manual_retry(state: ManualRetryState) -> dict:
    """
    手动重试：节点内自己控制重试逻辑。

    适用场景：
    - 需要自定义退避策略（如根据响应头调整）
    - 重试耗尽后需要执行特定清理逻辑
    - 需要记录每次重试的详细日志
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            # 模拟：50% 概率失败，但最后一次强制成功
            if random.random() < 0.5 and attempt < max_attempts:
                raise TimeoutError(f"连接超时 (attempt {attempt})")
            return {
                "data": f"成功（手动重试 {attempt} 次）",
                "attempts": attempt,
                "error": "",
            }
        except TimeoutError:
            # 指数退避：0.1s, 0.2s, 0.4s...
            sleep_time = 0.1 * (2 ** (attempt - 1))
            print(f"    第 {attempt} 次失败，退避 {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    # 所有尝试都失败
    return {
        "data": "",
        "attempts": max_attempts,
        "error": "max_retries_exceeded",
    }


def fallback_manual(state: ManualRetryState) -> dict:
    """手动重试的兜底节点"""
    return {
        "data": "兜底数据（使用缓存/默认值）",
        "error": f"主逻辑失败，已兜底（尝试 {state['attempts']} 次）",
    }


def route_manual(state: ManualRetryState) -> str:
    """根据手动重试结果路由"""
    if state.get("error") == "max_retries_exceeded":
        return "fallback"
    return END


def example_4_manual_retry():
    """
    示例4：手动重试与退避策略

    RetryPolicy vs 手动重试：
    - RetryPolicy：简单，自动，但耗尽后只能抛异常
    - 手动重试：灵活，可自定义逻辑，耗尽后可优雅降级
    """
    print("\n" + "=" * 70)
    print("示例 4：手动重试 + 指数退避")
    print("=" * 70)

    builder = StateGraph(ManualRetryState)
    builder.add_node("call_api", api_with_manual_retry)
    builder.add_node("fallback", fallback_manual)

    builder.add_edge(START, "call_api")
    builder.add_conditional_edges("call_api", route_manual, {"fallback": "fallback", END: END})
    builder.add_edge("fallback", END)

    graph = builder.compile()

    print("\n  模拟: API 50% 概率超时，最多重试 3 次")
    result = graph.invoke({"data": "", "attempts": 0, "error": ""})

    print(f"\n  最终数据: {result['data']}")
    print(f"  尝试次数: {result['attempts']}")
    if result.get("error"):
        print(f"  错误状态: {result['error']}")

    print("\n关键点：")
    print("  - 手动重试适合需要'精细控制'的场景")
    print("  - 指数退避避免对下游服务造成冲击")
    print("  - 重试耗尽后返回 error 标记，条件边路由到兜底")


# ============================================================================
# 示例 5：生产最佳实践 —— 状态设计、步数限制、LangSmith
# ============================================================================

class SafeState(TypedDict):
    """生产级状态设计示例"""
    query: str
    result: str
    step_count: int          # 步数计数器，防止无限循环
    errors: list             # 错误历史，便于排查


def safe_process(state: SafeState) -> dict:
    """带步数限制的处理节点"""
    step = state.get("step_count", 0) + 1
    if step > 5:
        # 步数超限，强制终止循环
        return {
            "result": "[ERROR] 步数超限，强制终止",
            "step_count": step,
            "errors": state.get("errors", []) + ["step_limit_exceeded"],
        }
    return {
        "result": f"处理完成（第 {step} 步）",
        "step_count": step,
    }


def route_loop_or_end(state: SafeState) -> str:
    """循环路由，但受 step_count 限制"""
    if state.get("step_count", 0) >= 3:
        return END
    return "process"


def example_5_production_best_practices():
    """
    示例5：生产最佳实践

    涵盖内容：
    1. 小型状态：只传必要字段，减少序列化开销
    2. 步数限制：防止逻辑错误导致无限循环
    3. 错误聚合：记录完整错误历史，不只是最后一次
    4. LangSmith 接入：一行配置开启全链路追踪
    """
    print("\n" + "=" * 70)
    print("示例 5：生产最佳实践")
    print("=" * 70)

    # ---- 1. 步数限制 ----
    print("\n[1] 步数限制 —— 防止无限循环")
    builder = StateGraph(SafeState)
    builder.add_node("process", safe_process)
    builder.add_edge(START, "process")
    # 条件边：前 3 次循环，之后结束
    builder.add_conditional_edges("process", route_loop_or_end, {END: END, "process": "process"})

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    result = graph.invoke(
        {"query": "test", "result": "", "step_count": 0, "errors": []},
        {"configurable": {"thread_id": "safe-demo"}},
    )
    print(f"  最终结果: {result['result']}")
    print(f"  执行步数: {result['step_count']}")
    print(f"  错误历史: {result.get('errors', [])}")

    # ---- 2. LangSmith 接入说明 ----
    print("\n[2] LangSmith 可观测性接入")
    print("  环境变量配置（在 .env 文件中）：")
    print("    LANGCHAIN_TRACING_V2=true")
    print("    LANGCHAIN_API_KEY=your-api-key")
    print("    LANGCHAIN_PROJECT=production-agents")
    print("\n  代码中无需改动，LangGraph 会自动上报追踪数据")
    print("  追踪内容：节点输入输出、执行耗时、Token 消耗、异常堆栈")

    # ---- 3. 检查清单 ----
    print("\n[3] 生产部署检查清单")
    checklist = [
        ("持久化 Checkpoint", "使用 PostgresSaver / SqliteSaver，不用 MemorySaver"),
        ("线程 ID 管理", "确保同一用户的对话使用一致的 thread_id"),
        ("节点重试策略", "为外部 API 调用配置 RetryPolicy"),
        ("错误分级处理", "预期错误 → State / 未知错误 → 抛出报警"),
        ("步数/Token 限制", "防止循环和超长上下文导致成本失控"),
        ("可观测性", "接入 LangSmith，配置关键指标告警"),
        (" Secrets 管理", "API Key 放环境变量，绝不硬编码"),
    ]
    for item, desc in checklist:
        print(f"  {'[OK]' if '不' not in desc[:10] else '[CHECK]'} {item}: {desc}")

    print("\n关键点：")
    print("  - 生产环境 '先防崩，再优化'——checkpoint 和错误处理是第一优先级")
    print("  - LangSmith 不是可选项，是生产排障的必备工具")


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - 生产级部署 (Production Ready)")
    print("=" * 70)

    try:
        example_1_checkpoint_basics()
        input("\n按 Enter 继续...")

        example_2_retry_policy()
        input("\n按 Enter 继续...")

        example_3_error_handling()
        input("\n按 Enter 继续...")

        example_4_manual_retry()
        input("\n按 Enter 继续...")

        example_5_production_best_practices()

        print("\n" + "=" * 70)
        print(" 完成！")
        print("=" * 70)
        print("\n核心要点：")
        print("  ✅ Checkpoint 持久化是生产部署的底线要求")
        print("  ✅ RetryPolicy 处理瞬态故障，手动重试处理复杂降级")
        print("  ✅ 错误处理三层：节点捕获 → 图路由 → 应用报警")
        print("  ✅ 步数限制和 Token 监控防止成本失控")
        print("  ✅ LangSmith 提供全链路可观测性")
        print("\n下一步：")
        print("  phase4_projects - 用所学知识构建完整项目")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
