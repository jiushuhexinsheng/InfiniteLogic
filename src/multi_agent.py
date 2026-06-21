"""
多 Agent 协作（Supervisor 模式）/ Multi-agent collaboration (Supervisor pattern).

三角色 / Three roles:
    Planner    — 把用户任务拆成可执行子任务清单 / Decompose into subtasks
    Researcher — 用 search_docs + web_search 收集事实 / Gather facts via search tools
    Writer     — 综合 Planner 计划 + Researcher 资料生成最终答案
                 Synthesize final answer from plan + research

工作流 / Workflow:
    user_input
        ↓
    Planner: 输出 JSON 子任务列表
    ↓
    Researcher: 对每个子任务调工具收集证据（最多 3 轮工具循环）
    ↓
    Writer: 综合答案（流式）

每个角色是一次独立 LLM 调用，可配不同 prompt / 工具子集。
Each role is a separate LLM call with its own prompt + tool subset.

不依赖 SessionStore（一次性任务，无需持久化中间过程）。
Does not depend on SessionStore — one-shot, no intermediate persistence.

适合场景 / When to use:
    - 调研 + 综述（"调研 X 并写综述"）
    - 对比分析（"比较 A 和 B 的优缺点"）
    - 多源信息整合
    不适合简单问答（成本 3x，价值未必相称）。
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from src.llm import stream_chat
from src.logging_setup import logger
from src.prompts.multi_agent import (
    PLANNER_PROMPT,
    RESEARCHER_PROMPT,
    RESEARCHER_TOOLS,
    WRITER_PROMPT,
)
from src.tools import TOOLS


async def _llm_simple(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    一次性收集流式调用的最终 message。
    Collect the final message from a streamed call.

    Planner / Writer 不需要逐 chunk 渲染；这里把流式累积成完整 message。
    Planner / Writer don't need chunk-by-chunk rendering; we just collect.
    """
    final: dict[str, Any] | None = None
    async for event in stream_chat(messages, tools=tools):
        if event["type"] == "done":
            final = event["message"]
    if final is None:
        raise RuntimeError("LLM returned no message")
    return final


def _filtered_tool_schemas(allowed: set[str]) -> list[dict[str, Any]]:
    """从全局 TOOLS 注册中心过滤出白名单工具的 schema。
    Filter tool schemas from the global TOOLS registry by whitelist."""
    return [s for s in TOOLS.schemas() if s["function"]["name"] in allowed]


async def _planner(user_input: str) -> list[dict[str, Any]]:
    """
    规划阶段：返回子任务列表。
    Plan stage: returns subtask list.
    """
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": user_input},
    ]
    # 不带工具：Planner 只负责规划，不查资料。
    # No tools: Planner only plans, doesn't research.
    msg = await _llm_simple(messages)
    raw = (msg.get("content") or "").strip()
    # 容错：剥掉可能的 ```json ... ``` 包装（部分 LLM 会自作主张加）。
    # Tolerance: strip optional ```json fences from over-zealous LLMs.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        plan = json.loads(raw)
        if isinstance(plan, list):
            return plan
    except json.JSONDecodeError:
        pass
    # 解析失败 → 退化为单任务（直接把用户输入当作唯一子任务）。
    # Parse failure → fallback to single-task plan.
    logger.warning("planner returned unparseable output: {}", raw[:200])
    return [{"id": 1, "task": user_input, "rationale": "fallback: full task"}]


async def _researcher(plan: list[dict[str, Any]]) -> dict[str, Any]:
    """
    研究阶段 / Research stage.

    带工具循环（最多 3 轮）让 LLM 调工具收集证据。
    Multi-turn tool loop (up to 3 rounds) to collect facts via tools.

    为什么 3 轮 / Why 3 rounds:
        够拿到大部分证据；防止 LLM 无限调工具拉成本。
        Enough to gather most facts; bounds cost vs runaway tool loops.
    """
    # 序列化 plan 让 Researcher 看清完整结构。
    # Pretty-print plan so Researcher sees full structure.
    plan_str = json.dumps(plan, ensure_ascii=False, indent=2)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RESEARCHER_PROMPT},
        {"role": "user", "content": f"Subtasks:\n{plan_str}\n\nGather findings and return the JSON object."},
    ]
    tools = _filtered_tool_schemas(RESEARCHER_TOOLS)

    for round_ in range(3):
        msg = await _llm_simple(messages, tools=tools)
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        # 没工具调用 = LLM 给出最终 findings JSON，跳出循环。
        # No tool_calls = LLM produced final findings; exit loop.
        if not tool_calls:
            break
        # 执行每个 tool_call 并把结果追加为 ToolMessage。
        # Execute each tool_call and append results as ToolMessages.
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await TOOLS.acall(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    final_content = (messages[-1].get("content") or "").strip()
    # 同 Planner 解析逻辑：剥 ```json 包装。
    # Same fence-stripping logic as Planner.
    if final_content.startswith("```"):
        final_content = final_content.strip("`")
        if final_content.lower().startswith("json"):
            final_content = final_content[4:].strip()
    try:
        return json.loads(final_content)
    except json.JSONDecodeError:
        # 解析失败时把原文塞 _raw 字段返回。
        # Parse failure → return raw text under _raw key.
        return {"_raw": final_content}


async def run_multi_agent(user_input: str) -> AsyncIterator[dict[str, Any]]:
    """
    跑完整流水线：Plan → Research → Write。
    Run the full pipeline. Yields events for UI rendering.

    事件类型 / Events:
        plan_done       — Planner 完成 / Planner finished
        research_done   — Researcher 完成 / Researcher finished
        content_delta   — Writer 流式回答 / Writer streaming
        reasoning_delta — Writer 思考流（DeepSeek thinking 模式）
        error / done
    """
    logger.info("multi_agent start input_len={}", len(user_input))

    # 1. Planner
    try:
        plan = await _planner(user_input)
    except Exception as exc:  # noqa: BLE001
        logger.exception("planner failed")
        yield {"type": "error", "message": f"Planner failed: {exc}"}
        return
    yield {"type": "plan_done", "plan": plan}

    # 2. Researcher
    try:
        findings = await _researcher(plan)
    except Exception as exc:  # noqa: BLE001
        logger.exception("researcher failed")
        yield {"type": "error", "message": f"Researcher failed: {exc}"}
        return
    yield {"type": "research_done", "findings": findings}

    # 3. Writer (流式) / Writer (streaming).
    # 把 plan + findings 拼成一个大 context 给 Writer。
    # Compose plan + findings into one big context for Writer.
    context = (
        f"User task:\n{user_input}\n\n"
        f"Plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
        f"Findings:\n{json.dumps(findings, ensure_ascii=False, indent=2)}\n\n"
        "Write the final answer."
    )
    messages = [
        {"role": "system", "content": WRITER_PROMPT},
        {"role": "user", "content": context},
    ]
    try:
        # 不传 tools：Writer 只综合，不再调工具。
        # No tools: Writer just synthesizes.
        async for event in stream_chat(messages):
            if event["type"] == "content_delta":
                yield event
            elif event["type"] == "reasoning_delta":
                yield event
            elif event["type"] == "done":
                # done 事件不向外转发（流水线在外层显式 yield "done"）。
                # Don't forward done; the pipeline emits its own done below.
                pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("writer failed")
        yield {"type": "error", "message": f"Writer failed: {exc}"}
        return

    logger.info("multi_agent done")
    yield {"type": "done"}