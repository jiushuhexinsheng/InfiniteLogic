"""
Prometheus 指标 / Prometheus metrics.

埋点 / Instrumented signals:
    openbase_turns_total{status}            一轮对话总数（done / error）
                                            Turn count by terminal status
    openbase_turn_latency_seconds           单轮端到端耗时直方图
                                            End-to-end turn latency histogram
    openbase_tool_calls_total{name,status}  工具调用次数（ok / error）
                                            Tool call count by name + status
    openbase_tool_latency_seconds{name}     工具调用耗时直方图
                                            Tool call latency histogram
    openbase_tokens_total{kind,model}       累计 token 计数（prompt/completion/reasoning）
                                            Cumulative token counts
    openbase_cost_usd_total{model}          累计估算成本（USD）
                                            Cumulative estimated cost
    openbase_llm_errors_total{kind}         LLM 调用错误计数
                                            LLM error count by kind

公开 / Exposure:
    FastAPI 应用挂 /metrics（src/web.py 自动注册）。
    Mounted at /metrics by src/web.py.

术语 / Terminology:
    - Counter: 单调递增计数器 / monotonically increasing counter
    - Histogram: 直方图，自动产出 _count / _sum / _bucket
                 histogram: emits _count / _sum / _bucket time series
    - label: 维度标签，用于切片 / dimension label for slicing

为什么不用 Gauge / Why no Gauge:
    Counter + Histogram 已经覆盖 turn / tool / token / cost / error；
    Gauge 适合表达"当前值"（如内存占用），本项目用不上。
    Counter + Histogram suffice. Gauge fits "current value" gauges
    (memory, queue depth, ...), which we don't need.
"""
from __future__ import annotations

import time
# 用 contextmanager 装饰器把"计时器"做成 with 上下文，调用方更优雅。
# contextmanager turns a generator into a `with`-usable timer.
from contextlib import contextmanager

# Counter / Histogram 来自 prometheus_client；底层是无锁原子操作。
# Counter / Histogram from prometheus_client; lock-free atomic ops underneath.
from prometheus_client import Counter, Histogram

# ─────────────────────────────────────────────────────────────────────
# 计数器 / Counters
# ─────────────────────────────────────────────────────────────────────
TURNS_TOTAL = Counter(
    "openbase_turns_total",                                  # 指标名（snake_case + 单位后缀） / Metric name
    "Total conversational turns processed by Agent.",        # 描述 / Description (shown in /metrics)
    labelnames=("status",),                                  # 维度：done | error
)

TOOL_CALLS_TOTAL = Counter(
    "openbase_tool_calls_total",
    "Total tool invocations.",
    labelnames=("name", "status"),                           # 工具名 + ok/error
)

LLM_ERRORS_TOTAL = Counter(
    "openbase_llm_errors_total",
    "LLM call errors by kind.",
    labelnames=("kind",),                                    # 异常类名（TimeoutError / HTTPStatusError / ...）
)

TOKENS_TOTAL = Counter(
    "openbase_tokens_total",
    "Cumulative token counts by kind and model.",
    labelnames=("kind", "model"),                            # prompt | completion | reasoning
)

COST_USD_TOTAL = Counter(
    "openbase_cost_usd_total",
    "Cumulative estimated cost (USD) by model.",
    labelnames=("model",),
)

# ─────────────────────────────────────────────────────────────────────
# 直方图 / Histograms
#
# buckets 必须升序；每个 bucket 上界统计 ≤ 该值的样本数。
# Buckets must be sorted ascending; each bucket counts samples ≤ its bound.
# 这些上界覆盖 0.5s 短查询到 5 分钟超时任务的范围。
# These bounds cover from <1s quick queries to 5min long tasks.
# ─────────────────────────────────────────────────────────────────────
TURN_LATENCY = Histogram(
    "openbase_turn_latency_seconds",
    "End-to-end turn latency.",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)

TOOL_LATENCY = Histogram(
    "openbase_tool_latency_seconds",
    "Tool call latency.",
    labelnames=("name",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)


# ─────────────────────────────────────────────────────────────────────
# 辅助上下文管理器 / Helper context managers
#
# @contextmanager 让 generator 函数支持 `with ... :` 语法。
# @contextmanager turns a generator into a `with`-usable resource.
# ─────────────────────────────────────────────────────────────────────
@contextmanager
def time_turn():
    """
    计时一轮 / Time one turn.

    用法 / Usage:
        with time_turn():
            ... do work ...
    """
    # time.monotonic() 是单调递增时钟，不受系统时间调整影响。
    # monotonic() is unaffected by NTP / DST adjustments.
    start = time.monotonic()
    try:
        yield
    finally:
        # finally 保证就算抛异常也会记录耗时。
        # finally ensures we record even when an exception is raised.
        TURN_LATENCY.observe(time.monotonic() - start)


@contextmanager
def time_tool(name: str):
    """
    计时一次工具调用 / Time one tool call.

    .labels(name=name) 是 label-set 绑定，每个不同的 name 是独立时序。
    .labels(name=name) binds a label set; each name is a separate series.
    """
    start = time.monotonic()
    try:
        yield
    finally:
        TOOL_LATENCY.labels(name=name).observe(time.monotonic() - start)