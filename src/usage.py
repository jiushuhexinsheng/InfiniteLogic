"""
Token 用量与成本统计 / Token usage + cost tracking.

从 LLM 响应抓 usage 字段累计；按模型价目表估算成本。
Accumulates `usage` from LLM responses; estimates cost using a price book.

数据模型 / Data:
    UsageStats:
        prompt_tokens     — 累计 prompt token 数
        completion_tokens — 累计 completion token 数
        reasoning_tokens  — 思考模式额外 token（DeepSeek）
        cost_usd          — 估算美金成本
        requests          — 调用次数

价目表硬编码 / Price book hardcoded:
    每千 token 价格（输入 / 输出），来自厂商公开页。
    Per-1K-tokens (input / output), pulled from public pricing pages.
    未匹配的模型按 0 计 → 不阻塞，方便测试。
    Unknown models cost 0 → doesn't block tests / non-prod use.
"""
from __future__ import annotations

from dataclasses import dataclass

# 价目表（每 1K token 的 USD）。
# Pricing table: USD per 1K tokens (input, output).
# 数字来自各厂商官网；要更新只改这里。
# Numbers from each vendor's public pricing page; update here only.
_PRICES: dict[str, tuple[float, float]] = {
    # DeepSeek
    "deepseek-chat":          (0.00014, 0.00028),
    "deepseek-reasoner":      (0.00055, 0.00219),
    "deepseek-v4-flash":      (0.0001,  0.0002),
    "deepseek-v4-pro":        (0.0006,  0.0022),
    # OpenAI
    "gpt-4o-mini":            (0.00015, 0.00060),
    "gpt-4o":                 (0.0025,  0.0100),
    "gpt-4.1":                (0.0020,  0.0080),
    # Anthropic (via OpenAI-protocol proxy like LiteLLM)
    "claude-sonnet-4-6":      (0.0030,  0.0150),
    "claude-opus-4-7":        (0.0150,  0.0750),
}


def _price_for(model: str) -> tuple[float, float]:
    """
    精确匹配 → 前缀匹配 → 0。
    Exact match → prefix match → zero.

    前缀匹配处理像 "deepseek-chat-20240701" 这类带版本号的模型。
    Prefix match handles versioned IDs like "deepseek-chat-20240701".
    """
    if model in _PRICES:
        return _PRICES[model]
    for key, price in _PRICES.items():
        if model.startswith(key):
            return price
    return (0.0, 0.0)


@dataclass
class UsageStats:
    """
    累计统计 / Cumulative stats.

    用 dataclass 让字段声明清晰；默认值 0 / 0.0 由 dataclass 自动生成 __init__。
    Dataclass for clean field declarations; defaults handled automatically.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        """三类 token 之和 / Sum of all three token kinds."""
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

    def add(self, usage: dict | None, model: str) -> None:
        """
        累加单次响应的 usage 字段。
        Add one response's usage.

        OpenAI 协议 usage 字段示例 / Example shape:
            {
              "prompt_tokens": 12,
              "completion_tokens": 34,
              "total_tokens": 46,
              "completion_tokens_details": {"reasoning_tokens": 8}
            }
        """
        # None 或空字典直接跳过 / Skip on None or empty.
        if not usage:
            return
        # int() 兜底：偶尔厂商返回字符串 / 浮点 / None。
        # int() guard: vendors sometimes return strings / floats / None.
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        # reasoning_tokens 嵌套在 completion_tokens_details 下。
        # reasoning_tokens nested under completion_tokens_details.
        rt_details = usage.get("completion_tokens_details") or {}
        rt = int(rt_details.get("reasoning_tokens") or 0)

        # 累加 / Accumulate.
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.reasoning_tokens += rt
        self.requests += 1

        # 计算本次成本：prompt 按输入价、completion+reasoning 按输出价。
        # Cost: prompt at input price, (completion+reasoning) at output price.
        in_price, out_price = _price_for(model)
        cost = (pt / 1000.0) * in_price + ((ct + rt) / 1000.0) * out_price
        self.cost_usd += cost

        # 同步写 Prometheus（包 try/except 防指标系统挂掉影响业务）。
        # Mirror to Prometheus; guarded so a metrics fault never breaks agent.
        try:
            from src.metrics import COST_USD_TOTAL, TOKENS_TOTAL

            if pt:
                TOKENS_TOTAL.labels(kind="prompt", model=model).inc(pt)
            if ct:
                TOKENS_TOTAL.labels(kind="completion", model=model).inc(ct)
            if rt:
                TOKENS_TOTAL.labels(kind="reasoning", model=model).inc(rt)
            if cost:
                COST_USD_TOTAL.labels(model=model).inc(cost)
        except Exception:  # noqa: BLE001
            pass

    def format(self) -> str:
        """CLI 友好的一行展示 / One-line CLI display."""
        return (
            f"requests={self.requests}  "
            f"prompt={self.prompt_tokens}  "
            f"completion={self.completion_tokens}  "
            f"reasoning={self.reasoning_tokens}  "
            f"total={self.total_tokens}  "
            f"~${self.cost_usd:.4f}"
        )

    def reset(self) -> None:
        """重置全部计数（/usage reset 调用）/ Reset all counters."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0
        self.requests = 0


# 全局单例：模块级变量；所有调用都累计到这一份。
# Global singleton; all callers accumulate into one instance.
USAGE = UsageStats()