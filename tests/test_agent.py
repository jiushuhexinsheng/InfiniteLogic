"""Agent 核心逻辑测试 / Agent core logic tests."""
from src.agent import _trim_history


class TestTrimHistory:
    """历史裁剪测试 / History trimming tests."""

    def test_no_trim_when_under_limit(self):
        """不足上限时不裁剪 / No trim when under limit."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = _trim_history(msgs, max_messages=10)
        assert len(result) == 3

    def test_trim_excess_messages(self):
        """超过上限时裁剪旧消息 / Trim oldest when over limit."""
        msgs = [
            {"role": "system", "content": "System prompt."},
        ] + [
            {"role": "user", "content": f"msg{i}"} for i in range(100)
        ]
        result = _trim_history(msgs, max_messages=50)
        # system + 49 most recent = 50
        assert len(result) == 50
        assert result[0]["role"] == "system"
        # 最新消息应保留 / Most recent should be kept.
        assert result[-1]["content"] == "msg99"

    def test_preserves_tool_call_pairing(self):
        """不切断 tool_call ↔ tool 配对 / Don't break tool_call/tool pairs."""
        msgs = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "run tool"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "search", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "assistant", "content": "answer"},
        ]
        result = _trim_history(msgs, max_messages=3)
        # 不应出现孤立的 tool 消息 / Should not have orphan tool messages.
        # 如果第一条是 tool → 已被丢弃。
        if result:
            assert result[0].get("role") != "tool"

    def test_empty_input(self):
        """空消息列表 / Empty message list."""
        result = _trim_history([], max_messages=10)
        assert result == []

    def test_only_system_message(self):
        """仅 system 消息 / Only system message."""
        msgs = [{"role": "system", "content": "Prompt."}]
        result = _trim_history(msgs, max_messages=10)
        assert len(result) == 1

    def test_multiple_system_messages(self):
        """多个连续 system 消息 / Multiple leading system messages."""
        msgs = [
            {"role": "system", "content": "Prompt 1."},
            {"role": "system", "content": "Prompt 2."},
            {"role": "user", "content": "hello"},
        ]
        result = _trim_history(msgs, max_messages=2)
        # 保留两个 system + 最近的 user / Keep both systems + recent user.
        assert len(result) <= 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "system"


class TestTrimHistoryEdgeCases:
    """边界情况 / Edge cases."""

    def test_max_messages_one(self):
        """max_messages=1 时至少保留 system / At least keep system."""
        msgs = [
            {"role": "system", "content": "Prompt."},
            {"role": "user", "content": "user1"},
            {"role": "user", "content": "user2"},
        ]
        result = _trim_history(msgs, max_messages=1)
        assert len(result) >= 1
        assert result[0]["role"] == "system"

    def test_mid_list_system_preserved(self):
        """中间出现的 system 消息不被特殊处理 / Mid-list system msgs stay in rest."""
        msgs = [
            {"role": "system", "content": "Prompt 1."},
            {"role": "user", "content": "msg1"},
            {"role": "system", "content": "Mid prompt."},  # 中间 system
            {"role": "user", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        result = _trim_history(msgs, max_messages=4)
        # 前导 system 保留，中间 system 在 tail 里会被保留或裁剪。
        assert result[0]["role"] == "system"
