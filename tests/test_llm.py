"""LLM 模块测试 / LLM module tests."""
from src.llm import _accumulate_tool_calls, _build_payload, _headers


class TestBuildPayload:
    """请求体构造测试 / Payload construction tests."""

    def test_basic_payload(self):
        """基本 payload / Basic payload."""
        msgs = [{"role": "user", "content": "hello"}]
        p = _build_payload(msgs)
        assert p["model"] is not None
        assert p["messages"] == msgs
        assert p["stream"] is True
        assert "temperature" in p
        assert "max_tokens" in p

    def test_payload_with_tools(self):
        """带 tools 的 payload / Payload with tools."""
        msgs = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "search"}}]
        p = _build_payload(msgs, tools=tools)
        assert p["tools"] == tools
        assert p["tool_choice"] == "auto"

    def test_payload_without_tools(self):
        """无 tools 不附加空数组 / No tools → no tools key."""
        msgs = [{"role": "user", "content": "hi"}]
        p = _build_payload(msgs, tools=None)
        assert "tools" not in p
        assert "tool_choice" not in p

    def test_payload_with_empty_tools(self):
        """空 tools 列表不附加 / Empty tools list → no tools key."""
        msgs = [{"role": "user", "content": "hi"}]
        p = _build_payload(msgs, tools=[])
        assert "tools" not in p

    def test_stream_options_include_usage(self):
        """流式模式包含 usage / Stream mode includes usage opt-in."""
        msgs = [{"role": "user", "content": "hi"}]
        p = _build_payload(msgs, stream=True, include_usage=True)
        assert p["stream_options"] == {"include_usage": True}


class TestHeaders:
    """请求头测试 / Request headers tests."""

    def test_headers_contain_auth(self):
        """包含 Authorization 头 / Contains Authorization header."""
        h = _headers()
        assert "Authorization" in h
        assert h["Authorization"].startswith("Bearer ")
        assert h["Content-Type"] == "application/json"
        assert "text/event-stream" in h["Accept"]


class TestAccumulateToolCalls:
    """Tool call 累积测试 / Tool call accumulator tests."""

    def test_single_chunk_tool_call(self):
        """单 chunk 完整 tool_call / Single chunk complete tool_call."""
        buf: dict = {}
        delta = {
            "index": 0,
            "id": "call_abc",
            "function": {"name": "search", "arguments": '{"q":"hello"}'},
        }
        _accumulate_tool_calls(buf, delta)
        assert 0 in buf
        assert buf[0]["id"] == "call_abc"
        assert buf[0]["function"]["name"] == "search"
        assert buf[0]["function"]["arguments"] == '{"q":"hello"}'

    def test_multi_chunk_accumulation(self):
        """多 chunk 拼接 / Multi-chunk accumulation."""
        buf: dict = {}
        # chunk 1: id + name
        _accumulate_tool_calls(buf, {
            "index": 0, "id": "call_x",
            "function": {"name": "calc"},
        })
        # chunk 2: arguments
        _accumulate_tool_calls(buf, {
            "index": 0,
            "function": {"arguments": '{"expr'},  # noqa: Q001
        })
        # chunk 3: more arguments
        _accumulate_tool_calls(buf, {
            "index": 0,
            "function": {"arguments": '":"1+1"}'},
        })
        assert buf[0]["function"]["name"] == "calc"
        assert buf[0]["function"]["arguments"] == '{"expr":"1+1"}'

    def test_multiple_parallel_tool_calls(self):
        """多个并行 tool_call 按 index 分桶 / Multiple parallel calls bucketed by index."""
        buf: dict = {}
        _accumulate_tool_calls(buf, {
            "index": 0, "id": "call_a",
            "function": {"name": "search"},
        })
        _accumulate_tool_calls(buf, {
            "index": 1, "id": "call_b",
            "function": {"name": "calc"},
        })
        assert 0 in buf
        assert 1 in buf
        assert buf[0]["function"]["name"] == "search"
        assert buf[1]["function"]["name"] == "calc"
