"""配置校验测试 / Config validation tests."""

import pytest
from pydantic import ValidationError

from src.config import Settings


class TestSettingsValidation:
    """配置类型校验 / Config type validation."""

    def test_default_values(self, monkeypatch):
        """默认值加载 / Default values load."""
        # 隔离 .env 文件，仅用环境变量 / Isolate from .env file.
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        # 清除可能存在的环境变量覆盖 / Clear any existing env overrides.
        for key in ("LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
                     "AGENT_RECURSION_LIMIT", "AGENT_MAX_HISTORY_MESSAGES"):
            monkeypatch.delenv(key, raising=False)
        s = Settings(_env_file=None)  # 跳过 .env 文件 / Skip .env file.
        assert s.llm_model == "deepseek-chat"
        assert s.llm_temperature == 0.0
        assert s.llm_max_tokens == 4096
        assert s.agent_recursion_limit == 50
        assert s.agent_max_history_messages == 80

    def test_temperature_bounds(self, monkeypatch):
        """温度必须在 [0, 2] / Temperature must be in [0, 2]."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_TEMPERATURE", "3.0")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_max_tokens_positive(self, monkeypatch):
        """max_tokens > 0 / Must be positive."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MAX_TOKENS", "0")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_recursion_limit_positive(self, monkeypatch):
        """recursion_limit > 0 / Must be positive."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_RECURSION_LIMIT", "-1")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_log_level_literal(self, monkeypatch):
        """日志级别限定枚举值 / Log level must be valid literal."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_env_override(self, monkeypatch):
        """环境变量覆盖默认值 / Env var overrides default."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("AGENT_RECURSION_LIMIT", "100")
        s = Settings(_env_file=None)
        assert s.llm_model == "gpt-4o-mini"
        assert s.agent_recursion_limit == 100

    def test_new_config_fields(self, monkeypatch):
        """P0-P2 新增配置字段有合理默认值 / New config fields have sane defaults."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        s = Settings(_env_file=None)
        # P0
        assert s.llm_retry_max >= 0
        assert s.llm_circuit_breaker_threshold > 0
        assert s.sandbox_mode in ("subprocess", "docker", "disabled")
        assert s.auth_enabled is True
        # P1
        assert s.session_wal_enabled is True
        assert s.session_ttl_days >= 0
        assert s.agent_summarize_threshold > 0
        # P2
        assert s.agent_parallel_tools is True
        assert s.cache_ttl_seconds > 0
        assert s.tracing_enabled is True
