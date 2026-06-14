"""
pytest 配置。

注入 LLM_API_KEY=test-key，单测不调用真实 LLM。
只覆盖纯函数：calculator / splitter / fusion / session / usage / tool_schema
"""

import os
import sys

import pytest

# 确保 src 在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 注入测试用 API key（避免 settings 加载时报错）
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("WORKSPACE_DIR", "./test_workspace")
