"""
InfiniteLogic 工具注册模块。

导入所有工具模块以触发 @tool 装饰器注册到 TOOLS 单例。
"""

# 导入工具模块 → 触发注册
from src.tools import (  # noqa: F401
    calculator,
    datetime_tool,
    file_tools,
    python_exec,
    rag_tool,
    search,
)

# 导出 TOOLS 注册中心供 agent 使用
from src.tools.base import TOOLS  # noqa: F401
