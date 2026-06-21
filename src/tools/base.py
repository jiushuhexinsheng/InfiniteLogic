"""
工具装饰器与注册中心 / Tool decorator + registry.

设计思路 / Design:
    @tool 包装一个普通函数，自动从函数签名 + 类型注解推导 OpenAI 工具 schema。
    The @tool decorator wraps a plain function and infers the OpenAI tool
    schema from its signature + type hints.

    全部工具注册到模块级 TOOLS 实例，Agent 通过 TOOLS.schemas() 拿到列表，
    通过 TOOLS.acall(name, args_dict) 异步执行。
    All tools register into a module-level TOOLS singleton. The Agent reads
    via TOOLS.schemas() and executes via TOOLS.acall().

异常兜底 / Exception safety:
    工具内部抛异常会被捕获并转字符串返回，确保 ReAct 循环不中断。
    Exceptions inside a tool are caught and converted to strings so the
    ReAct loop is not interrupted.

为何自己写而不用 LangChain 的 @tool / Why roll our own:
    - LangChain @tool 依赖 Pydantic Field + 大量元类魔法；本实现 100 行可读
      LangChain's @tool relies on Pydantic Field + metaclass magic; ours is 100 readable lines
    - 零依赖（除 pydantic 用于校验）
      Zero deps beyond pydantic (already used elsewhere)
"""
from __future__ import annotations

# asyncio: 检测协程 + to_thread / asyncio: detect coroutines + run in thread.
import asyncio

# inspect: 反射函数签名 / inspect: reflect function signature.
import inspect
from collections.abc import Callable

# typing: 类型注解工具 / Typing utilities.
from typing import Any, get_type_hints

# Callable 类型别名：所有工具都形如 (...) -> Any → str。
# Type alias for tool callables: any signature returning Any (coerced to str).
ToolFunc = Callable[..., Any]

# pydantic 在此文件其实未直接用，但保留 import 提示未来可能扩展 schema 推导。
# Pydantic not strictly used here; kept for future Pydantic-based schema generation.
from pydantic import BaseModel, Field, create_model  # noqa: F401


class _ToolRegistry:
    """
    工具注册中心 / Tool registry.

    存储结构 / Storage:
        self._tools: dict[name, {"func": callable, "schema": dict}]
    """

    def __init__(self) -> None:
        # 名称 → {func, schema} 的字典；线程内单线程不需要锁。
        # Name → record dict; no lock needed for single-threaded init.
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, func: ToolFunc, schema: dict[str, Any]) -> None:
        """
        注册工具（同名覆盖）/ Register a tool (overwrites on name collision).

        开发期允许覆盖；生产前应改成报错以避免静默替换。
        Useful in dev; consider raising in prod to catch silent overrides.
        """
        self._tools[name] = {"func": func, "schema": schema}

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI 协议的 tools 数组 / OpenAI-protocol tools array."""
        return [t["schema"] for t in self._tools.values()]

    def has(self, name: str) -> bool:
        """便于测试断言 / Convenience for tests / assertions."""
        return name in self._tools

    def call(self, name: str, args: dict[str, Any]) -> str:
        """
        同步执行工具，捕获异常转字符串。
        Sync invocation with exception → string conversion.

        异步函数不该走同步路径；返回错误提示。
        Async functions must use acall(); we report an error here.
        """
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        func = self._tools[name]["func"]
        try:
            # iscoroutinefunction 检测是否 async def。
            # iscoroutinefunction detects `async def` declarations.
            if asyncio.iscoroutinefunction(func):
                return f"Error: '{name}' is async; use acall()"
            # **args 展开 dict 为关键字参数 / Spread dict into kwargs.
            result = func(**args)
            return _to_string(result)
        except PermissionError as exc:
            # 权限错误：返回 message 本身（已经语义清晰，无需前缀）。
            # PermissionError: message is self-explanatory.
            return str(exc)
        except Exception as exc:
            # 其他异常：附带工具名便于排查。
            # Other errors: prefix with tool name for debugging.
            return f"Error in {name}: {exc}"

    async def acall(self, name: str, args: dict[str, Any]) -> str:
        """
        异步执行工具（异步函数走 await，同步函数线程池）。
        Async invocation: await coroutines, run sync funcs in a thread pool.

        why to_thread / 为何用 to_thread:
            同步工具可能跑长任务（subprocess、HTTP、文件 IO）；
            直接调会阻塞事件循环。to_thread 把它移到线程池。
            Sync tools may block (subprocess, HTTP, file IO); to_thread
            offloads to a thread pool so the event loop stays responsive.
        """
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        func = self._tools[name]["func"]
        try:
            if asyncio.iscoroutinefunction(func):
                # 协程：直接 await。
                # Coroutine: await directly.
                result = await func(**args)
            else:
                # 同步函数：用 lambda 包一层让 to_thread 能传 kwargs。
                # Sync: wrap in lambda so to_thread can pass kwargs.
                result = await asyncio.to_thread(lambda: func(**args))
            return _to_string(result)
        except PermissionError as exc:
            return str(exc)
        except Exception as exc:
            return f"Error in {name}: {exc}"


# 单例 / Singleton.
TOOLS = _ToolRegistry()


def _to_string(value: Any) -> str:
    """
    把任意返回值转字符串 / Coerce any return value to a string.

    OpenAI 协议要求 tool message.content 是字符串。
    OpenAI protocol requires tool message.content to be a string.
    """
    if isinstance(value, str):
        return value
    return str(value)


def _python_type_to_json(py_type: Any) -> dict[str, Any]:
    """
    粗糙的 Python → JSON Schema 类型映射。
    Crude Python type → JSON Schema mapping.

    支持基本类型；其他类型回退 "string"。
    Supports primitive types; falls back to "string" otherwise.

    扩展点 / Extension points:
        - 可加 list[X] / dict[K,V] 解析（typing.get_origin）
        - 可加 Pydantic BaseModel 嵌套 schema
    """
    mapping: dict[type, dict[str, Any]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    # 未匹配到的类型默认 string；LLM 通常仍能正确填值。
    # Unmatched types default to string; LLMs usually still fill correctly.
    return mapping.get(py_type, {"type": "string"})


def _build_schema(func: ToolFunc, description: str) -> dict[str, Any]:
    """
    从函数签名生成 OpenAI tool schema。
    Build the OpenAI tool schema from a function signature.

    步骤 / Steps:
      1. inspect.signature 拿参数对象 / Read params via inspect.signature.
      2. typing.get_type_hints 拿类型注解（解析字符串注解）
         Pull type hints (resolves stringified annotations from PEP 604).
      3. 拼出 JSON Schema / Compose JSON Schema.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        # 跳过 self / cls（虽然工具一般不是方法）。
        # Skip self / cls for safety (tools are usually plain funcs).
        if name in ("self", "cls"):
            continue
        # 拿类型注解；缺失 fallback str。
        # Fetch type hint; default to str if missing.
        py_type = hints.get(name, str)
        prop = _python_type_to_json(py_type)
        # 默认值缺失 → 必填字段 / No default → required.
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            # JSON Schema 用 default 字段表达可选 + 默认值。
            # JSON Schema uses `default` for optional + default value.
            prop["default"] = param.default
        properties[name] = prop

    # OpenAI tool schema 标准结构。
    # OpenAI tool schema canonical shape.
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def tool(description: str | None = None) -> Callable[[ToolFunc], ToolFunc]:
    """
    工具装饰器 / Tool decorator.

    用法 / Usage:
        @tool("Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

    若 description 省略，自动用函数 docstring 第一行。
    If description is omitted, the first line of the docstring is used.
    """

    def decorator(func: ToolFunc) -> ToolFunc:
        # 描述兜底链：参数 > docstring 首行 > 函数名 / Fallback chain for description.
        desc = description
        if desc is None:
            doc = (func.__doc__ or "").strip()
            desc = doc.splitlines()[0] if doc else func.__name__
        schema = _build_schema(func, desc)
        # 注册到全局 TOOLS / Register with global TOOLS.
        TOOLS.register(func.__name__, func, schema)
        # 返回原函数（装饰器不改变函数本身）。
        # Return original function unchanged.
        return func

    return decorator