"""
最小 Document 模型 / Minimal Document model.

替代 LangChain 的 Document，仅含 page_content + metadata。
A drop-in replacement for LangChain's Document with just content + metadata.

为什么自己定义 / Why our own:
    - LangChain 的 Document 带一堆从未用到的字段（type / id / lc_metadata 等）
      LangChain's Document carries unused fields (type / id / lc_metadata / ...)
    - 自定义 dataclass 5 行搞定，零依赖
      5-line dataclass, zero dependencies
"""
# dataclass: 自动生成 __init__ / __repr__ / __eq__；省去样板代码。
# dataclass: auto-generates __init__ / __repr__ / __eq__; saves boilerplate.
# field: 给字段设默认工厂（如 dict / list 等可变类型必须用 default_factory）。
# field: defaults via factory (mutable types must use default_factory).
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """
    文档/chunk 的数据载体 / Data carrier for a doc or chunk.

    字段 / Fields:
        page_content: 文本内容 / The raw text
        metadata: 任意元数据 dict（来源、页码、sha1 等）
                  Arbitrary metadata dict (source / page / sha1 / ...)
    """
    # 必填字符串 / Required string field.
    page_content: str
    # 可变类型默认值必须用 default_factory，否则所有实例会共享同一 dict。
    # Mutable defaults MUST use default_factory; otherwise all instances share one dict.
    metadata: dict[str, Any] = field(default_factory=dict)