"""
CrossEncoder Reranker — BGE Reranker 精排。

模型：BAAI/bge-reranker-base（~1.1GB）
输入 (query, passage) 对列表 → model.predict(pairs) → 分数数组 → 按分降序 top_k

设计 / Design:
    - 懒加载：首次调用 rerank() 时才加载 CrossEncoder（避免启动时加载 PyTorch）
    - 热切换：switch_model() 仅设置新模型名 + 清缓存，下次调用时重新加载
    - 单例：模块级 get_reranker() 共享同一实例
"""
from __future__ import annotations

from src.config import settings


class Reranker:
    """CrossEncoder 精排器。

    支持模型:
    - BAAI/bge-reranker-base（~1.1GB，默认）
    - BAAI/bge-reranker-v2-m3（~2.3GB，多语言）
    - cross-encoder/ms-marco-MiniLM-L-6-v2（~90MB，英文）
    """

    def __init__(
        self,
        model_name: str = settings.rag_reranker_model,
    ) -> None:
        """
        Args:
            model_name: HuggingFace 模型名称
        """
        self._model_name = model_name
        self._model = None  # CrossEncoder 实例，首次调用时懒加载

    def _ensure_loaded(self) -> None:
        """懒加载 CrossEncoder 模型。

        延迟 import + 懒构造：避免启动时拖入 PyTorch 等重依赖。
        Defer import + lazy init to avoid pulling in PyTorch at startup.
        """
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        # 首次加载约 5-15 秒（首次还要下载 ~1.1GB 权重到 ~/.cache/huggingface）。
        # First load takes 5-15s; first run downloads ~1.1GB weights.
        self._model = CrossEncoder(self._model_name)

    @property
    def model_name(self) -> str:
        """当前模型名 / Current model name."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载 / Whether the model has been loaded."""
        return self._model is not None

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_k: int = settings.rag_top_k_rerank,
    ) -> list[tuple[str, float]]:
        """对候选 passage 列表进行精排。

        Args:
            query: 查询文本
            passages: 候选 passage 文本列表
            top_k: 返回的 top-N 数量

        Returns:
            (passage_text, relevance_score) 列表，按分数降序
        """
        self._ensure_loaded()
        # CrossEncoder.predict 接受 (query, passage) 对列表，返回分数 ndarray。
        # CrossEncoder.predict takes (query, passage) pairs, returns scores ndarray.
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs)
        # 按分数降序取 top_k。
        # Sort by score desc, top-k.
        ranked = sorted(zip(passages, scores, strict=False), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def switch_model(self, model_name: str) -> None:
        """运行时切换 reranker 模型。

        Args:
            model_name: 新的 HuggingFace 模型名称
        """
        self._model_name = model_name
        self._model = None  # 清空旧模型缓存，下次调用时重新加载

    def reset(self) -> None:
        """释放模型缓存（热切换辅助）。

        仅清 _model 引用，不改变 _model_name。
        Clear cached model; preserves model_name for next load.
        """
        self._model = None


# 模块级单例 / Module-level singleton.
_RERANKER: Reranker | None = None


def get_reranker() -> Reranker:
    """获取/创建 Reranker 单例。

    首次调用构造实例；后续返回同一对象。
    First call creates instance; subsequent calls return the same object.
    """
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()
    return _RERANKER
