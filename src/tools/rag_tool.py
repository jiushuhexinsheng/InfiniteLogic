"""
RAG 检索工具 / RAG retrieval tool.

三段检索 / Three-stage retrieval:
    1. 并行召回 / Parallel recall:
       - Qdrant 向量召回 top K_retrieval / Qdrant vector recall
       - BM25 关键词召回 top K_bm25（如启用）/ BM25 keyword recall (if enabled)
    2. RRF 融合 / Reciprocal Rank Fusion
    3. CrossEncoder 精排 top K_rerank（如启用）/ CrossEncoder rerank (if enabled)

热切换设计 / Hot-swap design:
    向量库、BM25、Reranker 各占一个独立的 lru_cache。
    `reload_reranker` 仅清 reranker 缓存，不重载向量库（不卡）。
    Vector store, BM25, and reranker each own a separate cache.
    `reload_reranker` invalidates only the reranker cache (no reload of
    the heavy vector index).
"""
from functools import lru_cache
from typing import Any, Optional

from src.config import settings
# 引入 vectorstore 模块（不是 from import）便于热切换 / mocking。
# Import as a module to ease hot-swap and mocking.
from src.rag import vectorstore as vs
from src.rag.bm25_index import BM25Index, load_bm25_index
from src.rag.document import Document
from src.rag.fusion import rrf_fuse
from src.rag.reranker import get_reranker
from src.tools.base import tool


# ────────────────────────────────────────────────────────────────────
# 独立缓存 / Independent caches
#
# 每个组件单独缓存，互不影响。
# Each component caches independently for fine-grained hot-swap.
# ────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _vs_ready() -> bool:
    """
    向量库是否已建索引（缓存结果）。
    Cached availability check for the vector store.
    """
    return vs.index_exists()


@lru_cache(maxsize=1)
def _get_bm25() -> Optional[BM25Index]:
    """
    BM25 索引懒加载缓存 / BM25 index cache.

    关闭混合检索时返回 None；search_docs 据此跳过 BM25 路径。
    Returns None when hybrid is off → search_docs skips BM25.
    """
    if not settings.rag_hybrid_enabled:
        return None
    return load_bm25_index()


@lru_cache(maxsize=1)
def _get_reranker() -> Any | None:
    """
    Reranker 缓存（独立于向量/BM25）。
    Reranker cache (decoupled from vector / BM25).

    延迟加载：get_reranker() 内部惰性构造 CrossEncoder，此处仅校验开关。
    Lazy loading: get_reranker() defers CrossEncoder init; here we only check toggle.
    """
    if not settings.rag_reranker_enabled:
        return None
    return get_reranker()


def _reset_vs_cache() -> None:
    """ingest 完成后调用，让运行中的 search_docs 看到新索引。"""
    _vs_ready.cache_clear()


def _reset_bm25_cache() -> None:
    _get_bm25.cache_clear()


def _reset_reranker_cache() -> None:
    """清 Reranker 缓存（模型引用 + lru）。"""
    get_reranker().reset()
    _get_reranker.cache_clear()


# ────────────────────────────────────────────────────────────────────
# 热切换 API / Hot-swap API
# ────────────────────────────────────────────────────────────────────
def reload_reranker(
    model_name: str | None = None, enabled: bool | None = None
) -> str:
    """
    运行时切换 reranker，仅重载 reranker 缓存。
    Hot-swap reranker; only the reranker cache is invalidated.

    Args:
        model_name: 新 HF 模型名；None 表示不改 / new HF name; None to keep
        enabled:    True/False 开关；None 表示不改 / on/off; None to keep
    """
    if model_name is not None:
        settings.rag_reranker_model = model_name
        get_reranker().switch_model(model_name)
    if enabled is not None:
        settings.rag_reranker_enabled = enabled
    # 关键：只清 reranker 缓存。向量库 + BM25 不动。
    # Critical: clear only reranker cache. Vector + BM25 untouched.
    _reset_reranker_cache()
    state = "ON" if settings.rag_reranker_enabled else "OFF"
    return f"Reranker {state} | model: {settings.rag_reranker_model}"


def reload_hybrid(enabled: bool) -> str:
    """切换混合检索，仅重载 BM25 缓存。"""
    settings.rag_hybrid_enabled = enabled
    _reset_bm25_cache()
    return f"Hybrid retrieval {'ON' if enabled else 'OFF'}"


def reload_vectorstore() -> str:
    """
    重新检测向量库可用性（ingest 后 CLI 可调用）。
    Re-check vector store availability (call after ingest).

    顺带清 BM25 缓存：新 chunk 可能涉及新 sha1 → BM25 也变了。
    Also clear BM25 cache: new chunks may have updated BM25.
    """
    _reset_vs_cache()
    _reset_bm25_cache()
    return f"Vector store {'ready' if vs.index_exists() else 'empty'}"


# ────────────────────────────────────────────────────────────────────
# 工具函数 / Tool function
# ────────────────────────────────────────────────────────────────────
@tool("Search the local knowledge base. Use FIRST before web_search when the question may be answered by indexed docs.")
def search_docs(query: str) -> str:
    """Search the local knowledge base."""
    # 索引不存在则直接提示 / Bail out early if no index.
    if not _vs_ready():
        return "Knowledge base is empty. Run `python ingest.py` to ingest documents first."

    # ----- 1. 多路召回 / Multi-source recall -----
    # 向量召回：Qdrant 余弦相似度 / Vector recall: Qdrant cosine.
    vector_hits: list[Document] = vs.similarity_search(
        query, k=settings.rag_top_k_retrieval
    )

    # BM25 召回（如启用混合）/ BM25 recall (if hybrid).
    bm25_hits: list[Document] = []
    bm25 = _get_bm25()
    if bm25 is not None:
        # BM25.search 返回 (id, score, doc)；这里只要 doc。
        # BM25 returns (id, score, doc) triples; keep only docs.
        bm25_hits = [doc for _, _, doc in bm25.search(query, k=settings.rag_bm25_top_k)]

    # 两路都空说明知识库里真没相关内容。
    # Both empty → no relevant docs at all.
    if not vector_hits and not bm25_hits:
        return "No relevant documents found in the knowledge base."

    # ----- 2. RRF 融合 / Fuse rankings -----
    # 有 BM25 结果就融合；纯向量场景退回 vector_hits。
    # If BM25 available, fuse both. Otherwise use vector only.
    if bm25_hits:
        candidates = rrf_fuse([vector_hits, bm25_hits], k=settings.rag_rrf_k)
    else:
        candidates = vector_hits

    # ----- 3. 精排 / Rerank -----
    reranker = _get_reranker()
    if reranker is not None and candidates:
        # Reranker.rerank 返回 (passage_text, score) 列表，按分数降序。
        # Reranker.rerank returns (passage_text, score) sorted by score desc.
        passage_texts = [doc.page_content for doc in candidates]
        ranked = reranker.rerank(query, passage_texts, top_k=settings.rag_top_k_rerank)
        # 用 passage_text 反查原始 Document，保持 metadata 不丢。
        # Reverse-lookup original Document by passage_text to preserve metadata.
        doc_map = {d.page_content: d for d in candidates}
        candidates = [doc_map[text] for text, _ in ranked if text in doc_map]
    else:
        # 关闭精排时直接截断粗排前 top_k_rerank 条。
        # Reranker off: just truncate the fused list.
        candidates = candidates[: settings.rag_top_k_rerank]

    # ----- 4. 格式化输出 / Format output -----
    # "[i] source (p.X)\n content" 形式，便于 LLM 引用。
    # "[i] source (p.X)\n content" — easy for the LLM to cite.
    parts = []
    for i, doc in enumerate(candidates, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        loc = f"{source}" + (f" (p.{page})" if page else "")
        parts.append(f"[{i}] {loc}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)