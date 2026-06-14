"""
BM25 关键词索引 / BM25 keyword index.

与 Qdrant 向量召回互补：
Complements Qdrant vector recall:
    - 向量擅长语义近似（"开车" ↔ "驾驶"）
      Vectors capture semantics ("car" ↔ "automobile")
    - BM25 擅长精确关键词（产品型号、命令名、版本号）
      BM25 catches exact tokens (product codes, command names, versions)

设计 / Design:
    rank-bm25 内部存的是 tokenized 文档列表，没有原生增删 API。
    每次 add/remove 都重建底层 BM25Okapi 实例（O(N) 时间复杂度）。
    rank-bm25 has no native insert/delete; rebuild on every mutation.
    Acceptable up to ~100K chunks; switch to Tantivy/Lucene if scale grows.

按 sha1 分组 / Grouped by sha1:
    所有 chunk 在 docstore 中按 sha1 分桶；删除文件 = 删该桶所有 chunk。
    All chunks live under their file's sha1; delete = drop the bucket.

持久化 / Persistence:
    pickle 存 BM25Okapi（rank-bm25 内部是 numpy 数组），JSON 存 docstore。
    Persisted via pickle (BM25Okapi) + JSON (docstore).

⚠️ pickle 警告 / pickle warning:
    仅信任本地生成的 .pkl —— 不要加载来自外部的索引文件。
    Only trust locally generated .pkl files; never load untrusted indexes.
"""
from __future__ import annotations

# 标准库 / Stdlib.
import json
# pickle: 用于持久化 BM25Okapi（含 numpy 数组）。
# Used to persist BM25Okapi (which contains numpy arrays).
import pickle
from pathlib import Path
from typing import Optional

# jieba: 中文分词器；cut_for_search 模式针对搜索优化，切粒度更细。
# jieba: CJK tokenizer; cut_for_search yields finer tokens for search.
import jieba
# rank_bm25.BM25Okapi: 最经典的 BM25 实现 / Classic BM25 implementation.
from rank_bm25 import BM25Okapi

from src.config import settings
from src.rag.document import Document

# 索引文件名 / Index file basename.
_INDEX_NAME = "bm25"


def _tokenize(text: str) -> list[str]:
    """
    jieba 分词 + 小写化 / jieba tokenize + lowercase.

    cut_for_search 模式：粒度更细，召回更好（一个长词会同时切出短词）。
    cut_for_search produces finer tokens; better recall.
    """
    # 列表推导：分词 + 过滤空白 + 小写化。
    # List comprehension: tokenize + filter whitespace + lowercase.
    return [w.lower() for w in jieba.cut_for_search(text) if w.strip()]


class BM25Index:
    """
    简化版 BM25 索引 / Minimal BM25 index.

    内部数据结构 / Internal state:
        _docs:    list[Document]      按插入顺序保留 / Insertion-ordered
        _shas:    list[str]           与 _docs 并行的 sha1 数组 / Parallel sha1 array
        _bm25:    BM25Okapi | None    real index; None when empty
    """

    def __init__(
        self,
        docs: list[Document] | None = None,
        shas: list[str] | None = None,
    ) -> None:
        # list(...) 强制浅拷贝，避免外部修改污染索引。
        # list(...) shallow-copies to avoid external mutation.
        self._docs: list[Document] = list(docs or [])
        self._shas: list[str] = list(shas or [])
        self._bm25: BM25Okapi | None = None
        # 构造时即建索引（懒加载 vs eager 选 eager，后续查询零成本）。
        # Build the index eagerly so later queries have zero overhead.
        self._rebuild()

    def _rebuild(self) -> None:
        """
        重建底层 BM25 索引 / Rebuild the underlying BM25 index.

        BM25Okapi 构造时计算 idf + 平均长度，需要遍历所有文档。
        Constructor computes idf + avg length; scans all docs.
        """
        if not self._docs:
            # 空索引 = None；查询路径会跳过。
            # Empty index → None; search() returns [].
            self._bm25 = None
            return
        # 对每个文档先 tokenize 一次（avg 长度计算需要 token 数）。
        # Tokenize all docs (BM25 needs token sequences + avg length).
        tokens = [_tokenize(d.page_content) for d in self._docs]
        self._bm25 = BM25Okapi(tokens)

    # ──────────────────────────────────────────────────────
    # 构建与变更 / Build & mutate
    # ──────────────────────────────────────────────────────
    @classmethod
    def from_documents_with_sha1(
        cls, docs: list[Document], sha1: str
    ) -> "BM25Index":
        """从单个文件的 chunk 列表新建索引 / Build from one file's chunks."""
        if not docs:
            raise ValueError("Cannot build BM25Index from empty documents.")
        # 所有 chunk 共享同一 sha1（来自同一文件）。
        # All chunks share the file's sha1.
        return cls(docs=list(docs), shas=[sha1] * len(docs))

    def add_documents_with_sha1(self, docs: list[Document], sha1: str) -> None:
        """
        追加一个文件的所有 chunk / Append all chunks of one file.

        rank-bm25 不支持增量；这里只是 append 到内部数组并重建。
        rank-bm25 has no incremental API; we just append and rebuild.
        """
        if not docs:
            return
        self._docs.extend(docs)
        self._shas.extend([sha1] * len(docs))
        # 重建索引（O(N) where N = 当前所有 chunks）。
        # Rebuild (O(N) over all chunks).
        self._rebuild()

    def remove_by_sha1(self, sha1: str) -> int:
        """
        删除某 sha1 对应的全部 chunk，返回删除数。
        Drop all chunks tagged with `sha1`; returns count removed.
        """
        # 先用列表推导筛掉目标 sha1 的条目。
        # Use list comprehension to filter out target sha1 entries.
        keep_docs: list[Document] = []
        keep_shas: list[str] = []
        removed = 0
        # 同步遍历 _docs 和 _shas（zip）；保留非目标条目。
        # Iterate _docs + _shas in lockstep; keep non-target entries.
        for d, s in zip(self._docs, self._shas):
            if s == sha1:
                removed += 1
            else:
                keep_docs.append(d)
                keep_shas.append(s)
        # 没删任何东西 → 跳过重建省时间。
        # Nothing removed → skip rebuild to save time.
        if removed == 0:
            return 0
        self._docs = keep_docs
        self._shas = keep_shas
        self._rebuild()
        return removed

    # ──────────────────────────────────────────────────────
    # 检索 / Search
    # ──────────────────────────────────────────────────────
    def search(self, query: str, k: int = 20) -> list[tuple[int, float, Document]]:
        """
        BM25 检索；返回 (内部 id, 分数, 文档) 三元组列表，按分数降序。
        BM25 search → list of (id, score, doc) sorted by score desc.

        分数为 0 的条目（query 完全不匹配）会被过滤。
        Entries with score 0 (no token match) are filtered out.
        """
        # 空索引或空 query 直接返回 / Early exit on empty index or query.
        if self._bm25 is None or not self._docs:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        # get_scores 对所有文档算分；返回 ndarray，长度 = len(_docs)。
        # get_scores returns ndarray of scores, length = len(_docs).
        scores = self._bm25.get_scores(q_tokens)
        # 按分数排序拿前 k 个 / Sort by score and take top-k.
        ranked = sorted(
            ((i, float(scores[i])) for i in range(len(scores))),
            key=lambda x: x[1],
            reverse=True,
        )[:k]
        # 把 (id, score) 映射回 (id, score, doc)；过滤 0 分。
        # Map back to (id, score, doc); drop zero-score hits.
        return [(i, s, self._docs[i]) for i, s in ranked if s > 0]

    def __len__(self) -> int:
        """便于调试：len(index) 返回 chunk 数 / Convenience: chunk count."""
        return len(self._docs)

    # ──────────────────────────────────────────────────────
    # 持久化 / Persistence
    # ──────────────────────────────────────────────────────
    def save(self, dir_path: Path) -> None:
        """持久化到磁盘 / Persist to disk."""
        dir_path.mkdir(parents=True, exist_ok=True)
        if self._bm25 is not None:
            # pickle BM25Okapi 实例；包含 numpy idf 向量等。
            # pickle the BM25Okapi instance (contains numpy idf arrays).
            (dir_path / f"{_INDEX_NAME}.pkl").write_bytes(pickle.dumps(self._bm25))
        else:
            # 空索引：删除可能存在的旧 .pkl，避免污染。
            # Empty index: remove any stale .pkl.
            (dir_path / f"{_INDEX_NAME}.pkl").unlink(missing_ok=True)

        # docstore 用 JSON 存（可读、跨语言、无 pickle 安全风险）。
        # docstore as JSON: readable, cross-lang, no pickle risk.
        payload = {
            "docs": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self._docs
            ],
            "shas": self._shas,
        }
        (dir_path / f"{_INDEX_NAME}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, dir_path: Path) -> Optional["BM25Index"]:
        """
        从磁盘加载；JSON 不存在则返回 None（视为未建索引）。
        Load from disk; return None if JSON missing.
        """
        meta_path = dir_path / f"{_INDEX_NAME}.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # 把 JSON dict 还原成 Document 对象 / Reconstruct Document objects.
        docs = [
            Document(page_content=d["page_content"], metadata=d.get("metadata") or {})
            for d in meta.get("docs", [])
        ]
        shas = list(meta.get("shas") or [])
        # 构造器会自动 _rebuild() 重建 BM25Okapi。
        # Constructor auto-rebuilds the BM25Okapi.
        return cls(docs=docs, shas=shas)


def load_bm25_index() -> Optional[BM25Index]:
    """便捷加载 / Convenience loader."""
    return BM25Index.load(Path(settings.rag_persist_dir))