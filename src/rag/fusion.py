"""
检索结果融合 / Retrieval fusion.

RRF (Reciprocal Rank Fusion):
    score(d) = Σ 1 / (k + rank_i(d))
    多路检索结果按倒数排名加和；k 常用 60。
    Multi-source results combined via reciprocal-rank sums; k commonly 60.

为何 RRF / Why RRF:
    无需归一化分数（不同检索器分数尺度差异大）。
    No score normalization needed (vector cosine vs BM25 scale differ).
    经典论文 Cormack 2009 验证简单有效。
    Empirically robust per Cormack 2009.

直观理解 / Intuition:
    每个 doc 在每路里都有个排名 rank（1=最佳）；倒数 1/(k+rank) 越大越好。
    多路里都靠前的 doc 得分最高，是真正"两边都觉得相关"的结果。
    A doc gets a high RRF score iff it ranks well in multiple retrievers,
    i.e. multiple retrievers agree it's relevant.
"""
from src.rag.document import Document


def _doc_key(doc: Document) -> str:
    """
    用 page_content 作为 doc 唯一键 / Identify a doc by page_content.

    更稳健的做法：sha1 + chunk_idx 作 key，但需要 metadata 里有这些字段。
    保持简单：直接拿 page_content 字符串去重，对短文本足够。
    Could use sha1+chunk_idx for robustness; keeping it simple here.
    """
    return doc.page_content


def rrf_fuse(
    rankings: list[list[Document]],
    k: int = 60,
    top_n: int | None = None,
) -> list[Document]:
    """
    将多路有序 Document 列表融合，返回新的有序列表。
    Fuse multiple ranked Document lists into one ranked list.

    Args:
        rankings: 每路一个 Document 列表（已按相关性降序）
                  One Document list per source (sorted by relevance desc)
        k: RRF 常数；越大原排名差异被压缩越多
           RRF constant; larger k flattens rank differences
        top_n: 返回前 N 条（None 表示全部）/ Top N (None = all)
    """
    # 用两个 dict：scores 累加分数，docs 保留原始 Document 对象。
    # Two dicts: scores accumulates, docs preserves original Document objs.
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}

    # 遍历每路检索结果 / Iterate each retriever's ranking.
    for ranked in rankings:
        # enumerate(start=1)：rank 从 1 开始，避免 1/(k+0) 极值。
        # Start rank from 1; avoids the 1/(k+0) edge case.
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_key(doc)
            # 公式：score(d) += 1 / (k + rank)
            # Formula: score(d) += 1 / (k + rank).
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            # 第一次见到时记下 Document 对象；后续同 key 用同样 metadata。
            # First-seen wins; later occurrences share the same Document.
            docs.setdefault(key, doc)

    # 按累计分降序排列 keys / Sort keys by accumulated score desc.
    sorted_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)
    fused = [docs[k_] for k_ in sorted_keys]
    # top_n 截断（None 表示不截）/ Slice to top_n if requested.
    if top_n is not None:
        fused = fused[:top_n]
    return fused