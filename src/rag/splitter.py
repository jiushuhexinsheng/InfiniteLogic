"""
文本切分器 / Text splitter.

简化版的递归切分（参考 LangChain RecursiveCharacterTextSplitter）。
A simplified recursive splitter inspired by LangChain's
RecursiveCharacterTextSplitter.

策略 / Strategy:
    依次尝试以 "\n\n" / "\n" / " " / "" 作为分隔符切分，直到块 ≤ chunk_size。
    Try separators in order until chunks fit within chunk_size.

为何递归 / Why recursive:
    优先保留段落语义（先按段落切），段落仍太长才往细切。
    Prefers paragraph boundaries; falls back finer only when necessary.

术语 / Terminology:
    - chunk_size: 单块字符上限 / Max chars per chunk
    - chunk_overlap: 相邻块的重叠字符数（防止信息被边界切断）
                     Overlap between adjacent chunks to avoid splitting key info
"""
from src.rag.document import Document


def _split_with_separator(text: str, sep: str, max_size: int) -> list[str]:
    """
    按指定分隔符切分，并把碎片重新组合到不超过 max_size。
    Split by separator and re-merge fragments without exceeding max_size.

    步骤 / Steps:
        1. text.split(sep) → 一堆碎片（可能很小）
        2. 贪心合并连续碎片，单块上限 max_size
        3. 单个碎片仍超长 → 递归用更激进的分隔符再切
    """
    if sep == "":
        # 空分隔符 = 字符级硬切；用切片实现。
        # Empty separator = char-level hard cut; via slicing.
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

    # 按分隔符切成碎片 / Split into raw pieces.
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        # 重新拼上分隔符，避免合并后丢失换行 / 空格。
        # Re-attach separator so newlines / spaces survive re-merge.
        piece = part + sep
        # 若 current + piece 会爆 max_size：先把当前 current 推入 chunks。
        # If joining would overflow: flush current first.
        if len(current) + len(piece) > max_size:
            if current:
                chunks.append(current)
            # 单个 piece 仍然超长 → 递归用更细分隔符再切。
            # If a single piece is still too big, recurse with finer separators.
            if len(piece) > max_size:
                chunks.extend(_split_recursive(piece, max_size))
                current = ""
            else:
                current = piece
        else:
            current += piece
    # 收尾：最后一块没满也要推 / Flush the final partial chunk.
    if current:
        chunks.append(current)
    return chunks


def _split_recursive(text: str, max_size: int) -> list[str]:
    """
    递归尝试不同分隔符 / Try each separator recursively.

    优先级 / Priority:
        段落（空行）> 行 > 词（空格）> 字符
        Paragraph > line > word > char
    """
    # 已经够小 → 直接返回（避免空 list）。
    # Already small enough → return as single chunk (skip empty).
    if len(text) <= max_size:
        return [text] if text else []
    for sep in ("\n\n", "\n", " ", ""):
        chunks = _split_with_separator(text, sep, max_size)
        # 全部块都 ≤ max_size 才算成功；否则换更细分隔符。
        # Accept only if every chunk fits; else escalate to finer separator.
        if all(len(c) <= max_size for c in chunks):
            return chunks
    # 理论上 "" 分隔符必能切到 ≤ max_size，所以不会走到这里。
    # In theory "" always works; this is dead code.
    return [text]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    给相邻块加重叠 / Add overlap between adjacent chunks.

    策略 / Strategy:
        把上一块的最后 overlap 个字符复制到下一块开头。
        Prepend last `overlap` chars of previous chunk to the next.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = out[-1]
        # 取上一块的"尾部 overlap 字符"作为重叠头。
        # Take last `overlap` chars of previous chunk as overlap head.
        head = prev[-overlap:] if len(prev) > overlap else prev
        out.append(head + chunks[i])
    return out


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """对外接口：切一段文本 / Public API: split one text string."""
    chunks = _split_recursive(text, chunk_size)
    return _apply_overlap(chunks, chunk_overlap)


def split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """
    把多个 Document 切成更小的 Document 列表，保留 metadata。
    Split each Document into smaller pieces while preserving metadata.
    """
    out: list[Document] = []
    for doc in docs:
        # 跳过纯空白文档（PDF 偶尔解析出空白页）。
        # Skip whitespace-only docs (e.g. blank PDF pages).
        if not doc.page_content.strip():
            continue
        for chunk in split_text(doc.page_content, chunk_size, chunk_overlap):
            if chunk.strip():
                # dict(doc.metadata) 浅拷贝避免共享引用。
                # Shallow-copy metadata so chunks don't share dicts.
                out.append(Document(page_content=chunk, metadata=dict(doc.metadata)))
    return out