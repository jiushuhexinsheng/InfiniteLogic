"""
文档加载器 / Document loaders.

按扩展名分派到对应解析器，统一返回 Document 列表。
Dispatches by file extension to the right parser; returns Document list.

支持格式 / Supported formats:
    .txt .md   — 纯文本 / Plain text
    .pdf       — pypdf 按页加载 / pypdf, one Document per page
    .docx .doc — docx2txt 整篇加载 / docx2txt, whole document
    .csv       — 整文件文本 / Whole-file text
    .html .htm — BeautifulSoup 提取正文 / BeautifulSoup extraction

设计选择 / Design choices:
    - 每种格式延迟 import 对应库（避免启动时全量加载重依赖）
      Lazy import per format to keep startup fast
    - 解析失败不抛错而是返回 [] + 打 warning
      Failure → empty list + warning; never break the pipeline
"""
from pathlib import Path

from src.rag.document import Document


def load_file(path: Path) -> list[Document]:
    """
    根据扩展名加载文件，失败返回空列表。
    Load a file by extension; returns empty list on failure.
    """
    # 小写化以兼容 .PDF / .Pdf / .pdf。
    # Lowercase to accept .PDF / .Pdf / .pdf.
    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md", ".csv"):
            # 纯文本类：一次性读到字符串。
            # Plain-text-like: read into one string.
            return [
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name},
                )
            ]
        if suffix == ".pdf":
            # 延迟 import pypdf：仅在真有 PDF 时加载（PDF 解析库较大）。
            # Lazy-import pypdf; only load when needed.
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            # 每页一个 Document，附带 page 元数据。
            # One Document per page, with `page` metadata.
            return [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": path.name, "page": i + 1},
                )
                for i, page in enumerate(reader.pages)
            ]
        if suffix in (".docx", ".doc"):
            import docx2txt

            # docx2txt.process 返回整文档纯文本；"" if 提取失败。
            # docx2txt.process returns whole-doc text; "" on failure.
            text = docx2txt.process(str(path)) or ""
            return [Document(page_content=text, metadata={"source": path.name})]
        if suffix in (".html", ".htm"):
            from bs4 import BeautifulSoup

            html = path.read_text(encoding="utf-8")
            # get_text(separator="\n") 用换行连接 inline 文字，提升可读性。
            # separator="\n" makes the output more parseable.
            text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
            return [Document(page_content=text, metadata={"source": path.name})]
    except Exception as exc:  # noqa: BLE001
        # 单文件失败不阻塞整体流程；打 warn 让用户知道。
        # Single-file failures don't abort; warn so the user notices.
        print(f"  [warn] Failed to load {path.name}: {exc}")
    return []


# 支持的扩展名集合 / Supported extensions.
# 用 set 加速 in 检查（O(1)）/ Set for O(1) lookup.
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".doc", ".csv", ".html", ".htm"}


def collect_files(target: Path) -> list[Path]:
    """
    递归收集目标路径下所有受支持的文件。
    Recursively gather all supported files under target.
    """
    if target.is_file():
        # 单文件：扩展名匹配才收 / Single file: include only if supported ext.
        return [target] if target.suffix.lower() in SUPPORTED_EXTS else []
    # 目录：rglob("*") 递归遍历所有文件 / Dir: rglob recurses.
    return [
        p
        for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]