#!/usr/bin/env python3
"""
文档入库 CLI / Document ingestion CLI.

用法 / Usage:
    python ingest.py                    # 入库 RAG_DOCS_DIR 全部支持文件
    python ingest.py path/to/file.pdf   # 单文件入库
    python ingest.py --clear            # 删除整个 Qdrant 索引后重建

为什么单独脚本而不是 CLI 子命令 / Why a separate script:
    入库可能跑几分钟（首次下载 embedding 模型 + 大量文档）；
    单独脚本便于后台 nohup / cron 调度。
    Ingest may take minutes (first-time model download + many docs);
    a standalone script suits cron / nohup scheduling.
"""
# 标准库 / Stdlib only.
import argparse
import sys
from pathlib import Path

# rich Console 做彩色输出 / rich Console for colored output.
from rich.console import Console

from src.config import settings
from src.rag.ingest_pipeline import ingest
from src.rag.loader import SUPPORTED_EXTS, collect_files

console = Console()


def main() -> None:
    """CLI 参数解析 + 入库 / Parse CLI args + run ingestion."""
    # argparse 自动生成 --help 文档 / argparse auto-generates --help.
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    # nargs="?" 让位置参数可选 / nargs="?" makes the positional optional.
    parser.add_argument("path", nargs="?", help="File or directory (default: RAG_DOCS_DIR)")
    # action="store_true" 让 --clear 是 boolean flag / boolean flag style.
    parser.add_argument("--clear", action="store_true", help="Delete the existing index before ingesting")
    args = parser.parse_args()

    # 缺路径参数 → 用配置中的默认目录 / Default to configured dir if no path.
    target = Path(args.path) if args.path else Path(settings.rag_docs_dir)
    if not target.exists():
        console.print(f"[red]Path not found: {target}[/red]")
        sys.exit(1)

    # 扫描所有受支持的文件 / Gather all supported files.
    files = collect_files(target)
    if not files:
        console.print(f"[yellow]No supported files in {target}[/yellow]")
        console.print(f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}")
        # exit 0：不算错误；只是没东西可处理。
        # exit 0: nothing to do is fine, not an error.
        sys.exit(0)

    console.print(f"Found [bold]{len(files)}[/bold] files in [cyan]{target}[/cyan]")
    console.print(
        f"Embedding model: [cyan]{settings.rag_embedding_model}[/cyan]\n"
        f"(first run will download the ONNX model to ~/.cache/fastembed)"
    )

    # 调用入库流水线；返回统计字典。
    # Run the ingestion pipeline; returns stats dict.
    stats = ingest(files, clear=args.clear)

    # 输出统计 / Print stats.
    console.print(
        f"\n[bold]Done.[/bold] "
        f"new_files={stats['new_files']}  "
        f"updated={stats['updated']}  "
        f"new_chunks={stats['new_chunks']}  "
        f"skipped={stats['skipped']}"
    )


if __name__ == "__main__":
    main()