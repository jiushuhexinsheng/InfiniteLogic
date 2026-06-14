"""
网页搜索工具（DuckDuckGo）/ Web search via DuckDuckGo.

不需要 API key，免费但有速率限制。
No API key needed; free but rate-limited.

升级建议 / Upgrade options:
    - Tavily（需 API key，更稳）/ Tavily (requires API key, more reliable)
    - SerpAPI / Bing / Google CSE
    - Brave Search API（隐私友好 / privacy-friendly）
"""
import json

from src.tools.base import tool


@tool("Search the web via DuckDuckGo. Returns merged text. Input: query string.")
def web_search(query: str) -> str:
    # 延迟 import 防止 ddgs 启动慢（重导入耗时）。
    # Defer import to keep startup fast.
    from duckduckgo_search import DDGS

    # with 语句保证底层 HTTP session 释放。
    # `with` ensures the underlying HTTP session is closed.
    with DDGS() as ddgs:
        # ddgs.text() 返回 dict 迭代器；max_results 限制条数。
        # text() yields dicts; max_results caps the count.
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No results found."
    # 把每条结果格式化成 "[i] title\n body" 形式。
    # Format each result as "[i] title\n body".
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        parts.append(f"[{i}] {title}\n{body}")
    # 双换行分隔每条结果，便于 LLM 区分。
    # Blank line between results helps the LLM parse.
    return "\n\n".join(parts)


@tool("Search the web via DuckDuckGo. Returns JSON list with title/snippet/url. Use when you need source links.")
def web_search_results(query: str) -> str:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=4))
    if not results:
        return "[]"
    # JSON 输出比纯文本更适合需要引用 URL 的场景。
    # JSON output suits LLMs that need to cite URLs.
    payload = [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
        }
        for r in results
    ]
    # ensure_ascii=False 保留中文原文；indent=2 给 LLM 看着舒服。
    # ensure_ascii=False keeps CJK; indent=2 helps the LLM read.
    return json.dumps(payload, ensure_ascii=False, indent=2)