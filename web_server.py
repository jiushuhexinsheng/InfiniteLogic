#!/usr/bin/env python3
"""
Web UI 启动脚本 / Web UI launcher.

用法 / Usage:
    python web_server.py            # 默认 0.0.0.0:8000
    HOST=127.0.0.1 PORT=8080 python web_server.py

为什么有这个文件 / Why this file:
    `python -m src.web` 也能跑，但需要知道模块路径；这里提供更直观的入口。
    `python -m src.web` works too, but this entrypoint is more discoverable.
    同时支持 HOST / PORT 环境变量覆盖默认值（Docker / k8s 友好）。
    HOST / PORT env override is Docker / k8s friendly.
"""
# 标准库 / Stdlib.
import os

# uvicorn 是 ASGI 服务器；FastAPI 在 ASGI 之上。
# uvicorn is an ASGI server; FastAPI runs on top of ASGI.
import uvicorn


def main() -> None:
    """启动 uvicorn / Boot uvicorn."""
    # 从环境变量读 HOST / PORT，缺省回退到 0.0.0.0:8000。
    # Read HOST / PORT from env; fall back to 0.0.0.0:8000.
    host = os.environ.get("HOST", "0.0.0.0")
    # 注意 PORT 是字符串，转 int / Note: PORT is str, convert to int.
    port = int(os.environ.get("PORT", "8000"))
    # 用 "src.web:app" 字符串而非 import 对象：让 uvicorn 自己做模块查找。
    # Use string form so uvicorn handles the import (enables reload mode).
    uvicorn.run("src.web:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()