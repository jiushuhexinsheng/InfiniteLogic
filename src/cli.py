"""
交互式 CLI / Interactive CLI.

消费 agent.run_turn 的事件流，用 rich 渲染。
Consumes the event stream from agent.run_turn; renders via rich.

事件 / Events:
    reasoning_delta → 思考流 / reasoning stream
    content_delta   → 回答流 / answer stream
    tool_start/end  → 工具行 / tool lines
    error / done    → 终结 / terminators

三态机 / Three-state machine:
    in_reasoning / in_answer 跟踪当前正在哪一段，决定是否打印新前缀或换行。
    Track which section we're rendering to decide on prefixes / newlines.
"""
from __future__ import annotations

# 标准库 / Stdlib.
import uuid

# rich 是终端美化库 / rich is a terminal styling library.
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from src.agent import run_turn
from src.config import settings
from src.multi_agent import run_multi_agent
from src.session import SessionStore
from src.tools.rag_tool import reload_hybrid, reload_reranker
from src.usage import USAGE

# 全局 Console / Global rich Console singleton.
console = Console()


# 帮助文本，多处复用 / Reusable help text.
_HELP_TEXT = (
    "[dim]Commands:\n"
    "  /new                          new session (rotate thread_id)\n"
    "  /clear                        drop current thread history\n"
    "  /reranker on|off              toggle reranker\n"
    "  /reranker model <hf_name>     hot-swap reranker model\n"
    "  /hybrid on|off                toggle BM25+vector hybrid retrieval\n"
    "  /usage                        show cumulative token & cost stats\n"
    "  /usage reset                  reset usage counters\n"
    "  /team <task>                  run Planner/Researcher/Writer pipeline\n"
    "  /help                         show this help\n"
    "  /quit | /exit                 quit[/dim]"
)


def _welcome() -> None:
    """打印启动 banner / Print startup banner."""
    hint = ""
    if settings.llm_thinking_enabled:
        # 思考模式开启时附加提示，让用户知道会看到 Thinking 段落。
        # When thinking is on, hint at the extra rendered section.
        hint = f"\n[dim]Thinking: {settings.llm_reasoning_effort}[/dim]"
    console.print(
        Panel(
            "[bold cyan]InfiniteLogic Agent[/bold cyan]\n" + _HELP_TEXT + hint,
            border_style="cyan",
        )
    )


def _handle_command(cmd: str) -> bool:
    """
    处理 / 开头的命令；返回 True 表示已处理。
    Handle `/`-prefixed commands; True if recognized.

    分发设计 / Dispatch:
        简单 if-elif 链；几条命令不值得引入命令模式 / argparse。
        Simple if-elif chain; not worth a command pattern for a few cmds.
    """
    parts = cmd.strip().split()
    if not parts:
        return False
    head = parts[0].lower()

    if head == "/help":
        console.print(_HELP_TEXT)
        return True

    if head == "/reranker":
        # `/reranker on` 或 `/reranker off`
        if len(parts) == 2 and parts[1] in ("on", "off"):
            msg = reload_reranker(enabled=(parts[1] == "on"))
            console.print(f"[dim]{msg}[/dim]")
            return True
        # `/reranker model BAAI/bge-reranker-v2-m3`
        if len(parts) >= 3 and parts[1].lower() == "model":
            # 用 " ".join 而非 parts[2]：兼容路径里含空格的奇葩名（理论上不应该）。
            # join in case the model name contains spaces (unusual).
            model = " ".join(parts[2:])
            msg = reload_reranker(model_name=model)
            console.print(f"[dim]{msg}[/dim]")
            return True
        console.print("[yellow]Usage: /reranker on|off | /reranker model <hf_name>[/yellow]")
        return True

    if head == "/hybrid":
        if len(parts) == 2 and parts[1] in ("on", "off"):
            msg = reload_hybrid(parts[1] == "on")
            console.print(f"[dim]{msg}[/dim]")
            return True
        console.print("[yellow]Usage: /hybrid on|off[/yellow]")
        return True

    if head == "/usage":
        if len(parts) == 2 and parts[1] == "reset":
            USAGE.reset()
            console.print("[dim]Usage counters reset.[/dim]")
            return True
        console.print(f"[dim]Usage: {USAGE.format()}[/dim]")
        return True

    # 未识别命令 / Unknown command.
    return False


async def _render_multi_agent(task: str) -> None:
    """
    渲染多 Agent 流水线 / Render the multi-agent pipeline.

    三阶段事件 / Three-stage events:
        plan_done       → 显示子任务列表
        research_done   → 显示研究结果
        content_delta   → 流式渲染最终答案
    """
    in_answer = False
    async for event in run_multi_agent(task):
        etype = event["type"]
        if etype == "plan_done":
            console.print(
                f"\n[bold cyan]Plan:[/bold cyan]\n{event['plan']}",
                highlight=False,
            )
        elif etype == "research_done":
            console.print(
                f"\n[bold cyan]Findings:[/bold cyan]\n{event['findings']}",
                highlight=False,
            )
        elif etype == "content_delta":
            if not in_answer:
                console.print("\n[bold green]Final:[/bold green] ", end="")
                in_answer = True
            console.print(event["text"], end="", highlight=False)
        elif etype == "reasoning_delta":
            # 仅在用户开启思考显示时打印 / Only show if user opted in.
            if settings.show_reasoning and settings.llm_thinking_enabled:
                console.print(event["text"], end="", highlight=False)
        elif etype == "error":
            console.print(f"\n[bold red]{event['message']}[/bold red]")
        elif etype == "done":
            pass
    # 结尾换行避免下一段输入紧贴 / Trailing newline.
    console.print()


async def run_cli() -> None:
    """主交互循环 / Main interactive loop."""
    _welcome()

    # SessionStore 在 CLI 启动时打开，退出时关闭。
    # Open once at CLI start; close on exit.
    session = await SessionStore.open()
    try:
        # 启动时生成首个 thread_id / Initial thread_id at startup.
        thread_id = str(uuid.uuid4())
        # 仅显示前 8 位便于识别 / Show first 8 chars for brevity.
        console.print(f"[dim]Session: {thread_id[:8]}...[/dim]\n")

        while True:
            try:
                # Prompt.ask 阻塞读取一行输入 / Blocking line read.
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
            except (EOFError, KeyboardInterrupt):
                # Ctrl+D / Ctrl+C 优雅退出 / Graceful exit.
                console.print("\n[dim]Goodbye![/dim]")
                break

            # 空输入跳过 / Skip blank input.
            if not user_input.strip():
                continue

            cmd = user_input.strip().lower()
            # 退出命令（支持 / 前缀和裸命令两种）/ Exit commands.
            if cmd in ("/quit", "/exit", "quit", "exit"):
                console.print("[dim]Goodbye![/dim]")
                break
            # 新会话：换 thread_id（旧会话仍在 SessionStore 中保留）。
            # New session: rotate thread_id; the old one stays in store.
            if cmd == "/new":
                thread_id = str(uuid.uuid4())
                console.print(f"[dim]New session: {thread_id[:8]}...[/dim]\n")
                continue
            # 清当前会话：物理删除 SQLite 行 / Hard-delete current thread.
            if cmd == "/clear":
                await session.clear(thread_id)
                console.print("[dim]Current thread cleared.[/dim]\n")
                continue
            # /team <task> 走多 Agent 流水线 / Multi-agent pipeline.
            if cmd.startswith("/team "):
                # 剥掉 "/team " 前缀拿到任务文本 / Strip prefix.
                task = user_input.strip()[len("/team "):].strip()
                if task:
                    try:
                        await _render_multi_agent(task)
                        console.print(Rule(style="dim"))
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
                continue
            # 其他 / 开头命令 / Other slash commands.
            if cmd.startswith("/"):
                if _handle_command(user_input):
                    continue
                # 未识别的命令不传给 LLM，避免误调用 / Unknown command: don't forward.
                console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
                continue

            # 普通对话 / Regular conversation.
            try:
                await _render_turn(user_input, thread_id, session)
                console.print(Rule(style="dim"))
            except Exception as exc:  # noqa: BLE001
                # 异常不杀 CLI；打印后继续 / Don't kill CLI on error.
                console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
    finally:
        # finally 保证 SessionStore 总能正常关闭。
        # finally guarantees SessionStore closes even on exceptions.
        await session.close()


async def _render_turn(user_input: str, thread_id: str, session: SessionStore) -> None:
    """
    渲染单轮事件流 / Render one turn's events.

    三态机控制何时打印 prefix / 何时换行。
    Three-state machine controls when to emit prefixes / newlines.
    """
    # 状态变量：当前正在渲染哪一段。
    # State: which section we're currently rendering.
    in_reasoning = False
    in_answer = False

    async for event in run_turn(user_input, thread_id, session):
        etype = event["type"]

        # ----- 思考流 / Reasoning stream -----
        if etype == "reasoning_delta":
            # 配置关掉 reasoning 显示 → 整体跳过 / Skip if disabled.
            if not settings.show_reasoning or not settings.llm_thinking_enabled:
                continue
            if not in_reasoning:
                # 首个 reasoning 片段时打 "Thinking:" 前缀。
                # First reasoning chunk: emit "Thinking:" prefix.
                console.print("\n[dim magenta]Thinking:[/dim magenta] ", end="")
                in_reasoning = True
            # end="" 关闭自动换行；highlight=False 关闭自动着色（防止数字被染色）。
            # end="" disables newline; highlight=False prevents number coloring.
            console.print(event["text"], end="", highlight=False)

        # ----- 回答流 / Answer stream -----
        elif etype == "content_delta":
            if not in_answer:
                # 从 thinking 切到 answer 之间补换行。
                # Newline when transitioning from thinking to answer.
                if in_reasoning:
                    console.print()
                console.print("\n[bold green]Assistant:[/bold green] ", end="")
                in_answer = True
                in_reasoning = False
            console.print(event["text"], end="", highlight=False)

        # ----- 工具开始 / Tool started -----
        elif etype == "tool_start":
            # 切换上下文：前一段（answer 或 thinking）结束，补换行隔开。
            # Section switch: flush previous section with newline.
            if in_answer or in_reasoning:
                console.print()
                in_answer = False
                in_reasoning = False
            args = event.get("args", {})
            # 用 dim 显示 args 避免淹没主信息 / Dim args to keep focus on tool name.
            console.print(
                f"\n[yellow]  Tool:[/yellow] [bold]{event['name']}[/bold]"
                f"  [dim]{args}[/dim]"
            )

        # ----- 工具结束 / Tool finished -----
        elif etype == "tool_end":
            out = event.get("output", "")
            # 长输出截断到 150 字预览，避免淹屏。
            # Truncate long outputs to 150 chars preview.
            preview = out[:150] + "..." if len(out) > 150 else out
            console.print(f"  [dim green]→[/dim green] {preview}")

        # ----- 错误事件 / Error event -----
        elif etype == "error":
            console.print(f"\n[bold red]{event['message']}[/bold red]")

        # ----- 终止 / Terminal -----
        elif etype == "done":
            # 显式 done 事件不渲染（_render_turn 自然结束）。
            # Explicit done is a no-op; loop ends naturally afterward.
            pass

    # 收尾换行避免下一段输入紧贴。
    # Trailing newline for next prompt readability.
    console.print()