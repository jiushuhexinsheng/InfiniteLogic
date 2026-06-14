"""
安全计算器 / Safe calculator.

用 AST 白名单代替 eval()，禁止函数调用、变量名、属性访问。
Uses an AST whitelist instead of eval(); function calls, names, and
attribute access are forbidden.

为何不用 eval() / Why not eval():
    eval(user_input) 可被注入任意代码：
    `eval(user_input)` allows arbitrary code injection:
        eval("__import__('os').system('rm -rf /')")
    LLM 可能不小心生成恶意串；用户也可能故意；都得防。
    LLMs may accidentally produce malicious strings; users may try on purpose.

实现思路 / Approach:
    用 `ast.parse(mode="eval")` 把字符串解析成抽象语法树 (AST)，
    然后只遍历**白名单节点**：常量、二元运算、一元运算。
    遇到函数调用、变量名、属性访问等一律拒绝。
    Parse via `ast.parse(mode="eval")` then walk only a whitelist of
    nodes (constants, binary ops, unary ops). Anything else → reject.
"""
# 标准库 / Stdlib only.
import ast
import operator
from typing import Any, Callable

from src.tools.base import tool


# ─────────────────────────────────────────────────────────────────────
# AST 节点 → 实际操作函数 的映射。
# Map AST op nodes → runtime callables (operator.* module functions).
#
# operator.add(a, b) 等价于 a + b；这种映射避免在 _safe_eval 里写
# if/elif 长链。
# operator.add(a, b) ≡ a + b; this table avoids long if/elif chains.
#
# 显式类型：键是 AST 节点类型，值是接受任意数量 float 参数的 callable。
# Explicit typing: keys are AST op types, values are callables.
# ─────────────────────────────────────────────────────────────────────
_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,            # +
    ast.Sub: operator.sub,            # -
    ast.Mult: operator.mul,           # *
    ast.Div: operator.truediv,        # /  (always returns float)
    ast.Pow: operator.pow,            # **
    ast.Mod: operator.mod,            # %
    ast.FloorDiv: operator.floordiv,  # //
    ast.USub: operator.neg,           # -x（一元负号 / unary minus）
    ast.UAdd: operator.pos,           # +x（一元正号 / unary plus）
}


def _safe_eval(node: ast.AST) -> float:
    """
    递归求值 AST，只接受白名单节点。
    Recursively evaluate the AST, whitelist only.

    抛 ValueError 表示遇到不允许的节点（函数调用、变量名等）。
    Raises ValueError when an unsupported node is encountered.
    """
    # 数字常量节点：直接返回值。
    # Numeric constant: return value directly.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    # 二元运算：递归求左右子树后调用对应 op 函数。
    # Binary op: recurse on left/right then apply mapped op.
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))

    # 一元运算（如 -5）：递归求 operand。
    # Unary op (e.g. -5): recurse on operand.
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))

    # 其他一切（函数调用 Call / 名字 Name / 属性 Attribute / ...）一律拒绝。
    # Reject anything else (Call, Name, Attribute, ...).
    # ast.dump 给出可读的节点结构便于调试。
    # ast.dump prints a readable representation for debugging.
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool("Evaluate a math expression. Supports + - * / ** % //. No variables or functions. Example: '(3+5)*2/4'")
def calculator(expression: str) -> str:
    """
    Evaluate a math expression. Supports +, -, *, /, **, %, //.
    """
    # mode="eval" 限定只能解析单个表达式（不允许 "x=1" 之类的语句）。
    # mode="eval" restricts parsing to a single expression — no statements.
    tree = ast.parse(expression.strip(), mode="eval")
    # tree.body 是表达式根节点；递归求值。
    # tree.body is the expression's root node; recurse to evaluate.
    result = _safe_eval(tree.body)
    # `:g` 自动选最短表示（不带尾零；"4" 而非 "4.0"）。
    # `:g` format: shortest representation (e.g. "4" not "4.0").
    return f"{result:g}"