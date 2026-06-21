"""安全计算器测试 / Safe calculator tests."""
import ast

import pytest

from src.tools.calculator import _safe_eval, calculator


class TestSafeEval:
    """AST 白名单求值测试 / AST whitelist eval tests."""

    def test_simple_addition(self):
        tree = ast.parse("1 + 2", mode="eval")
        assert _safe_eval(tree.body) == 3.0

    def test_subtraction(self):
        tree = ast.parse("10 - 3", mode="eval")
        assert _safe_eval(tree.body) == 7.0

    def test_multiplication(self):
        tree = ast.parse("4 * 5", mode="eval")
        assert _safe_eval(tree.body) == 20.0

    def test_division_float(self):
        tree = ast.parse("7 / 2", mode="eval")
        assert _safe_eval(tree.body) == 3.5

    def test_exponent(self):
        tree = ast.parse("2 ** 10", mode="eval")
        assert _safe_eval(tree.body) == 1024.0

    def test_modulo(self):
        tree = ast.parse("10 % 3", mode="eval")
        assert _safe_eval(tree.body) == 1.0

    def test_floor_div(self):
        tree = ast.parse("7 // 2", mode="eval")
        assert _safe_eval(tree.body) == 3.0

    def test_negative_number(self):
        tree = ast.parse("-5", mode="eval")
        assert _safe_eval(tree.body) == -5.0

    def test_complex_expression(self):
        tree = ast.parse("(3 + 5) * 2 / 4", mode="eval")
        assert _safe_eval(tree.body) == 4.0

    def test_chained_ops(self):
        tree = ast.parse("2 ** 3 + 4 * 5 - 10 / 2", mode="eval")
        assert _safe_eval(tree.body) == 23.0  # 8 + 20 - 5

    def test_float_input(self):
        tree = ast.parse("3.14 * 2", mode="eval")
        assert _safe_eval(tree.body) == pytest.approx(6.28)


class TestCalculatorSecurity:
    """安全测试：确保 AST 白名单阻止注入 / Security: block injection."""

    def test_rejects_function_call(self):
        with pytest.raises(ValueError):
            tree = ast.parse("__import__('os')", mode="eval")
            _safe_eval(tree.body)

    def test_rejects_name_variable(self):
        with pytest.raises(ValueError):
            tree = ast.parse("x + 1", mode="eval")
            _safe_eval(tree.body)

    def test_rejects_attribute_access(self):
        with pytest.raises(ValueError):
            tree = ast.parse("obj.attr", mode="eval")
            _safe_eval(tree.body)

    def test_rejects_eval_call(self):
        with pytest.raises(ValueError):
            tree = ast.parse("eval('1+1')", mode="eval")
            _safe_eval(tree.body)

    def test_rejects_lambda(self):
        with pytest.raises(ValueError):
            tree = ast.parse("(lambda x: x)(1)", mode="eval")
            _safe_eval(tree.body)


class TestCalculatorTool:
    """calculator 工具函数测试 / Tool function tests."""

    def test_valid_expression(self):
        result = calculator("1 + 2")
        assert "3" in result

    def test_float_formatting(self):
        result = calculator("4 / 2")
        assert result == "2"  # `:g` 去尾零

    def test_large_number(self):
        result = calculator("2 ** 100")
        # `:g` 对大数会输出科学计数法 / `:g` format uses scientific for large nums.
        assert "1.26765" in result
