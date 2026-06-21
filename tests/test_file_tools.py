"""文件工具安全测试 / File tools security tests."""
import os

import pytest

from src.config import settings
from src.tools.file_tools import _resolve, list_directory, read_file, write_file


class TestPathResolution:
    """路径解析与沙箱测试 / Path resolution + sandbox tests."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        """用临时目录替代 workspace / Use temp dir as workspace."""
        self._orig = settings.workspace_dir
        settings.workspace_dir = str(tmp_path)
        yield
        settings.workspace_dir = self._orig

    def test_inside_workspace_ok(self, tmp_path):
        """正常路径解析 / Normal path resolved."""
        (tmp_path / "test.txt").write_text("hello", encoding="utf-8")
        result = _resolve("test.txt")
        assert result.exists()
        assert result.read_text() == "hello"

    def test_parent_traversal_blocked(self, tmp_path):
        """../ 越界被阻止 / ../ traversal blocked."""
        with pytest.raises(PermissionError, match="outside workspace"):
            _resolve("../../etc/passwd")

    def test_absolute_path_blocked(self, tmp_path):
        """绝对路径越界被阻止 / Absolute path blocked."""
        with pytest.raises(PermissionError, match="outside workspace"):
            _resolve("/etc/passwd")

    def test_windows_path_blocked(self, tmp_path):
        """Windows 路径越界被阻止 / Windows-style path blocked."""
        with pytest.raises(PermissionError):
            _resolve("C:\\Windows\\System32\\config")

    def test_symlink_inside_workspace(self, tmp_path):
        """工作区内软链接正常工作 / Symlink inside workspace works."""
        src = tmp_path / "src.txt"
        src.write_text("content", encoding="utf-8")
        link = tmp_path / "link.txt"
        os.symlink(src, link)
        result = _resolve("link.txt")
        assert result.read_text() == "content"


class TestFileOperations:
    """文件读写操作测试 / File read/write tests."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        self._orig = settings.workspace_dir
        settings.workspace_dir = str(tmp_path)
        yield
        settings.workspace_dir = self._orig

    def test_write_and_read(self, tmp_path):
        """写入并读取 / Write then read."""
        write_file("test.txt", "hello world")
        result = read_file("test.txt")
        assert result == "hello world"

    def test_write_creates_parent_dirs(self, tmp_path):
        """自动创建父目录 / Auto-create parent dirs."""
        result = write_file("sub/deep/file.txt", "nested")
        assert "Wrote" in result
        assert (tmp_path / "sub" / "deep" / "file.txt").exists()

    def test_read_missing_file(self, tmp_path):
        """读取不存在的文件 / Read missing file."""
        result = read_file("nonexistent.txt")
        assert "File not found" in result

    def test_list_directory(self, tmp_path):
        """列出目录内容 / List directory."""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "sub").mkdir()
        result = list_directory(".")
        assert "a.txt" in result
        assert "b.txt" in result
        assert "D sub" in result or "sub" in result

    def test_list_empty_directory(self, tmp_path):
        """空目录 / Empty directory."""
        result = list_directory(".")
        assert "empty" in result.lower()

    def test_list_not_a_directory(self, tmp_path):
        """非目录路径 / Not a directory."""
        (tmp_path / "file.txt").write_text("x")
        result = list_directory("file.txt")
        assert "Not a directory" in result
