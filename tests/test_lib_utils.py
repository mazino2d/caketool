from os import PathLike

import pytest
from src.caketool.utils.lib_utils import get_class, require_dependencies


def test_get_PathLike_class():
    assert get_class("os.PathLike") == PathLike
    assert get_class("os.PathLike") is PathLike


class TestRequireDependencies:
    def test_installed_package_executes(self):
        """Function executes normally when dependency is installed."""

        @require_dependencies("os")
        def func_with_os():
            return "success"

        assert func_with_os() == "success"

    def test_missing_package_raises_import_error(self):
        """ImportError raised when dependency is missing."""

        @require_dependencies("nonexistent_package_xyz")
        def func_with_missing():
            return "should not reach"

        with pytest.raises(ImportError, match="Missing required dependencies"):
            func_with_missing()

    def test_multiple_installed_packages(self):
        """Function executes with multiple installed dependencies."""

        @require_dependencies("os", "sys", "json")
        def func_with_multiple():
            return "all installed"

        assert func_with_multiple() == "all installed"

    def test_one_missing_among_multiple(self):
        """ImportError when one of multiple dependencies is missing."""

        @require_dependencies("os", "nonexistent_pkg")
        def func_partial():
            return "should not reach"

        with pytest.raises(ImportError, match="nonexistent_pkg"):
            func_partial()

    def test_dot_notation_top_level_check(self):
        """Dot notation checks top-level package."""

        @require_dependencies("os.path")
        def func_with_submodule():
            return "submodule ok"

        assert func_with_submodule() == "submodule ok"

    def test_preserves_function_metadata(self):
        """Decorator preserves function name and docstring."""

        @require_dependencies("os")
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_passes_args_and_kwargs(self):
        """Decorator passes arguments correctly."""

        @require_dependencies("os")
        def func_with_args(a, b, c=None):
            return a + b + (c or 0)

        assert func_with_args(1, 2) == 3
        assert func_with_args(1, 2, c=3) == 6
