import sys
from os import PathLike

import pytest

from caketool.utils.lib_utils import _install_hint, get_class, require_dependencies


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

    def test_missing_submodule_of_installed_package_raises(self):
        """Dot notation checks the full path, not only the top-level package."""

        @require_dependencies("os.nonexistent_submodule_xyz")
        def func_with_missing_submodule():
            return "should not reach"

        with pytest.raises(ImportError, match="os.nonexistent_submodule_xyz"):
            func_with_missing_submodule()

    def test_missing_parent_package_raises(self):
        """A missing parent package is reported, not raised as ModuleNotFoundError."""

        @require_dependencies("nonexistent_pkg_xyz.sub")
        def func_with_missing_parent():
            return "should not reach"

        with pytest.raises(ImportError, match="Missing required dependencies: nonexistent_pkg_xyz.sub"):
            func_with_missing_parent()

    def test_installed_submodule_executes(self):
        @require_dependencies("json.decoder")
        def func_with_submodule():
            return "ok"

        assert func_with_submodule() == "ok"


class TestInstallHint:
    def test_optional_extra_hint(self):
        assert _install_hint("google.cloud.aiplatform") == "caketool[gcp]"
        assert _install_hint("mlflow") == "caketool[onprem]"
        assert _install_hint("wandb.sdk") == "caketool[wandb]"

    def test_unknown_package_uses_top_level_name(self):
        assert _install_hint("some_pkg.sub") == "some_pkg"


class TestGetClass:
    def test_unimported_submodule(self, monkeypatch):
        """Submodules are imported on demand."""
        import email.mime

        monkeypatch.delitem(sys.modules, "email.mime.text", raising=False)
        monkeypatch.delattr(email.mime, "text", raising=False)

        cls = get_class("email.mime.text.MIMEText")

        assert cls is sys.modules["email.mime.text"].MIMEText

    def test_top_level_module(self):
        import json

        assert get_class("json") is json

    def test_missing_attribute_raises(self):
        with pytest.raises(AttributeError):
            get_class("os.NoSuchClassXyz")

    def test_missing_module_raises(self):
        with pytest.raises(ImportError):
            get_class("nonexistent_pkg_xyz.Thing")
