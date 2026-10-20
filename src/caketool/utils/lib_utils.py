import functools
import importlib
import importlib.util
import sys

# Install hint for optional dependencies, keyed by module path prefix.
_INSTALL_HINTS = {
    "google.cloud.aiplatform": "caketool[gcp]",
    "google.cloud.bigquery": "caketool[gcp]",
    "google.cloud.storage": "caketool[gcp]",
    "bigframes": "caketool[gcp]",
    "mlflow": "caketool[onprem]",
    "wandb": "caketool[wandb]",
    "pyspark": "caketool[spark]",
    "polars": "caketool[polars]",
}


def get_class(name: str):
    """Import and return a class (or any attribute) by its fully qualified dotted name.

    Equivalent to a dynamic ``from a.b import C`` where the full path is
    given as a single string.  This is useful for deferred or configurable
    imports where the exact class is not known until runtime.  Submodules
    are imported as needed, so ``"xml.etree.ElementTree.Element"`` works
    even if ``xml.etree.ElementTree`` has not been imported yet.

    Parameters
    ----------
    name : str
        Fully qualified dotted path, e.g. ``"sklearn.preprocessing.StandardScaler"``
        or ``"sklearn.linear_model.LogisticRegression"``.

    Returns
    -------
    type
        The resolved class or attribute object.

    Raises
    ------
    ImportError
        If the module part of ``name`` cannot be imported.
    AttributeError
        If the module exists but does not define the requested attribute.

    Examples
    --------
    >>> StandardScaler = get_class("sklearn.preprocessing.StandardScaler")
    >>> scaler = StandardScaler(with_mean=False)
    """
    module_path, _, attr = name.rpartition(".")
    if not module_path:
        return importlib.import_module(name)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        # Re-raise when a dependency of the module is missing, not the path itself.
        if e.name is None or not module_path.startswith(e.name):
            raise
        # Otherwise the parent is an attribute, not a module (e.g. "pkg.Class.attr").
        return getattr(get_class(module_path), attr)
    return getattr(module, attr)


def _is_installed(module: str) -> bool:
    """Return ``True`` if the full dotted ``module`` path can be imported."""
    if sys.modules.get(module) is not None:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # ImportError: a parent package is missing. ValueError: odd entries in sys.modules.
        return False


def _install_hint(module: str) -> str:
    """Return the ``pip install`` target for ``module``."""
    for prefix, hint in _INSTALL_HINTS.items():
        if module == prefix or module.startswith(prefix + "."):
            return hint
    return module.split(".")[0]


def require_dependencies(*packages: str):
    """
    Decorator to check if required packages are installed before calling a function.

    Parameters
    ----------
    *packages : str
        Module names to check. Dot notation checks the full submodule path
        (e.g., ``"google.cloud.aiplatform"`` requires that exact module, not
        just ``google``).

    Returns
    -------
    Callable
        Decorated function that checks dependencies before execution.

    Raises
    ------
    ImportError
        If any required package is not installed. The message names the
        ``pip install`` target, e.g. ``caketool[gcp]`` for Vertex AI.

    Examples
    --------
    >>> @require_dependencies("mlflow")
    ... def train_with_mlflow():
    ...     import mlflow
    ...     # ...

    >>> @require_dependencies("google.cloud.aiplatform", "google.cloud.storage")
    ... def train_with_vertex():
    ...     from google.cloud import aiplatform
    ...     # ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing = [package for package in packages if not _is_installed(package)]
            if missing:
                hints = list(dict.fromkeys(_install_hint(package) for package in missing))
                raise ImportError(
                    f"Missing required dependencies: {', '.join(missing)}. "
                    f"Install with: pip install {' '.join(repr(h) if '[' in h else h for h in hints)}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
