import functools
import importlib


def get_class(name: str):
    """Import and return a class (or any attribute) by its fully qualified dotted name.

    Equivalent to a dynamic ``from a.b import C`` where the full path is
    given as a single string.  This is useful for deferred or configurable
    imports where the exact class is not known until runtime.

    Parameters
    ----------
    name : str
        Fully qualified dotted path, e.g. ``"category_encoders.TargetEncoder"``
        or ``"sklearn.linear_model.LogisticRegression"``.

    Returns
    -------
    type
        The resolved class or attribute object.

    Examples
    --------
    >>> TargetEncoder = get_class("category_encoders.TargetEncoder")
    >>> enc = TargetEncoder(smoothing=1.0)
    """
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def require_dependencies(*packages: str):
    """
    Decorator to check if required packages are installed before calling a function.

    Parameters
    ----------
    *packages : str
        Package names to check. Use dot notation for subpackages
        (e.g., "google.cloud.aiplatform").

    Returns
    -------
    Callable
        Decorated function that checks dependencies before execution.

    Raises
    ------
    ImportError
        If any required package is not installed.

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
            missing = []
            for package in packages:
                # Convert dot notation to top-level package for import check
                top_level = package.split(".")[0]
                try:
                    importlib.import_module(top_level)
                except ImportError:
                    missing.append(package)
            if missing:
                raise ImportError(
                    f"Missing required dependencies: {', '.join(missing)}. "
                    f"Install with: pip install {' '.join(missing)}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
