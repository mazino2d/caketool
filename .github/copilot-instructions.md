# Caketool - AI Instructions

Caketool is a Python MLOps library providing reusable tools for feature engineering, model training, calibration, and monitoring.

## Build & Test Commands

```bash
# Setup environment
conda create -n caketool python=3.10
conda activate caketool
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/

# Lint & format
ruff check src/ tests/ --fix
ruff format src/ tests/
```

## Architecture

| Module | Purpose |
|--------|---------|
| `calibration/` | Score calibration via statistical transformations |
| `feature/` | Feature engineering: encoding, removal, infinity handling |
| `metric/` | Evaluation metrics: gini, PSI, sklearn wrappers |
| `model/` | Model wrappers: BoostTree (XGBoost), VotingModel, ensembles |
| `monitor/` | Drift detection, BigQuery-based monitoring |
| `report/` | Risk scoring and probability band analysis |
| `experiment/` | Google Cloud AI Platform experiment tracking |
| `utils/` | Helpers for arrays, strings, numbers, phone parsing, BigQuery |

## Code Conventions

### Language
- Use English only for all code, documentation, and content: variable names, function names, class names, comments, docstrings, commit messages, README, notebooks, and any other project files

### Type Annotations
- Use Python 3.10+ union syntax: `list | tuple | np.ndarray`
- Use lowercase built-in generics: `list[str]`, `dict[str, float]`
- Use `Literal` for constrained strings: `Literal["bins", "quantiles"]`

### Docstrings
Use numpy-style docstrings:
```python
def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI).
    
    Parameters
    ----------
    expected : np.ndarray
        Array of expected values.
    actual : np.ndarray
        Array of actual values.
    n_bins : int, optional
        Number of bins (default: 10).
    
    Returns
    -------
    float
        PSI value.
    """
```

### Module Exports
Re-export public API in `__init__.py` using explicit `as NAME`:
```python
from .feature_encoder import FeatureEncoder as FeatureEncoder
from .feature_remover import ColinearFeatureRemover as ColinearFeatureRemover
```

### Naming
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_prefix`

## Testing Patterns

- Tests in `tests/` using pytest
- Import source modules using `src.` prefix:
  ```python
  from src.caketool.some.module import SomeClass
  ```
- Use class-based test grouping (`class TestFoo`) with pytest fixtures
