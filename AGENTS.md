# Caketool — AGENTS.md

Guidance for AI coding agents and human contributors working in this repo.

Caketool is a Python toolkit for data scientists. It covers the lifecycle of a machine
learning or deep learning problem: exploratory analysis, feature engineering, training,
evaluation, explainability, calibration, monitoring and experiment tracking. Today it
focuses on tabular data and gradient-boosted trees; it is designed to grow into deep
learning and other data types without making the core heavier.

Supports Python 3.10–3.12. Published to [PyPI](https://pypi.org/project/caketool/); docs
(blog + API reference) at <https://mazino2d.github.io/caketool>.

## Language policy

Write everything in English: code, comments, docstrings, docs, notebooks, commit messages
and PR descriptions.

## Design principles

1. **Light core, heavy extras.** The core install stays small. A new framework or service
   (PyTorch, a cloud SDK, a tracker, a DataFrame engine) goes behind an optional extra:
   - declare it in `pyproject.toml` under `[project.optional-dependencies]` and in `all`;
   - add its install hint to `_INSTALL_HINTS` in `src/caketool/utils/lib_utils.py`;
   - import it inside the function that needs it, never at module top level, and guard
     that function with `@require_dependencies(...)`;
   - regenerate `requirements.txt` with `pip-compile --all-extras pyproject.toml`.
2. **Familiar APIs.** Follow scikit-learn conventions (`fit` / `transform` / `predict`,
   `fit` returns `self`, hyperparameters in `__init__`). Where it helps, offer both a
   stateless function and a fitted estimator for the same operation.
3. **ML correctness first.** A result that looks good but is wrong is the worst bug here.
   - No leakage: fit statistics, encoders and early stopping on training data (or the
     training part of each fold) only; time-based features use only rows strictly before
     the snapshot time.
   - Reproducible: anything random takes a `random_state` / seed.
   - Robust to messy data: NaN, infinities, constant columns, bool and categorical dtypes,
     empty inputs. Fail with a clear error rather than return a silently wrong number.
   - Backend parity: when a function supports several backends, every backend returns
     the same result for the same input.
   - Compatible with pandas 2 and 3.
4. **Stable public API.** Users import from the package namespaces (`caketool.model`,
   `caketool.metric`, …). Renaming or removing a public name is a breaking change: keep a
   deprecated alias for at least one release and mark the commit with `!`.

## Commands

```bash
# Setup
conda create -n caketool python=3.10 && conda activate caketool
pip install -e ".[dev,all]"
pre-commit install                    # ruff + ruff-format, nbstripout for notebooks

# Lint, format, test (what CI runs)
ruff check src/ tests/ --fix
ruff format src/ tests/
pytest tests/ -v --tb=short

# Dependencies
pip-compile --all-extras pyproject.toml   # refresh requirements.txt after changing deps

# Docs
pip install -e ".[docs]"
pdoc src/caketool                     # preview the API reference at http://localhost:8080
```

The docs site is MkDocs (blog in `docs/`) plus the pdoc API reference built into
`docs/api/`. `mkdocs.yml` inherits `.mkdocs/base.yml`, which CI fetches from
`mazino2d/github-workflows`; a local `mkdocs serve` needs that file.

## Layout

```text
src/caketool/<domain>/   One package per stage of the workflow (eda, feature, model, metric,
                         explainability, calibration, monitor, report, experiment, …).
                         Public names are re-exported in the package __init__.py.
src/caketool/utils/      Shared helpers: dynamic imports, optional-dependency checks
tests/test_<module>.py   pytest suite; pythonpath = src, so import caketool directly
demos/                   Example notebooks (outputs stripped on commit, excluded from ruff)
docs/blog/posts/         "ML Modeling 101" blog posts; docs/index.md lists them
```

A new stage of the workflow (for example deep learning models or text features) gets its
own package under `src/caketool/`, following the same structure.

## Code conventions

- Type hints with Python 3.10+ syntax: `X | None`, `list[str]`, `dict[str, float]`,
  `Literal["a", "b"]` for constrained strings.
- numpy-style docstrings (`Parameters`, `Returns`, `Raises`, `Examples`) on every public
  function and class; the API reference is generated from them.
- Re-export public names explicitly: `from .module import Name as Name` (or list them
  in `__all__`).
- Naming: `PascalCase` classes, `snake_case` functions, `UPPER_SNAKE_CASE` constants,
  `_prefix` for private helpers.
- Ruff with line length 120 (`pyproject.toml`); do not add other formatters.
- When the public API changes, update the package docstring and the README in the same PR.

## Testing

- pytest, tests grouped in classes (`class TestFoo`) with fixtures for shared data.
- Every bug fix comes with a test that fails without it; new code covers the edge cases
  in principle 3.
- Optional backends are tested with mocks injected through `sys.modules` (see
  `tests/test_experiment_tracker.py`) so the suite runs without cloud credentials.
- Keep tests fast and deterministic: small synthetic data, fixed seeds.

## Workflow

- Never push to `main`. Branch → PR → CI → squash merge.
- Commits and PR titles use Conventional Commits (`feat(model): …`, `fix(metric): …`);
  `!` marks a breaking change.
- CI (`.github/workflows/`) calls the shared workflows in `mazino2d/github-workflows`.
  The required check `ci / gate` runs ruff and pytest on Python 3.10; `ci-compat` runs
  pytest on 3.11 and 3.12.
- Releases: the version comes from git tags (setuptools-scm). Pushing `vX.Y.Z` publishes
  to PyPI; a tag containing `rc` publishes to TestPyPI.

## Definition of Done

- `ruff check` and `pytest` pass locally; CI is green.
- New behaviour and bug fixes have tests, including edge cases.
- Docstrings, package docstring and README match the new API.
- New dependencies sit behind an extra and `requirements.txt` is regenerated.
