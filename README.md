# Caketool

- Environment

```bash
conda create -n caketool python=3.10
conda activate caketool
pip install -e ".[dev]"
pip-compile pyproject.toml
pip-sync
```

- Publish libs

Version is automatically derived from git tags. Just create a tag and push — GitHub Actions will build and publish.

```bash
# Test on TestPyPI
git tag v1.8.0-rc1
git push origin v1.8.0-rc1

# Publish to PyPI
git tag v1.8.0
git push origin v1.8.0
```

- Local development

```bash
python -m pip install -e .  # Install on local machine
python -c "from caketool import __version__; print(__version__)"  # Check version
```
