- Enviroment

```bash
conda create -n caketool
conda activate caketool
pip-compile pyproject.toml
pip-sync
pip install pip-tools bumpver build twine ipykernel
```

- Publish libs

```bash
bumpver update --major # Increment the MAJOR version when you make incompatible API changes.
bumpver update --minor # Increment the MINOR version when you add functionality in a backwards compatible manner.
bumpver update --patch # Increment the PATCH version when you make backwards compatible bug fixes.
python -m pip install -e . # Install on local machine
rm ./dist/*
python -m build
twine upload -r testpypi dist/*
twine upload dist/*
pip install caketool
```