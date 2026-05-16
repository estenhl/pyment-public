# Contributing to pyment

Thanks for your interest in contributing.

## Reporting issues

Open an issue on GitHub with steps to reproduce, your platform, and the relevant package versions.

## Development setup

Set up a local environment as described in the README's [Build locally with poetry](./README.md#install-the-pyment-package) section.

### Pre-commit hook

`pre-commit` must be on your `PATH` when you run `git commit`. The `[dependency-groups]` dev entry installs `pre-commit` inside the poetry venv, but git hooks run from a regular shell that doesn't have the venv activated — so unless you `eval $(poetry env activate)` before every commit, you'll get a `pre-commit: not found` error.

The recommended fix is to install `pre-commit` globally with [pipx](https://pipx.pypa.io/) so it's always available regardless of which venv is active:

```
pipx install pre-commit
```

(Install pipx itself via your system package manager — `apt install pipx` on Debian/Ubuntu, `brew install pipx` on macOS, etc.)

Then, once per clone, activate the git hooks:

```
pre-commit install
```

This adds a check that runs before each `git commit`. The hook is **check-only** — it does not modify files. If lint or formatting issues are present, the commit aborts and you fix them deliberately.

## Code style

`pyment` uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. The pre-commit hook runs `ruff check` and `ruff format --check` on every commit. To fix issues locally:

```
ruff check --fix .
ruff format .
```

Project conventions beyond ruff (typing, docstrings, logging, etc.) live in [STYLE.md](./STYLE.md).

## Running tests

```
pytest tests/
```

A single file or test:

```
pytest tests/preprocessing/test_conform.py
pytest tests/preprocessing/test_conform.py::test_<name>
```

## Submitting changes

1. Fork the repo and create a feature branch from `main`.
2. Make your changes. Ensure tests pass and pre-commit hooks run clean.
3. Open a pull request against `main` with a clear description of what changed and why.
