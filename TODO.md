# TODO

Tracking outstanding work on `pyment-public`. Add new items as `- [ ]`; tick them off as `- [x]` when done. Keep each task verbose enough that you (or a future Claude Code session) can pick it up cold without rereading old conversation.

When a task lands a fact that future sessions should know (a chosen tool, a new convention, a release process), capture that fact in `CLAUDE.md` as part of finishing the task.

## Open

- [ ] **Diagnose push-CI failures in the Docker workflows**
  - `finetune.yml` and `preprocess-and-predict.yml` succeed on `workflow_dispatch` but consistently fail at the **"Run container"** step on push triggers. Every push between `3fdfbc0` and `43364a9` showed the same failure mode; `tests.yml` is unaffected and remains green.
  - The license validates, GPU sanity check passes, the image builds — only the container itself fails to run on push. Possible causes (none confirmed without log access): self-hosted runner workspace state being cleaner on push, concurrent-run contention for `/dev/nvidiactl`, timing between `Download weights` and `Run container`, or secret context differences.
  - First step: install `gh` in the devcontainer (or use a PAT) and pull the failing job logs to see the actual container exit reason. Without that, this is guesswork.

- [ ] **Lint sweep across the codebase (ruff)**
  - Setup is complete: ruff configured in `[tool.ruff]` (line length 80, single quotes, rules `E F W I G`, max-doc-length 72), pre-commit hook installed (check-only via `.pre-commit-config.yaml`), STYLE.md captures project-level conventions beyond ruff, pre-commit+pipx workflow documented in CONTRIBUTING.md.
  - Remaining: apply `ruff format` and `ruff check --fix .` across the codebase, plus enforce STYLE.md conventions (docstrings on publics, blank line after docstrings, `-> None`, `basicConfig` only in `main()`, `%`-style log placeholders, etc.) as each file is touched. Doing file-by-file in lexicographic order. Folder progress:
    - [x] `pyment/cli/` — `finetune_from_configuration.py` and `predict_from_fastsurfer_folder.py` done; `predict_from_bids_folder.py` deleted as orphan; unit tests added for `_resolve_optimizer` and `_parse_folder_name`.
    - [ ] `pyment/configurations/`
    - [ ] `pyment/data/`
    - [ ] `pyment/factories/`
    - [ ] `pyment/metrics/`
    - [ ] `pyment/models/`
    - [ ] `pyment/preprocessing/`
    - [ ] `pyment/utils/`
    - [ ] `pyment/__init__.py` and the `__init__.py` files under each subpackage (also add `__all__` to those that re-export symbols)
    - [ ] `scripts/`
    - [ ] `tests/` (existing files only — new test files added during the sweep are already clean)

- [ ] **Make `pytest tests` work without explicit `python -m`**
  - Bare `pytest tests` fails with `ModuleNotFoundError: No module named 'pyment'` in fresh shells. Two causes:
    1. The `pytest` binary doesn't auto-add CWD to `sys.path` — only `python -m pytest` does.
    2. `pyment` isn't reliably installed in site-packages — `pip show pyment` often returns "not found" despite the devcontainer's `postCreateCommand` running `pip install -e .`.
  - Two fixes worth applying:
    - Add `[tool.pytest.ini_options] pythonpath = ["."]` to `pyproject.toml`. Makes `pytest tests` work regardless of install state.
    - Investigate why the editable install evaporates — likely candidates: the chown loop overwriting metadata permissions, or some poetry/pip interaction with `POETRY_VIRTUALENVS_CREATE=false`.
  - Workaround until fixed: use `python -m pytest tests` instead of bare `pytest tests`.

- [ ] **Write a proper finetuning tutorial**
  - The finetuning surface (`pyment-finetune`) is currently only exercised by the vibe-coded GitHub Action in `.github/workflows/finetune.yml`. There is no user-facing tutorial — anyone trying to finetune locally has to reverse-engineer the config schema from `pyment/configurations/training_configuration.py`.
  - Tutorial should cover:
    - How to author a configuration JSON. The previous example configs (`configurations/local/finetune_ixi_*.json`) were deleted during the fixture reorg — author fresh worked examples for the tutorial. The CI fixture at `.github/workflows/fixtures/finetune_binary.json` is a working starting point to crib from.
    - What each top-level field in `TrainingConfiguration` controls (`dataset`, `data_split`, `model`, `target`, `batch_size`, `num_threads`, `loss`, `metrics`, `optimizer`, `epochs`, `destination`).
    - The `kind`-discriminator pattern in `SFCNConfiguration` and `TargetConfiguration` — currently `sfcn-bin`/`sfcn-reg` and `binary`/`regression`.
    - How to invoke: `pyment-finetune <config.json>`.
    - Output artifacts produced under `destination/`: `model/` (SavedModel), `history.json`, `predictions.csv`.
  - Probably belongs in `README.md` under a new "Finetuning" section, or as a standalone doc under a `docs/` directory.

- [ ] **Add new pretrained weight identifiers**
  - Extend the `IDENTIFIERS` dict in `pyment/models/utils/ensure_weights.py` with two new entries:
    - **sfcn-reg pretrained weights** — for the regression head (`RegressionSFCN`). Currently no identifier exists for it; only `multi-2025` (multi-task) is registered.
    - **Multi-task weights from a non-ABCD training session** — complements the existing `multi-2025` (which is the ABCD-trained run). Pick a naming convention that disambiguates the training cohort before adding — e.g. `multi-<dataset>-<year>`. Worth retroactively renaming `multi-2025` → `multi-abcd-2025` for symmetry; consider whether to keep `multi-2025` as a deprecated alias.
  - For each entry: upload the SavedModel `.data-00000-of-00001` and `.index` blobs to the repo and record their git blob SHAs (the same scheme used today — see `scripts/upload_weights_to_github.py`).
  - **Mirror the SHAs into CI workflows** — both `.github/workflows/finetune.yml` and `.github/workflows/preprocess-and-predict.yml` have a "Download weights" step that hardcodes the SHAs. Update both, or refactor those steps to call `ensure_weights` instead of duplicating the logic.
  - **When done:** update the "Pretrained weight resolution" section of `CLAUDE.md` to list the new identifiers and the naming convention.

## Done

<!-- Move tasks here when complete, with a short note on what landed. -->

- [x] **Finish session loose ends (cleanup)** — 2026-05-15
  - README path fixed (`tutorials/download_ixi.py` → `scripts/download_ixi.py`).
  - CI verified: `tests.yml` green on push, but the two Docker workflows fail on push triggers while succeeding on manual dispatch. Split out as its own open task above ("Diagnose push-CI failures...").

- [x] **Repository structure reorg** (ad-hoc, not previously planned) — 2026-05-15
  - Moved fixtures next to their consumers: `tests/fixtures/esten.nii.gz`, `.github/workflows/fixtures/{finetune_binary.json, ixi/...}`.
  - Deleted three orphan files: `configurations/local/finetune_ixi_{binary,regression}.json` and `data/ixi/finetune_config.json`.
  - Removed empty top-level `data/` and `configurations/` directories.
  - Updated consumers (`tests/conftest.py`, both workflow YAMLs, `CLAUDE.md`). Tests still pass (12/12).
  - Patch version bump: `pyproject.toml` 4.1.0 → 4.1.1 (no changelog entry, consistent with how v3.0.1 was handled).
