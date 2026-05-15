# TODO

Tracking outstanding work on `pyment-public`. Add new items as `- [ ]`; tick them off as `- [x]` when done. Keep each task verbose enough that you (or a future Claude Code session) can pick it up cold without rereading old conversation.

When a task lands a fact that future sessions should know (a chosen tool, a new convention, a release process), capture that fact in `CLAUDE.md` as part of finishing the task.

## Open

- [ ] **Diagnose push-CI failures in the Docker workflows**
  - `finetune.yml` and `preprocess-and-predict.yml` succeed on `workflow_dispatch` but consistently fail at the **"Run container"** step on push triggers. Every push between `3fdfbc0` and `43364a9` showed the same failure mode; `tests.yml` is unaffected and remains green.
  - The license validates, GPU sanity check passes, the image builds — only the container itself fails to run on push. Possible causes (none confirmed without log access): self-hosted runner workspace state being cleaner on push, concurrent-run contention for `/dev/nvidiactl`, timing between `Download weights` and `Run container`, or secret context differences.
  - First step: install `gh` in the devcontainer (or use a PAT) and pull the failing job logs to see the actual container exit reason. Without that, this is guesswork.

- [ ] **Pick a linting/formatting tool**
  - Nothing is configured today — no `ruff`, `black`, `flake8`, or `.pre-commit-config.yaml` in the repo.
  - Decide between **ruff** (modern, fast, single dep, replaces flake8 + isort + pyupgrade and has a built-in formatter) and **black + flake8** (traditional split, very stable). Other options on the table if either feels wrong.
  - Once chosen: add as a dev dependency in `pyproject.toml`, commit a config block, run it across `pyment/`, `scripts/`, and `tests/`, and consider adding a pre-commit hook or a CI check so it actually stays applied.
  - **When done:** document the chosen tool and how to invoke it (lint, format, fix) in `CLAUDE.md` under "Common commands".

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
