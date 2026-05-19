# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`pyment` is the pretrained multi-task neuroimaging model from *Learning diverse and generic representations of the brain with large-scale multi-task pretraining* (medRxiv 2025). The package wraps an SFCN architecture trained on T1 MRI to predict age, sex, handedness, BMI, fluid intelligence, and neuroticism, and exposes finetuning + inference flows for downstream use. Current release: **v4.1.0**.

Python is hard-pinned to **3.10.x** with **TensorFlow 2.11** (`tensorflow-macos` on Darwin, plain `tensorflow` elsewhere). Do not upgrade either casually — the SFCN code uses TF 1.x-style `tf.keras.optimizers.legacy.*` (see `_LEGACY_OPTIMIZERS` in `pyment/cli/finetune_from_configuration.py`) and weight checkpoints are TF SavedModel format.

## Common commands

Environment setup uses **poetry** with **pyenv** for Python version control:
```
pyenv local 3.10.4
poetry env use 3.10.4
poetry install
eval $(poetry env activate)
```

Tests (pytest, configured via `tests/conftest.py` — fixture loads a sample nifti from `tests/fixtures/esten.nii.gz`):
```
pytest                                          # all tests
pytest tests/preprocessing/test_conform.py      # one file
pytest tests/preprocessing/test_conform.py::test_<name>   # one test
```

Tests in CI run inside a Docker container:
```
docker build -f docker/tests.Dockerfile -t pyment-tests .
docker run --rm pyment-tests
```

`tests.Dockerfile` does a full `pip install .` (all runtime deps, including tensorflow). Do not revert this to `--no-deps` to slim the image — the test suite imports tensorflow and pandas, so a partial install will silently skip or error those tests.

CLIs installed by `poetry install` (entry points in `pyproject.toml`):
```
pyment-predict  <fastsurfer-folder>  -d <out.csv>       # inference
pyment-finetune <configuration.json>                    # finetuning
```

Sanity-check IXI predictions (should yield MAE ≈ 3.12):
```
python scripts/evaluate_ixi_predictions.py
```

Linting and formatting use **ruff**. Pre-commit hooks run `ruff check` and `ruff format --check` on every `git commit` — they are **check-only** and abort the commit on violations without modifying files. Apply fixes manually:
```
ruff check --fix .            # apply safe autofixes
ruff format .                 # apply formatting
ruff check .                  # report-only (what pre-commit runs)
ruff format --check .         # report-only (what pre-commit runs)
```
Config lives in `[tool.ruff]` in `pyproject.toml` — line length 80, single quotes enforced (`quote-style = "single"`), conservative rule set (`E, F, W, I`). After cloning, contributors must run `pre-commit install` once to activate the hooks.

Code conventions beyond what ruff enforces are documented in [STYLE.md](./STYLE.md).

## Architecture

### Configuration-driven finetuning (the core pattern)

`pyment-finetune` validates a JSON file against a Pydantic `TrainingConfiguration` tree rooted in `pyment/configurations/training_configuration.py`. Each sub-config has a paired `.build()` method that constructs the runtime object — so the config tree mirrors the runtime object tree one-to-one:

| Config (pydantic)                          | Builds                                          |
|--------------------------------------------|-------------------------------------------------|
| `FastSurferDatasetConfiguration`           | `FastSurferDataset` (loads `labels.csv` + folders) |
| `SFCNConfiguration` (discriminated by `kind`) | `BinarySFCN` or `RegressionSFCN`              |
| `TargetConfiguration` (discriminated by `kind`) | regression target, or binary target encoder |
| `DataSplitConfiguration`                   | train/validation split                          |

**Asymmetry to be aware of:** `SFCNConfiguration` is a union of `sfcn-bin` and `sfcn-reg` only — `MultiTaskSFCN` is **not** wired into the finetune config path (it's inference-only via `sfcn_factory`). If you need multi-task finetuning, you'd add a `MultiTaskSFCNConfiguration` and extend the union.

The CI fixture configuration that exercises this path lives at `.github/workflows/fixtures/finetune_binary.json`.

### Model family

`pyment/models/sfcn/` defines the 3D conv trunk in `SFCN` (base, abstract `construct_prediction_head`), with three concrete heads:
- `RegressionSFCN` — single regression output
- `BinarySFCN` — single sigmoid output
- `MultiTaskSFCN` — 6 hard-coded heads concatenated along the last axis (order: `age, sex, handedness, bmi, fluid_intelligence, neuroticism`). The target list in `predict_from_fastsurfer_folder.py` mirrors this order; changing one without the other will silently mislabel predictions.

`sfcn_factory(model_type)` dispatches by string (`sfcn-reg` | `sfcn-bin` | `sfcn-multi`).

### Pretrained weight resolution

`SFCN.__init__(weights=...)` accepts either a local path-prefix or a known identifier. Resolution happens in `pyment/models/utils/ensure_weights.py`:
- Local path: looks for `<path>.index` + `<path>.data-00000-of-00001`.
- Identifier: looks it up in the `IDENTIFIERS` dict (currently only `multi-2025`) and downloads the two blobs by SHA from the GitHub blob API into `~/.pyment/weights/`.

CI workflows replicate this download mechanism inline (see the `Download weights` step in `.github/workflows/finetune.yml` and `preprocess-and-predict.yml`) — if you change `IDENTIFIERS`, update the workflow SHAs too.

### Data path: FastSurfer-preprocessed MRI

Inference and finetuning both consume FastSurfer subject folders. The pipeline:
1. Raw `.nii.gz` → FastSurfer preprocessing (external, run via `scripts/preprocess.sh` or the `pyment-preprocess` Docker image) → per-subject folder with `mri/orig.mgz` + `mri/mask.mgz`.
2. `ensure_fastsurfer_crops_exists` (`pyment/data/utils/`) lazily generates the model input `mri/crop.mgz` at `(224, 192, 224)`.
3. `FastSurferDataset.to_tensorflow_generator` loads crops via **`tensorflow_neuroimaging.loaders.load_mgh`** — an external dependency (`git+https://github.com/estenhl/tensorflow-neuroimaging`) flagged in the README as experimental. The `verify-mgh-loader` CLI from that repo is the recommended sanity check before any large finetune.

### CLIs vs Docker

Two parallel surfaces serve roughly the same goals:
- **Local CLIs** (`pyment-predict`, `pyment-finetune`) assume FastSurfer is already installed and preprocessing is done.
- **Docker images** (`docker/*.Dockerfile`) bundle FastSurfer + pyment so users only mount input/output/license volumes. The CI workflows test the Docker path end-to-end against the IXI dataset.

Both paths share the same Python code under `pyment/`.

### CI

All workflows in `.github/workflows/` target a **self-hosted Linux/x64 GPU runner** (`runs-on: [self-hosted, Linux, X64]`) and require the `FREESURFER_LICENSE` secret. They run on push to `main` and via `workflow_dispatch`. The recent commit history shows a long permission-related saga around these runners (see commits `3fdfbc0` through `43364a9`) — be cautious with any further changes to user/permission handling in the Dockerfiles.
