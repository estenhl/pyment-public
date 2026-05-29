# TODO

Tracking outstanding work on `pyment-public`. Add new items as `- [ ]`; delete them when done — anything worth preserving long-term belongs in `CLAUDE.md`.

## Open

- [ ] **Fix `tf.Assert` calls in `load_mgh` header parsing**
  - `tf.Assert` returns an op that must be part of the computation graph to be evaluated. The current calls in `_parse_header` (version check, rasflag check) discard the return value, making them silent no-ops in graph mode (`tf.function`). Replace with `tf.debugging.assert_equal` / `tf.debugging.assert_greater` which integrate into the graph automatically, or wrap with `tf.control_dependencies`.

- [ ] **Fix `load_mgh` reshape order to work for non-square volumes**
  - The current code reshapes the flat image bytes to `(W, H, D)` then transposes `[2, 1, 0]` to `(D, H, W)`. This accidentally matches nibabel for volumes where W == D (e.g. 224×192×224 crops) but produces wrong values for asymmetric volumes. Fix: reshape directly to `(D, H, W)` (the actual on-disk byte order) so the `[2, 1, 0]` transpose correctly yields `(W, H, D)` matching nibabel for any shape.

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
  - Blobs already uploaded for two new identifiers; remaining steps:
    - [ ] Add all new identifiers to `IDENTIFIERS` in `pyment/models/utils/ensure_weights.py`.
    - [ ] **Mirror the SHAs into CI workflows** — both `.github/workflows/finetune.yml` and `.github/workflows/preprocess-and-predict.yml` have a "Download weights" step that hardcodes the SHAs. Update both, or refactor those steps to call `ensure_weights` instead of duplicating the logic.
    - [ ] **Test each identifier** — extend `scripts/evaluate_ixi_predictions.py` for new multi-task identifiers.
    - [ ] Update the "Pretrained weight resolution" section of `CLAUDE.md` to list all identifiers and the naming convention.

- [ ] **Expose SFCN-reg 2022 weights** *(deferred — needs scoping)*
  - The SFCN-reg 2022 model was trained on a different preprocessing pipeline than the current FastSurfer-based flow, so serving it requires either a separate Docker image or a documented alternative preprocessing path. Needs a decision on whether to bother before any implementation work starts.
  - If pursued: upload weights via `scripts/upload_weights_to_github.py`, record blob SHAs, add identifier to `IDENTIFIERS`, add a smoke test (load → single forward pass → shape/range check).
