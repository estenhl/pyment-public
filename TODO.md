# TODO

Tracking outstanding work on `pyment-public`. Add new items as `- [ ]`; delete them when done — anything worth preserving long-term belongs in `CLAUDE.md`.

## Open

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
    - [ ] Upload SFCN-reg 2022 weights (`.data-00000-of-00001` + `.index`) via `scripts/upload_weights_to_github.py` and record the blob SHAs.
    - [ ] Add all new identifiers to `IDENTIFIERS` in `pyment/models/utils/ensure_weights.py`.
    - [ ] **Mirror the SHAs into CI workflows** — both `.github/workflows/finetune.yml` and `.github/workflows/preprocess-and-predict.yml` have a "Download weights" step that hardcodes the SHAs. Update both, or refactor those steps to call `ensure_weights` instead of duplicating the logic.
    - [ ] **Test each identifier** — extend `scripts/evaluate_ixi_predictions.py` for new multi-task identifiers, and add a smoke test (load → single forward pass → shape/range check) for the regression identifier which doesn't have IXI ground truth.
    - [ ] Update the "Pretrained weight resolution" section of `CLAUDE.md` to list all identifiers and the naming convention.
