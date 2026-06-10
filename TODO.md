# TODO

Tracking outstanding work on `pyment-public`. Add new items as `- [ ]`; delete them when done — anything worth preserving long-term belongs in `CLAUDE.md`.

## Open

- [ ] **Replace IXI with ds000030 (OpenNeuro)** — the IXI download URL (biomedic.doc.ic.ac.uk) now returns 403. Replacement: [ds000030](https://openneuro.org/datasets/ds000030) (UCLA Consortium for Neuropsychiatric Phenomics, 272 subjects: healthy controls + schizophrenia/bipolar/ADHD). `participants.tsv` provides `age` (continuous) and `diagnosis` (4-class, collapsible to binary) — suitable targets for tutorial finetuning studies.
  - [x] Replace `scripts/download_ixi.py` and `scripts/create_ixi_labels.py` with a single `scripts/download_dataset.py` that downloads a configurable number of ds000030 subjects via the OpenNeuro S3/GitHub raw URL and writes a `labels.csv` with `participant_id`, `age`, and `diagnosis` columns
  - [x] Rewrite `scripts/evaluate_ixi_predictions.py` for ds000030 — update subject-ID parsing (BIDS `sub-XXXXX` format) and metadata join to use `participants.tsv` columns
  - [ ] Select a small fixed sample of ds000030 subjects for CI fixtures; replace NIfTIs in `.github/workflows/fixtures/ixi/raw/` with the new subjects and rename the folder to `fixtures/ds000030/`
  - [ ] Regenerate `labels.csv` fixture for the new sample subjects
  - [ ] Regenerate `predictions.csv` fixture — run the full preprocess+predict pipeline on the new subjects to get reference outputs (requires self-hosted GPU runner)
  - [ ] Regenerate or drop `fixtures/ixi/fastsurfer/` — 3 preprocessed subjects tracked in git; regenerate with ds000030 subjects if still needed by any workflow step
  - [ ] Update `.github/workflows/fixtures/finetune_binary.json` — update target field to match the new `labels.csv` columns (`diagnosis` binary or `age` regression)
  - [ ] Update `.github/workflows/preprocess-and-predict.yml` — replace `fixtures/ixi/` path references with `fixtures/ds000030/`
  - [ ] Update `.github/workflows/finetune.yml` — replace `fixtures/ixi/` path references with `fixtures/ds000030/`
  - [x] Update `README.md` — replace all ~15 IXI references across the download, preprocess, predict, evaluate, and finetune tutorial sections

- [ ] **Use `from_logits=True` for numerical stability in classification heads** — `BinarySFCN` and `CategoricalSFCN` currently apply their activation (`sigmoid`/`softmax`) inside the prediction head, so losses must use `from_logits=False`. Removing the activations from the heads and switching to `from_logits=True` would let TensorFlow fuse the activation and cross-entropy into a numerically stable `log_softmax` / `log_sigmoid` operation. Requires updating both model classes and the example configs.

- [ ] **Expose SFCN-reg 2022 weights** *(deferred — needs scoping)*
  - The SFCN-reg 2022 model was trained on a different preprocessing pipeline than the current FastSurfer-based flow, so serving it requires either a separate Docker image or a documented alternative preprocessing path. Needs a decision on whether to bother before any implementation work starts.
  - If pursued: upload weights via `scripts/upload_weights_to_github.py`, record blob SHAs, add identifier to `IDENTIFIERS`, add a smoke test (load → single forward pass → shape/range check).
