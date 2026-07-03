# TODO

Tracking outstanding work on `pyment-public`. Add new items as `- [ ]`; delete them when done — anything worth preserving long-term belongs in `CLAUDE.md`.

## Open
- [ ] **Investigate run-to-run prediction drift in `preprocess-and-predict` CI**
  - The `.github/workflows/preprocess-and-predict.yml` fixture comparison was found to be flaky: re-running the same code/image against the same ds000030 raw inputs produces slightly different `multi-2025`/`reg-2025` predictions each time (~0.001-0.002 years mean diff), so `.github/workflows/fixtures/ds000030/predictions.csv` can't be pinned exactly. Worked around for now by loosening the compare step's `atol` to `1e-2`.
  - `pyment/data/utils/ensure_fastsurfer_crops_exists.py` (crop generation from `orig.mgz`/`mask.mgz`) is plain deterministic array math with no GPU/randomness, so the drift most likely originates upstream in FastSurfer's own preprocessing (`scripts/preprocess.sh` → FastSurferCNN segmentation/registration, GPU-based) rather than in `pyment`'s code — but this is unconfirmed.
  - To confirm: run `scripts/preprocess.sh` twice against the same raw input on the same machine, then diff the resulting `orig.mgz`/`mask.mgz` voxel arrays (e.g. via nibabel + `np.array_equal`/`max abs diff`). If they're bit-identical, the drift is coming from somewhere else (e.g. SFCN inference nondeterminism, or an environment mismatch) and the `atol` workaround should be revisited.
- [ ] **Docker + CI for antspynet preprocessing**
  - Add a Docker image analogous to `docker/preprocess_and_predict.Dockerfile` that bundles antspyx/antspynet + pyment, running `scripts/preprocess_folder_with_antspynet.py` instead of FastSurfer, then `pyment-predict` (flat format) against the result.
  - Add a GitHub Actions workflow analogous to `.github/workflows/preprocess-and-predict.yml` to build and smoke-test it end-to-end (e.g. against the same ds000030 fixtures), so the antspynet path gets the same CI coverage as the FastSurfer path.
- [ ] **Expose SFCN-reg 2022 weights** *(deferred — needs scoping)*
  - The SFCN-reg 2022 model was trained on a different preprocessing pipeline than the current FastSurfer-based flow, so serving it requires either a separate Docker image or a documented alternative preprocessing path. Needs a decision on whether to bother before any implementation work starts.
  - If pursued: upload weights via `scripts/upload_weights_to_github.py`, record blob SHAs, add identifier to `IDENTIFIERS`, add a smoke test (load → single forward pass → shape/range check).
