# TODO

Tracking outstanding work on `pyment-public`. Add new items as `- [ ]`; delete them when done — anything worth preserving long-term belongs in `CLAUDE.md`.

## Open
- [ ] **Docker + CI for antspynet preprocessing**
  - Add a Docker image analogous to `docker/preprocess_and_predict.Dockerfile` that bundles antspyx/antspynet + pyment, running `scripts/preprocess_folder_with_antspynet.py` instead of FastSurfer, then `pyment-predict` (flat format) against the result.
  - Add a GitHub Actions workflow analogous to `.github/workflows/preprocess-and-predict.yml` to build and smoke-test it end-to-end (e.g. against the same ds000030 fixtures), so the antspynet path gets the same CI coverage as the FastSurfer path.
- [ ] **Expose SFCN-reg 2022 weights** *(deferred — needs scoping)*
  - The SFCN-reg 2022 model was trained on a different preprocessing pipeline than the current FastSurfer-based flow, so serving it requires either a separate Docker image or a documented alternative preprocessing path. Needs a decision on whether to bother before any implementation work starts.
  - If pursued: upload weights via `scripts/upload_weights_to_github.py`, record blob SHAs, add identifier to `IDENTIFIERS`, add a smoke test (load → single forward pass → shape/range check).
