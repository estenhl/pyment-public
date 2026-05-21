# Building docker containers

## Downloading pretrained weights

The `preprocess_and_predict` and `predict` images bundle pretrained weights at build time. Populate `checkpoints/pyment/` from the repo root before building:

```
# download all identifiers (multi-2025, multi-2025-no-abcd, reg-2025)
python scripts/download_weights.py --destination checkpoints/pyment

# or download a specific identifier only
python scripts/download_weights.py --identifiers multi-2025 --destination checkpoints/pyment
```

## Building docker container for preprocessing and predicting

Run from the repo root after downloading weights:

```
docker build \
    -f docker/preprocess_and_predict.Dockerfile \
    -t estenhl/pyment-preprocess-and-predict:<VERSION> \
    .
```

The weights identifier used at runtime defaults to `multi-2025` and can be overridden with the `PYMENT_WEIGHTS` environment variable:

```
docker run --env PYMENT_WEIGHTS=reg-2025 estenhl/pyment-preprocess-and-predict:<VERSION> ...
```

## Building docker container for preprocessing
Note that building the container requires a folder called `checkpoints` that contains the FastSurfer segmentation checkpoints in a subfolder called `fastsurfer`. This folder should contain the files `aparc_vinn_axial_v2.0.0.pkl`, `aparc_vinn_coronal_v2.0.0.pkl`, and `aparc_vinn_sagittal_v2.0.0.pkl`. The command should be run from the root of the repository:

```
docker build \
    -f docker/preprocess.Dockerfile \
    -t estenhl/pyment-preprocess:<VERSION> \
    .
```

## Building docker container for predicting from preprocessed images

Run from the repo root after downloading weights:

```
docker build \
    -f docker/predict.Dockerfile \
    -t estenhl/pyment-predict:<VERSION> \
    .
```
