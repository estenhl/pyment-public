# Building docker containers

## Building docker container for preprocessing
Note that for now, building the container requires a folder called <checkpoints> that contains the FastSurfer segmentation checkpoints in a subfolder called `fastsurfer`. This folder should contain the files `aparc_vinn_axial_v2.0.0.pkl`, `aparc_vinn_coronal_v2.0.0.pkl`, and `aparc_vinn_sagittal_v2.0.0.pkl`. The command should be run from the root of the repository:

```
docker build \
    -f docker/preprocess.Dockerfile \
    -t estenhl/pyment-preprocessing:1.0.0 \
    .
```

## Building docker container for preprocessing
Note that for now, building the container requires a folder called <checkpoints> that contains the multi-task model checkpoints in a subfolder called `pyment`. This folder should contain the files `sfcn-multi.data-00000-of-00001`and `sfcn-multi.index`. The command should be run from the root of the repository:
```
docker build \
    -f docker/predict.Dockerfile \
    -t estenhl/pyment-predict:1.0.0 \
    .
```
