## Build docker container for preprocessing
Note that for now, building the container requires a folder called <checkpoints> that contains the FastSurfer segmentation checkpoints

```
docker build \
    -f docker/preprocess.Dockerfile \
    -t pyment/preprocessing:1.0.0 \
    .
```

## Run docker container for preprocessing
Running the container for preprocessing requires three volumes:
- Inputs: A folder containing input data. All nifti-files detected in this folder or one of its subfolders will be processed
- Outputs: A folder where the preprocessed images will be written.
- Licenses: A folder containing the freesurfer license
```
docker run --rm \
    --user $(id -u):$(id -g) \
    --volume <path_to_input>:/input \
    --volume <path_to_ouput>:/output \
    --volume <path_to_licenses>:/licenses \
    --gpus all \
    pyment/preprocessing:1.0.0
```
