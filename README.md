## Build docker container for preprocessing
docker build \
    -f docker/preprocess.Dockerfile \
    -t pyment/preprocessing:1.0.0 \
    --build-arg CHECKPOINTS_FOLDER=<path_to_fastsurfer_checkpoints>
    .

## Run docker container for preprocessing
docker run --rm -v <path_to_input>:/input -v <path_to_ouput>:/output -v <path_to_licenses>:/licenses --gpus all pyment/preprocessing:1.0.0
