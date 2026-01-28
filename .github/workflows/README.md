## Run the Docker preprocess-and-predict sanity check locally
The GitHub Action for verifying the preprocess-and-predict container can be run locally with
```
act push \
    -W .github/workflows/preprocess-and-predict.yml \
    -s FREESURFER_LICENSE="$(cat $HOME/licenses/freesurfer.txt)" \
    -P ubuntu-latest=nvidia/cuda:12.4.1-base-ubuntu22.04 \
    --container-architecture linux/amd64 \
    --container-options "--gpus all" \
    --bind
```