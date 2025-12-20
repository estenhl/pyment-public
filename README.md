---
title: pyment-public
model_file: artifacts/sfcn-multi.onnx
license: cc-by-nc-4.0
pipeline_tag: image-classification
tags:
  - neuroimaging
  - multi-task
  - structural t1 mri
  - brain age
---

# Installation

<details>
<summary>With pip</summary>
If you have a python 3.10.4 installation, ideally in a virtual environment, you can install the package with:

```
pip install git+https://github.com/estenhl/pyment-public
```

</details>
<details>
<summary>Manual installation</summary>
## Configure system
<details>
<summary>Ubuntu</summary>

First we need to download and install CUDA 11.2:
```
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
sudo sh cuda_11.2.2_460.32.03_linux.run --silent --toolkit --installpath=/usr/local/cuda-11.2
```

Next, cudnn must be installed. Download a suitable deb-file from
https://developer.nvidia.com/rdp/cudnn-archive. Then install the file:
```
sudo dpkg -i ~/Downloads/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev
sudo cp /usr/include/cudnn*.h /usr/local/cuda-11.2/include/
sudo cp -P /usr/lib/x86_64-linux-gnu/libcudnn*.so* /usr/local/cuda-11.2/lib64/
sudo ldconfig
```
Finally, we must configure the system paths in .bashrc:
```
echo 'export CUDA_HOME=/usr/local/cuda-11.2' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64' >> ~/.bashrc
```

</details>

## Install pyenv and Python

<details>
<summary> <b>Ubuntu</b> </summary>

On Ubuntu, install `pyenv` via `curl`:
```
curl https://pyenv.run | bash
```

After installation, add pyenv to the `~/.bashrc`-file to enable terminal shortcuts:
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
source ~/.bashrc
```

</details>
<details>
<summary> <b>macOS</b> </summary>
On macOS, install `pyenv` via `brew`:
```
brew update
brew install pyenv
```

After installation, add pyenv to the `~/.zshrc`-file to enable terminal shortcuts:
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init - zsh)"' >> ~/.zshrc
```

</details>

## Install correct Python version
The models in this repository expects Python version 3.10.4:
```
pyenv install 3.10.4
```

## Configure Python environment
Next, we can set up a python environment for running the code in the repository:
```
pyenv local 3.10.4
poetry env use 3.10.4
poetry install
```

## Activate the environment
The environment can be activated with:
```
eval $(poetry env activate)
```
</details>

# Tutorials
## Download the IXI dataset
All the approaches described below rely on having the IXI dataset downloaded. If you want to run the models on your own data you can skip this step, but will need to replace the path in the subsequent scripts accordingly. Otherwise, the IXI dataset can be downloaded via
```
python tutorials/download_ixi.py
```
## Generate predictions

<details>
<summary> Preprocess and predict with docker </summary>

Preprocessing and predicting with Docker relies on running the docker container that first runs FastSurfer preprocessing over all raw input images and then runs the model on the resulting preprocessed images. The docker-container will result in both a folder with preprocessed images, and a file predictions.csv containing all predictions.

Running the container relies on mounting three volumes:
- Inputs: A folder containing input data. All nifti-files detected in this folder or one of its subfolders will be processed
- Outputs: A folder where the preprocessed images and predictions will be written. This must be created prior to running the container
- Licenses: A folder containing the freesurfer license. The file must be named freesurfer.txt
```
mkdir -p ~/data/ixi/outputs
docker pull estenhl/pyment-preprocess-and-predict:1.0.0
docker run --rm -it \
    --user $(id -u):$(id -g) \
    --volume $HOME/data/ixi/images:/input \
    --volume $HOME/data/ixi/outputs:/output \
    --volume $HOME/licenses:/licenses \
    --gpus all \
    estenhl/pyment-preprocess-and-predict:1.0.0
```
</details>
<details>
<summary> Preprocess and predict in Python </summary>

Preprocessing and predicting manually relies on using the scripts provided in this repository to generate predictions via two steps

### Preprocessing
The images must be preprocessed using FastSurfer. First, FastSurfer must be downloaded. If any of the subsequent steps fail, a comprehensive installation-guide can be found in the [FastSurfer GitHub repository](https://github.com/Deep-MI/FastSurfer/blob/dev/doc/overview/INSTALL.md#native-ubuntu-2004-or-ubuntu-2204). The following steps downloads and installs FastSurfer into the folder `~/repos/fastsurfer`. First, some system packages must be installed:
```
sudo apt-get update && apt-get install -y --no-install-recommends wget git ca-certificates file
```
Next, we can clone FastSurfer, and change to the correct branch:
```
mkdir -p ~/repos
export FASTSURFER_HOME=~/repos/fastsurfer
git clone --branch stable https://github.com/Deep-MI/FastSurfer.git $FASTSURFER_HOME
(cd $FASTSURFER_HOME && git checkout v2.0.1)
```
Then we can create a python environment for fastsurfer, and install its dependencies. Note that the packages are installed using pip from the newly created virtual environment, not the system default:
```
mkdir -p ~/venvs
export FASTSURFER_VENV=~/venvs/fastsurfer
python -m venv $FASTSURFER_VENV
# The SimpleITK version in the requirements-file has been yanked, so we manually install a valid version prior to installing the remaining requirements.
$FASTSURFER_VENV/bin/pip install simpleitk==2.1.1.2
# SimpleITK then has to be removed from requirements.txt before installing the rest
grep -v "simpleitk==2.1.1" $FASTSURFER_HOME/requirements.txt | $FASTSURFER_VENV/bin/pip install -r /dev/stdin
```

Finally, we can run the preprocessing script, pointing towards the python from the virtual environment. Note that a valid freesurfer license must also be passed to this script, and that the $FASTSURFER_HOME variable must be set:
```
sh scripts/preprocess.sh --license <path-to-license> --python ~/venvs/fastsurfer/bin/python ~/data/ixi/images ~/data/ixi/preprocessed
```

### Generate predictions
After preprocessing, we can generate predictions for the IXI dataset using the scripts in the repository. First, ensure the virtual environment is loaded:
```
eval $(poetry env activate)
```
Next, make an output-folder for the predictions and run the prediction-script:
```
mkdir ~/data/ixi/outputs
python scripts/predict_from_fastsurfer_folder.py ~/data/ixi/preprocessed -d ~/data/ixi/outputs/predictions.csv
```
</details>

<details>
<summary> Preprocess and predict in two steps with docker </summary>
Preprocessing and predicting in two steps via docker requires using the two prebuilt docker containers for the two steps independently.

### Preprocessing
Running the container for preprocessing requires mounting three volumes:
- Inputs: A folder containing input data. All nifti-files detected in this folder or one of its subfolders will be processed
- Outputs: A folder where the preprocessed images will be written. This must be created prior to running the container
- Licenses: A folder containing the freesurfer license. The file must be named freesurfer.txt
```
mkdir -p ~/data/ixi/outputs
docker pull estenhl/pyment-preprocessing:1.0.0
docker run --rm \
    --user $(id -u):$(id -g) \
    --volume $HOME/data/ixi/images:/input \
    --volume $HOME/data/ixi/outputs:/output \
    --volume <path_to_licenses>:/licenses \
    --gpus all \
    estenhl/pyment-preprocessing:1.0.0
```

### Generate predictions
Running the container for predictions requires two volumes:
- Fastsurfer: The folder containing fastsurfer-processed images
- Outputs: The folder where the predictions are written
```
docker pull estenhl/pyment-predict:1.0.0
docker run --rm -it \
    --user $(id -u):$(id -g) \
    --volume $HOME/data/ixi/outputs/fastsurfer:/fastsurfer \
    --volume $HOME/data/ixi/outputs:/output \
    --gpus all \
    estenhl/pyment-predict:1.0.0
```

</details>

## Evaluate predictions

Evaluate the IXI predictions with
```
python tutorials/evaluate_ixi_predictions.py
```
If everything is set up correctly, this should yield an MAE of 3.12. Note that the paths to both the labels and predictions can be given as keyword arguments to the script if they don't reside in the standard locations.
