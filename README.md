# Installation


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

# Tutorials
## Download the IXI dataset
All the approaches described below rely on having the IXI dataset downloaded. If you want to run the models on your own data you can skip this step, but will need to replace the path in the subsequent scripts accordingly. Otherwise, the IXI dataset can be downloaded via
```
python tutorials/download_ixi.py
```
## Generate predictions
<details>
<summary> Preprocess and predict manually </summary>

Preprocessing and predicting manually relies on using the scripts provided in this repository to generate predictions via two steps

### Preprocessing
The images must be preprocessed using FastSurfer. First, FastSurfer must be downloaded. If any of the subsequent steps fail, a comprehensive installation-guide can be found in the [FastSurfer GitHub repository](https://github.com/Deep-MI/FastSurfer/blob/dev/doc/overview/INSTALL.md#native-ubuntu-2004-or-ubuntu-2204). The following steps downloads and installs FastSurfer into the folder `~/repos/fastsurfer`. First, some system packages must be installed:
```
sudo apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    file
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
</details>
