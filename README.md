# Installation
<details>
<summary> <b>macOS</b> </summary>

### Install pyenv
To select the appropriate python version, ```pyenv``` must be installed. On macOS this can be done via brew:
```
brew update
brew install pyenv
```
After installation we can add pyenv to the ```~/.zshrc```-file to enable terminal shortcuts:
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init - zsh)"' >> ~/.zshrc
```

### Install Python
The models in this repository expects Python version 3.10.4, which can be set up using ```pyenv```:
```
pyenv install 3.10.4
```

### Configure Python environment
Next, we can set up a python environment for running the code in the repository:
```
poetry env use 3.10.4
poetry install
```

### Activate the environment
The environment can be activated with:
```
eval $(poetry env activate)
```

</details>