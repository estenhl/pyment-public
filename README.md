# Installation


### Install pyenv and Python

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

The models in this repository expects Python version 3.10.4:
```
pyenv install 3.10.4
```
</details>

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
