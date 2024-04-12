This is a repository containing pretrained models for neuroimaging data used in various scientific publications. The publications are listed [here](#publications), and all the models are listed [here](#models). After version 3.0.0 the nature of this repo changed, a description of which can be found in the [changelog](CHANGELOG.md). The models posted here try to mimic the behaviour and interface of the [pretrained models in the Keras applications package](https://keras.io/api/applications/). Besides the possibility of importing this library in Python and interacting with the models as Python-objects, we demonstrate three use cases for interaction here:
- [Jupyter notebooks](notebooks)
- [Command-line scripts](scripts)
- [Docker containers](docker)

## Installation
### Setup
Prior to installing the package and its prerequisites, it is recommended to create a virtual environment. This can be done via Anaconda, which can be downloaded from [here](https://docs.anaconda.com/free/anaconda/install/). After installation, an environment is created as follows:
```
conda create --name pyment python=3.9
```

### Via pip
Although the package is not currently available on PyPI, it is possible to install it via pip directly from git, using the following command:
```
pip install git+https://github.com/estenhl/pyment-public
```

### Manually
It is also possible to download the package from git and manually install it (the following demonstrates how its done on a Unix-based OS).
```
git clone git@github.com:estenhl/pyment-public.git
cd pyment-public
pip install -r requirements.txt
pip install -e .
```

### Verification
There are two ways the installation can be verified. By running the unit-tests (while still in the pyment-public folder):
```
pytest tests
```
If this fails, it is probably the prerequisites that has not been installed properly, rerun
the ```pip install -r requirements.txt```-command and monitor the output. You can also check that the package is properly installed on the system (not from the pyment-public folder):
```
python -c "import pyment"
```
If this fails, it is the ```pip install -e .```-command that failed, return to that and check for error messages.

## Publications
This is an overview of the publications from where the pretrained models originate. The shorthand-column denotes the name that is used to refer to the publications below. Note that the corresponding author is not necessarily equivalent to what is listed in the publication, but instead refers to the author in charge of the modelling.
| Title | Abbreviation | Publication year | Corresponding author | Citation |
| --- | :-: | :-: | :-: | :-: |
| [Deep neural networks learn general and clinically relevant representations of the ageing brain](https://doi.org/10.1016/j.neuroimage.2022.119210) | brain-age-general | 2022 | [Esten Høyland Leonardsen](mailto:estenhl@uio.no) | [.bib](citations/brain-age-general.bib) |
| [Genetic architecture of brain age and its causal relations with brain and mental disorders](https://doi.org/10.1038/s41380-023-02087-y) | brain-age-genetics | 2023 | [Esten Høyland Leonardsen](mailto:estenhl@uio.no) | [.bib](citations/brain-age-genetics.bib) |
| [Constructing personalized characterizations of structural brain aberrations in patients with dementia and mild cognitive impairment using explainable artificial intelligence](https://doi.org/10.1101/2023.06.22.23291592) | explainable-dementia | 2024 | [Esten Høyland Leonardsen](mailto:estenhl@uio.no) | [.bib](citations/explainable-dementia.bib) |

## Architectures
This is an overview of the model architectures used in the pretrained models.
| Name | Abbreviation | Description |
| --- | :-: | --- |
| RegressionSFCN | sfcn-reg | Base SFCN with a regression prediction head |
| RankingSFCN | sfcn-rank | Base SFCN with a ranking prediction head for regression problems |
| SoftClassificationSFCN | sfcn-sm | Base SFCN with a softmax prediction head, as per the original SFCN |
| BinarySFCN | sfcn-bin | Base SFCN with a binary prediction head for binary classification problems |

## Models
This is an overview of the actual pretrained models. The names are what should be used in the python-code to load the correct weights. Note that the names are not necessarily unique, but the tuple (name, architecture) is. The training set size refers to _samples_, not _participants_, and can thus have multiple session per participant.
| Name | Architecture | Source publication | Description | Training sample size | Expected out-of-sample error | URL |
| :-: | :-: | :-: | --- | :-: | :-: | :-: |
| brain-age-2022 | sfcn-reg | [brain-age-general](http://doi.org/10.1016/j.neuroimage.2022.119210) | Brain age regression model trained on heterogeneous dataset | 34285 | MAE=3.9 | [link](https://api.github.com/repos/estenhl/pyment-public/git/blobs/f87a66558433308bb8a5ecfb6aaa784811c5cd45) |
| brain-age-2022 | sfcn-rank | [brain-age-general](http://doi.org/10.1016/j.neuroimage.2022.119210) | Brain age ranking model trained on heterogeneous dataset | 34285 | MAE=5.92 | [link](https://api.github.com/repos/estenhl/pyment-public/git/blobs/5d1bc5fc66327eb905acf81d9956f0391277b078) |
| brain-age-2022 | sfcn-sm | [brain-age-general](http://doi.org/10.1016/j.neuroimage.2022.119210) | Brain age soft classification model trained on heterogeneous dataset | 34285 | MAE=5.04 | [link](https://api.github.com/repos/estenhl/pyment-public/git/blobs/7b4f7bf4c989b80877b0bc0efe8b5125157788b5) |
| dementia-2024-fold-X | sfcn-bin | [explainable-dementia](http://doi.org/10.1101/2023.06.22.23291592) | Dementia classification model trained on multiple datasets.<br />Contains mostly patients with probable AD, but also other aetiologies<br />Fold number refers to the fold that were held out during training, if no fold is specified the first is used. | 1366 |  | [fold 0](https://api.github.com/repos/estenhl/pyment-public/git/blobs/1f43aafd2461d7e5b4f9ebb6d62e0f2ab363e1b8)<br /> [fold 1](https://api.github.com/repos/estenhl/pyment-public/git/blobs/a0da6b724f3c1477ae2f461c49a91b7d2f46ac72)<br /> [fold 2](https://api.github.com/repos/estenhl/pyment-public/git/blobs/cec0eb79f043a3415f5ab13977dfda24e1f7dc30)<br /> [fold 3](https://api.github.com/repos/estenhl/pyment-public/git/blobs/c885fee44d4839d37d8bcdfd970391788ee85004)<br /> [fold 4](https://api.github.com/repos/estenhl/pyment-public/git/blobs/35d3b0343b83a9851a140cab7baed2dd36e35185)<br /> |

## License
The code and models in this repo is released under the [CC-BY-NC license](LICENSE.md) for <b>non-commercial</b> use.

## Citations
If you use a pretrained from this model, please cite the originating publication:


### brain-age-general
```
@article{leonardsen2022deep,
    title = {Deep neural networks learn general and clinically relevant representations of the ageing brain},
    author = {Esten H. Leonardsen and Han Peng and Tobias Kaufmann and Ingrid Agartz and Ole A. Andreassen and Elisabeth Gulowsen Celius and Thomas Espeseth and Hanne F. Harbo and Einar A. Høgestøl and Ann-Marie de Lange and Andre F. Marquand and Didac Vidal-Piñeiro and James M. Roe and Geir Selbæk and Øystein Sørensen and Stephen M. Smith and Lars T. Westlye and Thomas Wolfers and Yunpeng Wang},
    journal = {NeuroImage},
    volume = {256},
    pages = {119210},
    year = {2022},
    issn = {1053-8119},
    doi = {https://doi.org/10.1016/j.neuroimage.2022.119210},
    url = {https://www.sciencedirect.com/science/article/pii/S1053811922003342},
}
```


### explainable-dementia
```
@article {leonardsen2024constructing,
    title={Constructing personalized characterizations of structural brain aberrations in patients with dementia and mild cognitive impairment using explainable artificial intelligence},
	author={Esten H. Leonardsen and Karin Persson and Edvard Gr{\o}dem and Nicola Dinsdale and Till Schellhorn and James M. Roe and Didac Vidal-Pi{\~n}eiro and {\O}ystein S{\o}rensen and Tobias Kaufmann and Eric Westman and Andre Marquand and Geir Selb{\ae}k and Ole A. Andreassen and Thomas Wolfers and Lars T. Westlye and Yunpeng Wang and Alzheimer{\textquoteright}s Disease Neuroimaging Initiative and Australian Imaging Biomarkers and Lifestyle flagship study of ageing},
	journal={medRxiv},
    year={2024},
	publisher={Cold Spring Harbor Laboratory Press},
	doi={https://doi.org/10.1101/2023.06.22.23291592},
	url={https://www.medrxiv.org/content/early/2024/02/22/2023.06.22.23291592},
}
```


### brain-age-genetics
```
@article{leonardsen2023genetic,
    title={Genetic architecture of brain age and its causal relations with brain and mental disorders},
    author={Leonardsen, Esten H and Vidal-Pi{\~n}eiro, Didac and Roe, James M and Frei, Oleksandr and Shadrin, Alexey A and Iakunchykova, Olena and de Lange, Ann-Marie G and Kaufmann, Tobias and Taschler, Bernd and Smith, Stephen M and others},
    journal={Molecular Psychiatry},
    volume={28},
    number={7},
    pages={3111--3120},
    year={2023},
    publisher={Nature Publishing Group UK London},
    issn = {1476-5578},
    doi = {https://doi.org/10.1038/s41380-023-02087-y},
    url = {https://www.nature.com/articles/s41380-023-02087-y},
}
```