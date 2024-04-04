Following (pending) release v3.0.0 (and onwards) this repository will serve solely as a model zoo for pretrained neuroimaging models from various publications. This entails that the utilities that were previously packaged alongside the models (e.g. for training models) has been stripped, to avoid bloating the repository. If you are interested in specific code for a specific paper either check out the previous releases or email me at [estenhl@uio.no](mailto:estenhl@uio.no)

### Publications
This is an overview of the publications from where the pretrained models originate. The shorthand-column denotes the name that is used to refer to the publications below. Note that the corresponding author is not necessarily equivalent to what is listed in the publication, but instead refers to the author in charge of the modelling.
| Title | Abbreviation | Publication year | Corresponding author |
| --- | :-: | :-: | :-: |
| [Deep neural networks learn general and clinically relevant representations of the ageing brain](https://doi.org/10.1016/j.neuroimage.2022.119210) | brain-age-general | 2022 | [Esten Høyland Leonardsen](mailto:estenhl@uio.no) |
| [Genetic architecture of brain age and its causal relations with brain and mental disorders](https://doi.org/10.1038/s41380-023-02087-y) | brain-age-genetics | 2023 | [Esten Høyland Leonardsen](mailto:estenhl@uio.no) |
| [Constructing personalized characterizations of structural brain aberrations in patients with dementia and mild cognitive impairment using explainable artificial intelligence](https://doi.org/10.1101/2023.06.22.23291592) | explainable-dementia | 2024 | [Esten Høyland Leonardsen](mailto:estenhl@uio.no) |

### Architectures
This is an overview of the model architectures used in the pretrained models.
| Name | Abbreviation | Description |
| --- | :-: | --- |
| RegressionSFCN | sfcn-reg | Base SFCN with a regression prediction head |
| RankingSFCN | sfcn-rank | Base SFCN with a ranking prediction head for regression problems |
| SoftClassificationSFCN | sfcn-sm | Base SFCN with a softmax prediction head, as per the original SFCN |
| BinarySFCN | sfcn-bin | Base SFCN with a binary prediction head for binary classification problems |

### Models
This is an overview of the actual pretrained models. The names are what should be used in the python-code to load the correct weights. Note that the names are not necessarily unique, but the tuple (name, architecture) is. The training set size refers to _samples_, not _participants_, and can thus have multiple session per participant.
| Name | Architecture | Source publication | Description | Training sample size | Expected out-of-sample error | URL |
| :-: | :-: | :-: | --- | :-: | :-: | :-: |
| brain-age-2022 | sfcn-reg | [brain-age-general](http://doi.org/10.1016/j.neuroimage.2022.119210) | Brain age regression model trained on heterogeneous dataset | 34285 | MAE=3.9 | [link](https://api.github.com/repos/estenhl/pyment-public/git/blobs/54b7f9545f1120cb302ff7342aaa724513f75219) |
| brain-age-2022 | sfcn-rank | [brain-age-general](http://doi.org/10.1016/j.neuroimage.2022.119210) | Brain age ranking model trained on heterogeneous dataset | 34285 | MAE=5.92 | [link](https://api.github.com/repos/estenhl/pyment-public/git/blobs/5d1bc5fc66327eb905acf81d9956f0391277b078) |
| brain-age-2022 | sfcn-sm | [brain-age-general](http://doi.org/10.1016/j.neuroimage.2022.119210) | Brain age soft classification model trained on heterogeneous dataset | 34285 | MAE=5.04 | [link](https://api.github.com/repos/estenhl/pyment-public/git/blobs/7b4f7bf4c989b80877b0bc0efe8b5125157788b5) |
| dementia-2024{-fold-X} | sfcn-bin | [explainable-dementia](http://doi.org/10.1101/2023.06.22.23291592) | Dementia classification model trained on multiple datasets.<br />Contains mostly patients with probable AD, but also other aetiologies<br />Fold number refers to the fold that were held out during training, if no fold is specified the first is used | 1366 |  | [fold 0](https://api.github.com/repos/estenhl/pyment-public/git/blobs/1f43aafd2461d7e5b4f9ebb6d62e0f2ab363e1b8)<br /> [fold 1](https://api.github.com/repos/estenhl/pyment-public/git/blobs/a0da6b724f3c1477ae2f461c49a91b7d2f46ac72)<br /> [fold 2](https://api.github.com/repos/estenhl/pyment-public/git/blobs/cec0eb79f043a3415f5ab13977dfda24e1f7dc30)<br /> [fold 3](https://api.github.com/repos/estenhl/pyment-public/git/blobs/c885fee44d4839d37d8bcdfd970391788ee85004)<br /> [fold 4](https://api.github.com/repos/estenhl/pyment-public/git/blobs/35d3b0343b83a9851a140cab7baed2dd36e35185)<br /> |