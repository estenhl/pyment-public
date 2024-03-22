Following (pending) release v3.0.0 (and onwards) this repository serves solely as a model zoo for pretrained neuroimaging models from various publications (see list below). This means that a lot of the utilities that was previously packaged alongside the models (for e.g. training) has been stripped to avoid bloating the repository. If you are interested in specific code for a specific paper either check out the previous releases or email me at [estenhl@uio.no](mailto:estenhl@uio.no)

### Publications
This is an overview of the publications from where the pretrained models originate. The shorthand-column denotes the name that is used to refer to the publications below. Note that the corresponding author is not necessarily equivalent as for the publication, but instead the author in charge of the modelling.
| Title | Shorthand title | Publication year | Corresponding author |
| --- | --- | --- | --- |
| [Deep neural networks learn general and clinically relevant representations of the ageing brain](10.1016/j.neuroimage.2022.119210) | brain-age-general | 2022 | [E.H. Leonardsen](mailto:estenhl@uio.no) |
| [Genetic architecture of brain age and its causal relations with brain and mental disorders](https://doi.org/10.1038/s41380-023-02087-y) | brain-age-genetics | 2023 | [E.H. Leonardsen](mailto:estenhl@uio.no) |
| [Constructing personalized characterizations of structural brain aberrations in patients with dementia and mild cognitive impairment using explainable artificial intelligence](https://doi.org/10.1101/2023.06.22.23291592) | dementia-explainable | 2024 | [E.H. Leonardsen](mailto:estenhl@uio.no) |

### Architectures
This is an overview of the model architectures used in the pretrained models.
| Name | Type | Publications |
| --- | --- | --- |
| SFCN-reg | Regression | brain-age-general, brain-age-genetics |
| SFCN-rank | Ranking | brain-age-general |
| SFCN-sm | Soft classification | brain-age-general |
| SFCN-bin | Binary classification | dementia-explainable |

### Models
This is an overview of the actual pretrained models. The names are what should be used in the python-code to load the correct weights. Note that the names are not necessarily unique, but the tuple (name, architecture) is. The training set size refers to _samples_, not _participants_, and can thus have multiple session per participant.
| Name | Architecture | Publication | Description | Training set size | Expected out-of-sample performance |
| --- | --- | --- | --- | --- | --- |
| brain-age-2022 | SFCN-reg | brain-age-general | Brain age regression model trained on heterogeneous dataset | 34,285 | MAE=3.9 |
| brain-age-2022 | SFCN-rank | brain-age-general | Brain age ranking model trained on heterogeneous dataset | 34,285 | MAE=5.92 |
| brain-age-2022 | SFCN-sm | brain-age-general | Brain age soft classification model trained on heterogeneous dataset | 34,285 | MAE=5.04 |
| brain-age-2023-fold-1 | SFCN-reg | brain-age-genetics | Brain age regression model trained on heterogeneous dataset (fold 1 from the data split used in the publication held out of training) | | |
| brain-age-2023-fold-2 | SFCN-reg | brain-age-genetics | Brain age regression model trained on heterogeneous dataset (fold 2 from the data split used in the publication held out of training) | | |
| brain-age-2023-fold-3 | SFCN-reg | brain-age-genetics | Brain age regression model trained on heterogeneous dataset (fold 3 from the data split used in the publication held out of training) | | |
| brain-age-2023-fold-4 | SFCN-reg | brain-age-genetics | Brain age regression model trained on heterogeneous dataset (fold 4 from the data split used in the publication held out of training) | | |
| brain-age-2023-fold-5 | SFCN-reg | brain-age-genetics | Brain age regression model trained on heterogeneous dataset (fold 5 from the data split used in the publication held out of training) | | |
| dementia-2024-fold-1 | SFCN-bin | dementia-explainable | Dementia classification model trained on multiple datasets. Contains mostly patients with probable AD, but also other aetiologies (fold 1 from the data split used in the publication held out of training) | | |


