A collection of Dockerfiles used for building containers around the predictive models contained in the repository. All containers rely on neuroimaging data to run, and it should be structured as demonstrated in the [preprocessing notebook](../notebooks/Download%20and%20preprocess%20IXI%20T1%20data.ipynb).

## Images
| Name | Architecture | Weights | Includes preprocessing | Includes explainability |
| :-: | :-: | :-: | :-: | :-: |
| estenhl/sfcn-reg_brain-age:2022 | sfcn-reg | brain-age-2022 | No | No |
| estenhl/sfcn-bin_dementia:2024 | sfcn-bin | dementia-2024 | No | No |
| estenhl/sfcn-bin_dementia:2024-explainable | sfcn-bin | dementia-2024 | No | Yes |

## Usage
### estenhl/sfcn-reg_brain-age:2022
<b>NOTE: This image requires preprocessed MRIs</b>
```
docker pull estenhl/sfcn-reg_brain-age:2022
docker run --rm -it \
    -v </path/to/data>:/input \
    -v </path/where/outputs/are/stored>:/output \
    estenhl/sfcn-reg_brain-age:2022
```
### estenhl/sfcn-bin_dementia:2024
<b>NOTE: This image requires preprocessed MRIs</b>
```
docker pull estenhl/sfcn-bin_dementia:2024
docker run --rm -it \
    -v </path/to/data>:/input \
    -v </path/where/outputs/are/stored>:/output \
    estenhl/sfcn-bin_dementia:2024
```
### estenhl/sfcn-bin_dementia:2024-explainable
<b>NOTE: This image requires preprocessed MRIs</b>
```
docker pull estenhl/sfcn-bin_dementia:2024-explainable
docker run --rm -it \
    -v </path/to/data>:/input \
    -v </path/where/outputs/are/stored>:/output \
    estenhl/sfcn-bin_dementia:2024-explainable
```