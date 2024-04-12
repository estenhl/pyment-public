A collection of Dockerfiles used for building containers with predictive models for neuroimaging data.

### Images
| Name | Model | Weights | Includes preprocessing |
| --- | --- | --- | --- |
| estenhl/sfcn-reg_brain-age:2022 | SFCN-reg | brain-age-2022 | No |

### How to use
#### estenhl/sfcn-reg_brain-age:2022
NOTE: This container requires the data to be preprocessed
````
docker pull estenhl/sfcn-reg_brain-age:2022
docker run --rm -it \
    -v </path/to/data>:/input \
    -v </path/where/outputs/are/stored>:/output \
    estenhl/sfcn-reg_brain-age:2022
````

