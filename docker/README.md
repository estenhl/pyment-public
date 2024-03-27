A collection of Dockerfiles used for building containers containing predictive models for neuroimaging data.

### Containers
| name | model | weights | includes preprocessing |
| --- | --- | --- | --- |
| estenhl/sfcn_reg_brain_age_2022 | SFCN-reg | brain-age-2022 | No |

### How to use
````
docker pull estenhl/sfcn_reg_brain_age_2022
docker run --rm -it \
    -v </path/to/data>:/input \
    -v </path/where/outputs/are/stored>:/output \
    sfcn_reg_brain_age_2022
````

