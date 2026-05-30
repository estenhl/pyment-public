# Changelog
All notable changes to this project _after release v3.0.0_ will be documented in this file.

## v5.0.0
Updated runtime to Python 3.13 and TensorFlow 2.21. Ported model code to Keras 3 (removed legacy optimizer namespace, fixed layer naming, replaced `tf.concat` with `Concatenate` layer). Finetuned models now saved in native Keras format (`.keras`). Added `pyment-verify-loader` CLI endpoint. Docker images updated to use Python 3.13 with FastSurfer isolated to Python 3.10.

## v4.1.2
Replaced the external tensorflow-neuroimaging library with internal functionality

## v4.1.0
Added more functionality for automatic testing and sanity checking

## v4.0.0
Following release v4.0.0 this repository will host the pretrained multi-task model from [Learning diverse and generic representations of the brain with large-scale multi-task pretraining](https://www.medrxiv.org/content/10.64898/2025.12.19.25342659v1). Preliminary results indicate that this model outperforms previous models in this repo both in terms of accurately predicting brain age and for transfer learning. Older models can be found by traversing the version history

## v3.0.0
Following (pending) release v3.0.0 (and onwards) this repository will serve as a model zoo for pretrained neuroimaging models from various publications. This entails that the utilities that were previously packaged alongside the models (e.g. for training models) has been stripped, to avoid bloating the repository. If you are interested in specific code for a specific paper either check out the previous releases or email me at [estenhl@uio.no](mailto:estenhl@uio.no)
