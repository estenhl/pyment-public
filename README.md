Repository containing code, models and tutorials for the paper [Deep neural networks learn general and clinically relevant representations of the ageing brain](https://www.medrxiv.org/content/10.1101/2021.10.29.21265645v1)

# Installation (via terminal and Anaconda)

1. Clone the github repo<br />
```git clone git@github.com:estenhl/pyment-public.git```
2. Enter the folder<br />
```cd pyment-public```
3. Create a conda environment<br />
```conda create --name pyment python=3.9```
4. Activate environment<br />
```conda activate pyment```
5. Install required packages<br />
```pip install -r requirements.txt```
6. Install Tensorflow<br />
a. Tensorflow for GPU<br />
```pip install tensorflow-gpu```<br />
b. Tensorflow for CPU<br />
```pip install tensorflow```
6. Source the package<br />
```conda develop .```

# Preparing data
While the models adhere to the Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) interface and can thus be used however one wants, we have provided [Dataset](https://github.com/estenhl/pyment-public/blob/main/pyment/data/datasets/nifti_dataset.py)/[Generator](https://github.com/estenhl/pyment-public/blob/main/pyment/data/generators/async_nifti_generator.py)-classes for nifti-files which are used in the tutorials. For these classes to work off-the-shelf the Nifti-data has to be organized in the following folder structure:
```
.
├── labels.csv
└── images
      ├── image1.nii.gz
      ├── image2.nii.gz
     ...
      └── imageN.nii.gz
```
where ```labels.csv``` is a csv-file with column ```id``` (corresponding to image1, image2, etc) and column ```age```.

## Preprocessing
Before training the models all images were ran through the following preprocessing pipeline:

1. Extract brainmask with ```recon-all -autorecon1``` (FreeSurfer)
2. Transform to *.nii.gz with ```mri_convert``` (FreeSurfer)
3. Translate to FSL space with ```fslreorient2std``` (FSL)
4. Register to MNI space with ```flirt -dof 6``` (FSL, linear registration), and the standard FSL template ```MNI152_T1_1mm_brain.nii.gz```
5. Crop away borders of ```[6:173,2:214,0:160]```

A full example which downloads the IXI dataset and preprocesses it can be found in the [Preprocessing tutorial](https://github.com/estenhl/pyment-public/blob/main/notebooks/Download%20and%20preprocess%20IXI.ipynb)

# Estimating brain age in Python
Estimating brain age using the trained brain age model from the paper consists of downloading the weights, instantiating the model with said weights, and calling [Model.fit()](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict) with an appropriate generator. A full tutorial (which relies on having a prepared dataset) can be found in the [Python prediction tutorial](https://github.com/estenhl/pyment-public/blob/main/notebooks/Encode%20dataset%20as%20feature%20vectors.ipynb)

Instructions for downloading, building and using our docker containers for brain age predictions can be found in the [docker](https://github.com/estenhl/pyment-public/tree/main/docker)-folder

# License
The code and models in this repo is released under the CC-BY-NC license.

# Citation
If you use code or models from this repo, please cite
```
@article{leonardsen_deep_2022,
	title = {Deep neural networks learn general and clinically relevant representations of the ageing brain},
	volume = {256},
	rights = {All rights reserved},
	issn = {1053-8119},
	url = {https://www.sciencedirect.com/science/article/pii/S1053811922003342},
	doi = {10.1016/j.neuroimage.2022.119210},
	pages = {119210},
	journaltitle = {{NeuroImage}},
	shortjournal = {{NeuroImage}},
	author = {Leonardsen, Esten H. and Peng, Han and Kaufmann, Tobias and Agartz, Ingrid and Andreassen, Ole A. and Celius, Elisabeth Gulowsen and Espeseth, Thomas and Harbo, Hanne F. and Høgestøl, Einar A. and Lange, Ann-Marie de and Marquand, Andre F. and Vidal-Piñeiro, Didac and Roe, James M. and Selbæk, Geir and Sørensen, Øystein and Smith, Stephen M. and Westlye, Lars T. and Wolfers, Thomas and Wang, Yunpeng},
	date = {2022-08-01},
}
```
