Repository containing code, models and tutorials for the paper "Deep neural networks learn general and clinically relevant represen-
tations of the ageing brain"

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
Before training the models all images were ran through the following preprocessing pipeline:

1. Extract brainmask with ```recon-all -autorecon1``` (FreeSurfer)
2. Transform to *.nii.gz with ```mri_convert``` (FreeSurfer)
3. Translate to FSL space with ```fslreorient2std``` (FSL)
4. Register to MNI space with ```flirt -dof 6``` (FSL, linear registration), and the standard FSL template ```MNI152_T1_1mm_brain.nii.gz```
5. Crop away borders of ```[6:173,2:214,0:160]```
A full example which downloads the IXI dataset and preprocesses it can be found in the [Preprocessing tutorial](https://github.com/estenhl/pyment-public/blob/main/notebooks/Download%20and%20preprocess%20IXI.ipynb)

# Estimating brain age in Python
Estimating brain age using the trained brain age model from the paper consists of downloading the weights, instantiating the model with said weights, and calling [Model.fit()](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict) with an appropriate generator. A full tutorial (which relies on having a prepared dataset) can be found in the [Python prediction tutorial](https://github.com/estenhl/pyment-public/blob/main/notebooks/Encode%20dataset%20as%20feature%20vectors.ipynb)
