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

# Estimating brain age with Docker (with preprocessed images)
If you want to use docker containers for predictions there are two options:
      1. Configuring and building your own docker container
      2. Using one of our prebuilt containers
## 1a. Build your own docker container
We recommend only building your own docker container if you want to configure it yourself, e.g. by using a different model or your own trained weights. If so, there is a set of dockerfiles in the [docker](https://github.com/estenhl/pyment-public/tree/main/docker)-folder that can be used as starting points. Building a container for estimating brain age using SFCN-reg with our pretrained weights can e.g. be done via
```
docker build \
      --tag estenhl/sfcn-reg-predict-brain-age \
      --file docker/Dockerfile.predict .
```
## 1b. Download our prebuilt docker containers
We have built a set of docker containers containing different models, weights and preprocessing schemes in our [dockerhub account](https://hub.docker.com/search?q=estenhl&type=image). Downloading one of these can be done via
```
docker pull estenhl/sfcn-reg-predict-brain-age
```

## 2. Estimate brain age using the container
When you have a container with the model, it needs to be run to get the brain age estimates. If you are using one of our containers it needs access to two volumes (e.g. folders on your local computer) which is passed to the ```docker run``` command via the [```--mount```](https://docs.docker.com/storage/bind-mounts/)-argument. One folder should contain the images and the labels as explained in the [Preparing data](#preparing-data)-section. The second should be a folder where the predictions and the logs are written.<br />
<b>NOTE: If using a container which does not preprocess data (e.g. does not explicitly have ```preprocess``` in its name) the data needs to be preprocessed according to our [Preprocessing pipeline](#preprocessing) beforehand</b></br>
Running our prebuilt ```sfcn-reg-predict-brain-age```-container can be done via e.g.
```
docker run \
      --rm \
      --name predict-brain-age \
      --mount type=bind,source=<path-to-folder-with-images>,target=/images \
      --mount type=bind,source=<path-to-folder-with-predictions>,target=/predictions \
      estenhl/sfcn-reg-predict-brain-age
```

# Estimating brain age with Docker (including preprocessing)
## Building your own container
### 1. Build a freesurfer container
To build the freesurfer container you need a [freesurfer 5.3.0 tar.gz](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/5.3.0/freesurfer-Linux-centos6_x86_64-stable-pub-v5.3.0.tar.gz)-file available _within_ the root folder of this project
```
docker build \
      --tag estenhl/freesurfer:5.3 \
      --file docker/Dockerfile.freesurfer \
      --build-arg tarPath=freesurfer-Linux-centos6_x86_64-stable-pub-v5.3.0.tar.gz .
```
### 1b. Test the freesurfer container
Test the freesurfer-container by copying in an image and running ```recon-all```. Note that you need to copy in a valid free-surfer license for the container to run successfully
```
docker run \
      -it \
      --rm \
      --mount type=bind,source=/Users/esten/freesurfer-license.txt,target=/usr/local/freesurfer/license.txt \
      --mount type=bind,source=<path-to-example-image>,target=/tmp.nii.gz \
      estenhl/freesurfer:5.3
mkdir subjects
recon-all -sd subjects -s tmp -i /tmp.nii.gz -autorecon1
```
### 2. Build a FreeSurfer and FSL container
```
docker build \
      --tag estenhl/freesurfer_and_fsl:6.0 \
      --file docker/Dockerfile.freesurfer_and_fsl .
```

### 2b. Test the FreeSurfer and FSL container
Test the FSL-portion of the container by running flirt
```
flirt \
      -in tmp.nii.gz \
      -out flirted.nii.gz \
      -ref /usr/local/fsl/data/linearMNI/MNI152lin_T1_1mm_brain.nii.gz
```