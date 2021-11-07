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