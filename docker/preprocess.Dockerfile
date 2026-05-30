FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    apt-utils git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /envs && python -m venv /envs/fastsurfer

RUN mkdir /repos && \
    git clone https://github.com/Deep-MI/FastSurfer.git /repos/FastSurfer \
    && cd /repos/FastSurfer \
    && git checkout v2.0.1

ENV FASTSURFER_HOME=/repos/FastSurfer

# Install requirements. The SimpleITK version in requirements.txt is yanked,
# so we need to install it manually and remove it from requirements.txt.
RUN /envs/fastsurfer/bin/pip install --upgrade pip && \
    /envs/fastsurfer/bin/pip install simpleitk==2.1.1.2 --upgrade && \
    sed -i '/simpleitk/d' ${FASTSURFER_HOME}/requirements.txt && \
    /envs/fastsurfer/bin/pip install -r ${FASTSURFER_HOME}/requirements.txt

#COPY ${CHECKPOINTS_FOLDER} ${FASTSURFER_HOME}/FastSurferCNN/checkpoints
COPY checkpoints/fastsurfer ${FASTSURFER_HOME}/FastSurferCNN/checkpoints

RUN mkdir /scripts
COPY scripts/preprocess.sh /scripts/preprocess.sh

CMD ["/bin/sh", "/scripts/preprocess.sh", \
     "--license", "/licenses/freesurfer.txt", \
     "--python", "/envs/fastsurfer/bin/python", \
     "/input", "/output/fastsurfer"]
