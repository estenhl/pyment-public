# Use Ubuntu as the base image, which is compatible with FreeSurfer's Linux requirements
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for FreeSurfer
RUN apt-get update && apt-get install -y \
    tcsh \
    xfonts-base \
    libglu1-mesa \
    libfreetype6 \
    libxmu6 \
    libxpm4 \
    libxi6 \
    wget \
    perl \
#    && rm -rf /var/lib/apt/lists/*

# Set up environment variables for FreeSurfer
ENV FREESURFER_HOME=/usr/local/freesurfer
ENV SUBJECTS_DIR=$FREESURFER_HOME/subjects
ENV FUNCTIONALS_DIR=$FREESURFER_HOME/sessions

# Download and extract FreeSurfer
# Note: Replace the URL below with the actual URL to download FreeSurfer 5.3.0,
# and ensure you have permission to use it.
# This example assumes you have the file available locally or have a direct URL for download.
COPY freesurfer-Linux-centos6_x86_64-stable-pub-v5.3.0.tar.gz /tmp/freesurfer-Linux-centos6_x86_64-stable-pub-v5.3.0.tar.gz

RUN tar -xzvf /tmp/freesurfer-Linux-centos6_x86_64-stable-pub-v5.3.0.tar.gz -C /usr/local \
    && rm /tmp/freesurfer-Linux-centos6_x86_64-stable-pub-v5.3.0.tar.gz

COPY image.nii.gz /image.nii.gz

#RUN apt-get install -y libperl-dev

ENTRYPOINT bash
