FROM python:3.10-slim AS py310
FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    apt-utils git patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY --from=py310 /usr/local/bin/python3.10 /usr/local/bin/python3.10
COPY --from=py310 /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=py310 /usr/local/lib/libpython3.10.so.1.0 /usr/local/lib/libpython3.10.so.1.0
RUN ldconfig

RUN mkdir -p /envs && python3.10 -m venv /envs/fastsurfer

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

# Clear executable stack flags on torch libs — required on glibc 2.35+
RUN find /envs/fastsurfer -name "*.so" | xargs patchelf --clear-execstack

COPY checkpoints/fastsurfer ${FASTSURFER_HOME}/FastSurferCNN/checkpoints

RUN mkdir /scripts
COPY scripts/preprocess.sh /scripts/preprocess.sh

CMD ["/bin/sh", "/scripts/preprocess.sh", \
     "--license", "/licenses/freesurfer.txt", \
     "--python", "/envs/fastsurfer/bin/python", \
     "/input", "/output/fastsurfer"]
