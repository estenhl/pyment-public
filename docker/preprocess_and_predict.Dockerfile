FROM estenhl/pyment-preprocess

RUN mkdir -p /repos/pyment

COPY scripts /repos/pyment/scripts
COPY pyment /repos/pyment/pyment
COPY pyproject.toml /repos/pyment/
COPY README.md /repos/pyment/
COPY LICENSE.md /repos/pyment/

RUN pip install --upgrade pip poetry-core build && \
    cd /repos/pyment && \
    pip install --no-cache-dir ".[cuda]" && \
    find /usr/local/lib/python3.13/site-packages/nvidia \
        -mindepth 2 -maxdepth 2 -name lib -type d \
        > /etc/ld.so.conf.d/nvidia-pip.conf && \
    ldconfig

RUN mkdir -p /.pyment/weights && \
    chmod -R 1777 /.pyment
COPY checkpoints/pyment/* /.pyment/weights/

ENV WEIGHTS=multi-2025
ENV MODEL=sfcn-multi
ENV TARGETS="age sex handedness bmi fluid_intelligence neuroticism"

CMD ["/bin/sh", "-c", "\
  mkdir -p /output/fastsurfer && \
  /scripts/preprocess.sh \
    --license /licenses/freesurfer.txt \
    --python /envs/fastsurfer/bin/python \
    /input /output/fastsurfer && \
  pyment-predict \
    /output/fastsurfer \
    --destination /output/predictions.csv \
    --weights /.pyment/weights/${WEIGHTS}.h5 \
    --model ${MODEL} \
    --targets ${TARGETS} \
"]
