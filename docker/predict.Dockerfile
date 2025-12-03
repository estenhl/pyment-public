FROM python:3.10.4-slim

RUN mkdir -p /repos/pyment

COPY scripts /repos/pyment/scripts
COPY pyment /repos/pyment/pyment
COPY pyproject.toml /repos/pyment/
COPY README.md /repos/pyment/
COPY LICENSE.md /repos/pyment/

RUN pip install --upgrade pip poetry-core build && \
    cd /repos/pyment && \
    pip install --no-cache-dir .

RUN mkdir -p /.pyment/weights && \
    chmod -R 1777 /.pyment
COPY checkpoints/pyment /.pyment/weights

CMD ["python", "/repos/pyment/scripts/predict_from_fastsurfer_folder.py", \
     "/fastsurfer", \
     "-d", "/output/predictions.csv"]
