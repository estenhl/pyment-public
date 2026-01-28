FROM python:3.10.2-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

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
COPY checkpoints/pyment/multi-2025.* /.pyment/weights/

CMD ["pyment-predict", "/fastsurfer", "-d", "/output/predictions.csv"]
