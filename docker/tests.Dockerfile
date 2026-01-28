FROM python:3.10.2-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repos/pyment

COPY pyment /repos/pyment/pyment
COPY tests /repos/pyment/tests
COPY data /repos/pyment/data
COPY pyproject.toml /repos/pyment/
COPY README.md /repos/pyment/
COPY LICENSE.md /repos/pyment/

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      numpy==1.23.5 \
      nibabel==5.2.1 \
      scipy==1.11.4 \
      pytest==8.3.3 && \
    pip install --no-cache-dir --no-deps .

CMD ["pytest", "-q"]
