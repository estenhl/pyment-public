FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repos/pyment

COPY pyment /repos/pyment/pyment
COPY tests /repos/pyment/tests
COPY pyproject.toml /repos/pyment/
COPY README.md /repos/pyment/
COPY LICENSE.md /repos/pyment/

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "pytest>=9.0.3,<10.0.0" && \
    pip install --no-cache-dir .

CMD ["pytest", "-q"]
