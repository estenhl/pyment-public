FROM estenhl/pyment-preprocessing:1.0.0

RUN apt-get update && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev git \
    && rm -rf /var/lib/apt/lists/*

ENV PYENV_ROOT=/root/.pyenv
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN curl https://pyenv.run | bash && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc

RUN eval "$(pyenv init -)" && \
    pyenv install 3.10.4

RUN mkdir -p /envs && \
    $PYENV_ROOT/versions/3.10.4/bin/python -m venv /envs/pyment

RUN mkdir -p /repos/pyment

COPY scripts /repos/pyment/scripts
COPY pyment /repos/pyment/pyment
COPY pyproject.toml /repos/pyment/
COPY README.md /repos/pyment/
COPY LICENSE.md /repos/pyment/

RUN /envs/pyment/bin/pip install --upgrade pip poetry-core build && \
    cd /repos/pyment && \
    /envs/pyment/bin/pip install --no-cache-dir .

CMD ["/bin/sh", "-c", \
    "/scripts/preprocess.sh \
        --license /licenses/freesurfer.txt \
        --python /envs/fastsurfer/bin/python \
        /inputs \
        /outputs/fastsurfer \
    && /envs/pyment/bin/python /repos/pyment/scripts/predict_from_fastsurfer_folder.py \
        /outputs/fastsurfer \
        -d /outputs/predictions.csv"]
