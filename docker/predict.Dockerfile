FROM estenhl/pyment-preprocessing:1.0.0

RUN python -m venv /envs/pyment

RUN mkdir /repos/pyment

COPY . /repos/pyment

RUN cd /repos/pyment && \
    /envs/pyment/bin/pip install --upgrade pip && \
    /envs/pyment/bin/pip install .
