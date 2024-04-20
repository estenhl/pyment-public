FROM tensorflow/tensorflow:2.16.1

RUN mkdir /root/.pyment && mkdir /root/.pyment/models

WORKDIR /code

COPY pyment /code/pyment
COPY data /code/data
COPY scripts/predict.py /code/predict.py
COPY weights.h5 /root/.pyment/models/regression_sfcn_brain_age_weights.h5
COPY requirements.txt /code/

RUN pip install --no-cache-dir -r /code/requirements.txt

ENTRYPOINT ["python", "predict.py", \
            "-m", "sfcn-reg", \
            "-w", "brain-age-2022", \
            "-i", "/input", \
            "-p", "mri/cropped.nii.gz", \
            "-d", "/output/predictions.csv"]
