FROM tensorflow/tensorflow:2.16.1

RUN mkdir /root/.pyment && mkdir /root/.pyment/models

WORKDIR /code

COPY data /code/data
COPY pyment /code/pyment
COPY scripts/predict.py /code/predict.py
COPY weights.h5 /root/.pyment/models/binary_sfcn_dementia_2024_fold_0_weights.h5
COPY requirements.txt /code/

RUN pip install -r /code/requirements.txt

ENTRYPOINT ["python", "predict.py", \
            "-m", "sfcn-bin", \
            "-w", "dementia-2024", \
            "-i", "/input", \
            "-p", "mri/cropped.nii.gz", \
            "-d", "/output/predictions.csv"]
