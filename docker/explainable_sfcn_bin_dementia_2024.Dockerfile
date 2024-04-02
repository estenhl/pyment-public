FROM tensorflow/tensorflow:2.16.1

RUN mkdir /root/.pyment && mkdir /root/.pyment/models

WORKDIR /code

COPY pyment /code/pyment
COPY scripts/explain.py /code/explain.py
COPY weights.h5 /root/.pyment/models/binary_sfcn_dementia_2024_fold_0_weights.h5
COPY strategy.json /code/strategy.json
COPY requirements.txt /code/

RUN apt-get update && apt-get install -y git

RUN pip install -r /code/requirements.txt && \
    pip install git+https://github.com/estenhl/keras-explainability.git

#ENTRYPOINT ["python"]
#CMD ["explain.py", \
#     "-m", "sfcn-bin", \
#     "-w", "dementia-2024", \
#     "-i", "/input", \
#     "-s", "/code/strategy.json", \
#     "-p", "mri/cropped.nii.gz", \
#     "-d", "/output/predictions.csv", \
#     "-n", "5"]
