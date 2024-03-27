import argparse
import logging
import os
import nibabel as nib
import numpy as np
import pandas as pd

from tensorflow.keras import Model
from tqdm import tqdm

from pyment.models import get_model_class


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_from_path(model: Model, filename: str):
    image = nib.load(filename)
    image = image.get_fdata()
    image = np.expand_dims(image, 0)

    prediction = model.predict(image, verbose=0)[0]
    prediction = model.postprocess(prediction)

    return prediction

def predict(modelname: str, weights: str, input: str, pattern: str = None,
            destination: str = None, size: int = None):
    model_class = get_model_class(modelname)
    model = model_class(weights=weights)

    if os.path.isfile(input):
        logging.debug(f'Predicting for single file')
        prediction = predict_from_path(model, input)
        logging.info(f'Prediction for {input}: {prediction}')
    elif os.path.isdir(input):
        assert pattern is not None, \
            ('When a folder of images is provided as input, a pattern must '
             'also be given. It is assumed the given folder contains '
             'subfolders, and the pattern describes the path from each '
             'subfolder to the image files')
        subfolders = os.listdir(input)

        if size is not None:
            subfolders = subfolders[:size]

        paths = [os.path.join(input, subfolder, pattern) \
                 for subfolder in subfolders]
        predictions = [predict_from_path(model, path) for path in tqdm(paths)]
        predictions = pd.DataFrame({'id': subfolders,
                                    'prediction': predictions})
        logging.info(f'Predictions:\n{predictions}')

        if destination is not None:
            predictions.to_csv(destination, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generates a prediction for a given '
                                     'nifti image (or set of images) from a '
                                     'given model with a given set of weights')

    parser.add_argument('-m', '--model', required=True,
                        help='Name of the model to use')
    parser.add_argument('-w', '--weights', required=True,
                        help='Name of the weights to use')
    parser.add_argument('-i', '--input', required=True,
                        help=('Path to input data. Must be either a single '
                              '(nifti) image, or a folder containing several'))
    parser.add_argument('-p', '--pattern', required=False, default=None,
                        help=('Pattern used when the input is a folder of '
                              'subfolders. The pattern describes the path '
                              'to the image files relative to a subfolder'))
    parser.add_argument('-d', '--destination', required=False, default=None,
                        help='(Optional) path where predictions are written')
    parser.add_argument('-n', '--size', required=False, default=None, type=int,
                        help=('Number of images to predict for (used in '
                              'combination with an input folder)'))

    args = parser.parse_args()

    predict(modelname=args.model,
            weights=args.weights,
            input=args.input,
            pattern=args.pattern,
            destination=args.destination,
            size=args.size)
