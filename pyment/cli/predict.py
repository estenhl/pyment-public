""" Module containing the script for procuring predictions. """
import argparse
import logging
import os
import nibabel as nib
import numpy as np
import pandas as pd

from tensorflow.keras import Model
from tqdm import tqdm

from pyment.models import get_model_class


LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

def _predict_from_path(model: Model, filename: str):
    image = nib.load(filename)
    image = image.get_fdata()
    image = np.expand_dims(image, 0)

    prediction = model.predict(image, verbose=0)[0]
    prediction = model.postprocess(prediction)

    return prediction

def predict(modelname: str, weights: str, inputs: str, pattern: str = None,
            destination: str = None, size: int = None):
    """ Creates predictions for an input nifti image (or set of images)
    from a given model with a given set of weights.

    Parameters:
    -----------
    modelname : str
        A string identifying the model architecture to use. See the
        table in data/architectures.csv for available architectures.
    weights : str
        A string identifying the weights to use. See data/models.csv
        for a set of available weights.
    inputs : str
        Either the path to a nifti image, or to a folder containing
        multiple images. If the latter, the images doesn't have to
        reside directly in the folder specified. Instead, for each
        subfolder in the folder the <pattern> parameter determines the
        relative path to the image.
    pattern : str
        If <inputs> is a folder, this parameter determines the relative
        path from each subfolder to the nifti images.
    destination : str
        (Optional) path where a CSV-file containing the predictions are
        written.
    size : int
        (Optional) number of images to run. Only relevant if <inputs> is
        a folder containing multiple subfolders.

    Returns:
    --------
    np.ndarray
        Processed predictions.
    """
    model_class = get_model_class(modelname)
    model = model_class(weights=weights)

    if os.path.isfile(inputs):
        logging.debug('Predicting for single file')
        prediction = _predict_from_path(model, inputs)
        logging.info('Prediction for %s: %.2f', inputs, prediction)
    elif os.path.isdir(inputs):
        assert pattern is not None, \
            ('When a folder of images is provided as input, a pattern must '
             'also be given. It is assumed the given folder contains '
             'subfolders, and the pattern describes the path from each '
             'subfolder to the image files')
        subfolders = os.listdir(inputs)

        if size is not None:
            subfolders = subfolders[:size]

        predictions = pd.DataFrame({}, columns=['id', 'prediction'])

        for subfolder in tqdm(subfolders):
            path = os.path.join(inputs, subfolder, pattern)

            if not os.path.isfile(path):
                logger.warning('File %s does not exist. Skipping prediction '
                               'for %s', path, subfolder)
                continue

            prediction = _predict_from_path(model, path)
            predictions = pd.concat([
                predictions,
                pd.DataFrame({'id': [subfolder], 'prediction': [prediction]})
            ], ignore_index=True)

            if destination is not None:
                predictions.to_csv(destination, index=False)

        logging.info('Predictions:\n%s', str(predictions))


def main():
    parser = argparse.ArgumentParser('Generates a prediction for a given '
                                     'nifti image (or set of images) from a '
                                     'given model with a given set of weights')

    parser.add_argument('-m', '--model', required=True,
                        help=('A string identifying the model architecture to '
                              'use. See the table in data/architectures.csv '
                              'for available architectures'))
    parser.add_argument('-w', '--weights', required=True,
                        help=('A string identifying the weights to use. See '
                              'data/models.csv for a set of available '
                              'weights'))
    parser.add_argument('-i', '--inputs', required=True,
                        help=('Either the path to a nifti image, or to a '
                              'folder containing multiple images. If the '
                              'latter, the images doesn\'t have to reside '
                              'directly in the folder specified. Instead, for '
                              'each subfolder in the folder the <pattern> '
                              'parameter determines the relative path to the '
                              'image.'))
    parser.add_argument('-p', '--pattern', required=False, default=None,
                        help=('If <inputs> is a folder, this parameter '
                              'determines the relative path from each '
                              'subfolder to the nifti images.'))
    parser.add_argument('-d', '--destination', required=False, default=None,
                        help=('(Optional) path where a CSV-file containing '
                              'the predictions are written.'))
    parser.add_argument('-n', '--size', required=False, default=None, type=int,
                        help=('(Optional) number of images to run. Only '
                              'relevant if <inputs> is a folder containing '
                              'multiple subfolders.'))

    args = parser.parse_args()

    predict(modelname=args.model,
            weights=args.weights,
            inputs=args.inputs,
            pattern=args.pattern,
            destination=args.destination,
            size=args.size)


if __name__ == '__main__':
    main()
