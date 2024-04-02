import argparse
import json
import logging
import os
import nibabel as nib
import numpy as np
import pandas as pd

from explainability import LRP, LRPStrategy
from tensorflow.keras import Model
from tqdm import tqdm

from pyment.models import get_model_class


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def explain_from_path(model: Model, explainer: Model, filename: str):
    nifti_image = nib.load(filename)
    image = nifti_image.get_fdata()
    image = np.expand_dims(image, 0)

    prediction = model.predict(image, verbose=0)[0]
    prediction = model.postprocess(prediction)

    explanation = explainer.predict(image)[0,...,0]

    return nifti_image, prediction, explanation

def predict(modelname: str, weights: str, input: str, strategy: str,
            pattern: str, destination: str, size: int):
    model_class = get_model_class(modelname)
    model = model_class(weights=weights)

    with open(strategy, 'rb') as f:
        strategy = json.load(f)

    strategy = LRPStrategy(**strategy)
    explainer = LRP(model, layer=-1, idx=0, strategy=strategy)

    if destination is not None:
        if not os.path.isdir(destination):
            raise ValueError(f'Folder {destination} does not exist')
        if not os.path.isdir(os.path.join(destination, 'explanations')):
            os.mkdir(os.path.join(destination, 'explanations'))

    if os.path.isfile(input):
        logging.debug(f'Predicting for single file')
        nifti, prediction, explanation = explain_from_path(model, explainer, input)
        logging.info(f'Prediction for {input}: {prediction}')
    elif os.path.isdir(input):
        assert pattern is not None, \
            ('When a folder of images is provided as input, a pattern must '
             'also be given. It is assumed the given folder contains '
             'subfolders, and the pattern describes the path from each '
             'subfolder to the image files')
        subfolders = os.listdir(input)
        subfolders = subfolders[:size] if size is not None else subfolders

        paths = [os.path.join(input, subfolder, pattern) \
                 for subfolder in subfolders]

        predictions = []

        for i, path in tqdm(enumerate(paths), total=len(paths)):
            nifti, prediction, explanation = explain_from_path(model,
                                                               explainer,
                                                               path)
            predictions.append({
                'id': subfolders[i],
                'prediction': prediction
            })

            if destination is not None:
                explanation = nib.Nifti1Image(explanation,
                                              affine=nifti.affine,
                                              header=nifti.header)
                nib.save(explanation, os.path.join(destination,
                                                   'explanations',
                                                   f'{subfolders[i]}.nii.gz'))

        predictions = pd.DataFrame(predictions)
        logging.info(f'Predictions:\n{predictions}')

        if destination is not None:
            predictions.to_csv(os.path.join(destination, 'predictions.csv'),
                               index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generates a prediction and an '
                                     'explanation for a given '
                                     'nifti image (or set of images) from a '
                                     'given model with a given set of  '
                                     'weights, and an LRP strategy')

    parser.add_argument('-m', '--model', required=True,
                        help='Name of the model to use')
    parser.add_argument('-w', '--weights', required=True,
                        help='Name of the weights to use')
    parser.add_argument('-i', '--input', required=True,
                        help=('Path to input data. Must be either a single '
                              '(nifti) image, or a folder containing several'))
    parser.add_argument('-s', '--strategy', required=True,
                        help='Path to JSON containing LRP strategy')

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
            strategy=args.strategy,
            pattern=args.pattern,
            destination=args.destination,
            size=args.size)
