import argparse
import pandas as pd

from utils import configure_environment

configure_environment()

from pyment.data import AsyncNiftiGenerator, NiftiDataset
from pyment.models import get as get_model, ModelType


def predict_brain_age(*, folder: str, model: str, weights: str = None,
                      batch_size: int, threads: int = None, 
                      normalize: bool = False, destination: str):
    dataset = NiftiDataset.from_folder(folder, target='age')

    preprocessor = lambda x: x/255. if normalize else x

    if threads is None or threads == 1:
        raise NotImplementedError(('Predicting from synchronous generator '
                                   'is not implemented'))

    generator = AsyncNiftiGenerator(dataset, preprocessor=preprocessor,
                                    batch_size=batch_size, threads=threads)

    model = get_model(model, weights=weights)

    ids = dataset.ids
    labels = dataset.y
    predictions = model.predict(generator)

    if model.type == ModelType.REGRESSION:
        predictions = predictions.squeeze()

    df = pd.DataFrame({'age': labels, 'prediction': predictions}, index=ids)
    df.to_csv(destination)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Estimates brain age for images from a '
                                      'given folder using a given model with '
                                      'given weights'))

    parser.add_argument('-f', '--folder', required=True,
                        help=('Folder containing images. Should have a '
                              'csv-file called \'labels.csv\' with columns '
                              'id and age, and a subfolder \'images\' '
                              'containing nifti files'))
    parser.add_argument('-m', '--model', required=True,
                        help='Name of the model to use (e.g. sfcn-reg)')
    parser.add_argument('-w', '--weights', required=False, default=None,
                        help='Weights to load in the model')
    parser.add_argument('-b', '--batch_size', required=True, type=int,
                        help='Batch size to use while predicting')
    parser.add_argument('-t', '--threads', required=False, default=None, 
                        type=int, help=('Number of threads to use for reading '
                                        'data. If not set, a synchronous '
                                        'generator will be used'))
    parser.add_argument('-n', '--normalize', action='store_true',
                        help=('If set, images will be normalized to range '
                              '(0, 1) before prediction'))
    parser.add_argument('-d', '--destination', required=True,
                        help=('Path where CSV containing ids, labels '
                              'and predictions are stored'))

    args = parser.parse_args()

    predict_brain_age(folder=args.folder, 
                      model=args.model, 
                      weights=args.weights, batch_size=args.batch_size,
                      threads=args.threads, normalize=args.normalize,
                      destination=args.destination)