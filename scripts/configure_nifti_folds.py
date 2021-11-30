import argparse
import csv
import logging
import math
import os
import numpy as np
import pandas as pd

from collections import Counter
from functools import reduce
from typing import List

from utils import configure_environment

configure_environment()

from pyment.data import NiftiDataset


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_nifti_folds(*, folders: List[str], targets: List[str], 
                          stratification: List[str] = None, k: int = 1, 
                          test_portion: float = .0, destination: str):
    if os.path.isdir(destination):
        raise ValueError(f'Folder {destination} already exists')

    if targets is None:
        targets = []
    if stratification is None:
        stratification = []
    
    datasets = [NiftiDataset.from_folder(folder, target=targets) \
                for folder in folders]
    dataset = reduce(lambda x, y: x + y, datasets)

    logger.info(f'Instantiated dataset with {len(dataset)} datapoints')

    if test_portion > 0.0:
        assert test_portion <= 0.5, \
            'Unable to configure folds with test portion > 0.5'

        test_splits = math.ceil(1 / test_portion)
        folds = dataset.stratified_folds(test_splits, stratification)
        test = folds[-1]
        dataset = reduce(lambda x, y: x + y, folds[:-1])

        logger.info(f'Reserved {len(test)} images for test')
        logger.info(f'Using {len(dataset)} images for folds')

    folds = dataset.stratified_folds(k, stratification)

    os.mkdir(destination)

    for i in range(len(folds)):
        folds[i].save(os.path.join(destination, f'fold_{i}.json'))

    names = [f'fold {i}' for i in range(k)]
    datasets = folds

    if test_portion > 0.0:
        test.save(os.path.join(destination, 'test.json'))
        names.append('test')
        datasets.append(test)

    df = pd.DataFrame({}, index=names)
    df.index.name = 'split'

    for variable in datasets[0].variables:
        if datasets[0].labels[variable].dtype == object:
            df[variable] = [Counter(dataset.labels[variable]) \
                            for dataset in datasets]
        else:
            df[f'{variable} mean'] = [np.nanmean(dataset.labels[variable]) \
                                    for dataset in datasets]
            df[f'{variable} max'] = [np.nanmax(dataset.labels[variable]) \
                                    for dataset in datasets]
            df[f'{variable} min'] = [np.nanmin(dataset.labels[variable]) \
                                    for dataset in datasets]

    pd.set_option('display.max_columns', 999)
    logger.info(f'Datasets:\n{str(df)}\n')
    df.to_csv(os.path.join(destination, 'overview.csv'),
              quoting=csv.QUOTE_NONNUMERIC)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Configures and saves K folds of a '
                                      'nifti dataset which is instantiated '
                                      'from multiple source folders'))

    parser.add_argument('-f', '--folders', required=True, nargs='+',
                        help=('Source folders containing nifti data. Each '
                              'folder should have a subfolder \'images\' and '
                              'a csv-file \'labels.csv\' with a column \'id\' '
                              'corresponding to the image filenames and a '
                              'column for each target'))
    parser.add_argument('-t', '--targets', required=False, default=None,
                        nargs='+', help='Targets for the dataset')
    parser.add_argument('-s', '--stratification', required=False, default=None,
                        nargs='+', help='Variables used for stratification')
    parser.add_argument('-k', '--folds', required=False, type=int, default=1,
                        help='Number of folds to generate')
    parser.add_argument('-e', '--test_portion', required=False, type=float, default=.0,
                        help=('Portion of data which is reserved for an '
                              'explicit test set'))
    parser.add_argument('-d', '--destination', required=True,
                        help=('Path to folder where resulting datasets are '
                              'stored'))

    args = parser.parse_args()

    configure_nifti_folds(folders=args.folders, targets=args.targets, 
                          stratification=args.stratification, k=args.folds, 
                          test_portion=args.test_portion,
                          destination=args.destination)