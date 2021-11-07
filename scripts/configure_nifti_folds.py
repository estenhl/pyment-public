import argparse
import logging

from functools import reduce
from typing import List

from pyment.data import NiftiDataset


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_nifti_folds(*, folders: List[str], targets: List[str], 
                          stratification: List[str] = None,
                          folds: int = 1, test_portion: float = .0,
                          destination: str):
    if targets is None:
        targets = []
    if stratification is None:
        stratification = []
    
    datasets = [NiftiDataset.from_folder(folder) for folder in folders]
    dataset = reduce(lambda x, y: x + y, datasets)
    dataset = dataset.stratified(stratification)

    logger.info(f'Instantiated dataset with {len(dataset)} datapoints')

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
                          stratification=args.stratification, folds=args.folds, 
                          test_portion=args.test_portion,
                          destination=args.destination)