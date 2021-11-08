import argparse
import os

from typing import List

from utils import configure_environment

configure_environment()

from pyment.data import NiftiDataset


def configure_nifti_dataset(*, folder: str, target: List[str], 
                            destination: str):
    assert os.path.isdir(os.path.join(folder, 'images')), \
        'Folder must have a subfolder \'images\''
    assert os.path.isfile(os.path.join(folder, 'labels.csv')), \
        'Folder must contain a csv-file \'labels.csv\''

    dataset = NiftiDataset.from_folder(folder, target=target)

    dataset.save(destination)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Configures and saves a NiftiDataset '
                                      'from a given folder according to the '
                                      'given keyword arguments'))

    parser.add_argument('-f', '--folder', required=True, 
                        help=('Folder containing data. Must include a '
                              'csv-file \'labels.csv\' and an \'images\' '
                              'folder with corresponding images. The '
                              'csv-file should have a column \'id\' where '
                              'the ids match the filenames (without '
                              'suffix) of the images'))
    parser.add_argument('-t', '--target', required=False, nargs='+', 
                        default=None, help=('Variable used as target for the '
                                            'dataset. Must correspond to one '
                                            'or more of the columns of '
                                            '\'labels.csv\''))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where dataset json file is stored')

    args = parser.parse_args()

    configure_nifti_dataset(folder=args.folder, target=args.target,
                            destination=args.destination)