import os
import pandas as pd

from pyment.models import get_model_class
from pyment.models.weight_repository import WeightRepository
from shutil import rmtree

from conftest import METADATA_FOLDER, TemporaryFolder


def test_weight_repository():
    """ Tests the WeightRepository. Downloads weights for all known
    models and instantiates model objects with them. """

    with TemporaryFolder() as folder:
        df = pd.read_csv(os.path.join(METADATA_FOLDER, 'models.csv'))

        for _, row in df.iterrows():
            name = row['name']
            architecture = row['architecture']
            filename = row['filename']

            path = WeightRepository.get_weights(architecture, name,
                                                folder=folder)
            cls = get_model_class(architecture)
            model = cls()

            assert os.path.isfile(os.path.join(folder, filename)), \
                ('WeightRepository does not download weights and store them '
                 'in the prespecified destination')

            try:
                model.load_weights(path)
            except Exception:
                assert False, \
                    f'Unable to load weights {name} for model {architecture}'

def test_weight_repository_invalid_architecture():
    """ Tests that the WeightRepository raises an error if queried for
    weights for a non-existing architecture. """

    try:
        WeightRepository.get_weights('invalid', 'brain-age-2022')
        assert False, \
            ('WeightRepository does not raise an error when called with an '
             'invalid model architecture')
    except Exception:
        pass

def test_weight_repository_invalid_name():
    """ Tests that the WeightRepository raises an error if queried for
    weights for a non-existing model. """

    try:
        WeightRepository.get_weights('sfcn-reg', 'invalid')
        assert False, \
            ('WeightRepository does not raise an error when called with an '
             'invalid model architecture')
    except Exception:
        pass

def test_weight_repository_creates_folder_if_necessary():
    try:
        path = WeightRepository.get_weights('RegressionSFCN', 'brain-age-2022',
                                            folder='temp')
        assert os.path.isdir('temp'), \
            'WeightRepository does not create folder if it does not exist'
    finally:
        rmtree('temp')
