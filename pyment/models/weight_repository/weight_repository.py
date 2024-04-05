from __future__ import annotations

import base64
import logging
import os
import requests
import pandas as pd

from ...utils import MODELS_DIR, METADATA_DIR


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightRepository:
    """ A class representing a repository for model weights. Allows for
    looking up and, optionally, downloading, weights for the pretrained
    models in the model zoo. """

    _BASE_URL = 'https://api.github.com/repos/estenhl/pyment-public/git/blobs'

    @staticmethod
    def _download_weight(sha: str, filename: str,
                         base_url: str = _BASE_URL) -> None:
        url = f'{base_url}/{sha}'
        logging.info(f'Downloading {url} to {filename}')

        resp = requests.get(url, stream=True)
        content = resp.json()['content']

        with open(filename, 'wb') as f:
            f.write(base64.b64decode(content))

    @staticmethod
    def get_weights(architecture: str, name: str,
                    folder: str = MODELS_DIR) -> str:
        """ Returns the path to a file containing weights to the
        given model. If necessary, the weights are downloaded.

        Parameters
        ----------
        architecture : str
            The name of the architecture to which the weights belong.
        name : str
            The name of the pretrained model.
        folder : str
            The folder where weights are stored. Defaults to the
            MODELS_DIR set in utils.py in the top-level folder of the
            package

        Returns
        -------
        string
            The path to the weights belonging to the given architecture
            and name

        Raises
        ------
        KeyError
            If no pretrained weight corresponding to the given
            architecture and name exists
        ValueError
            If multiple pretrained weights corresponding to the given
            architecture and name exists
        """
        table = pd.read_csv(os.path.join(METADATA_DIR, 'models.csv'))
        key = (architecture, name)
        rows = table.loc[(table['architecture'] == architecture) & \
                        (table['name'] == name)]

        if len(rows) == 0:
            error = f'Unknown architecture/name combination {key}'
            logging.error(error)
            raise KeyError(error)
        elif len(rows) > 1:
            error = f'Multiple entries for architecture/name combination {key}'
            logging.error(error)
            raise ValueError(error)

        row = rows.iloc[0]
        path = os.path.join(folder, row['filename'])

        if not os.path.isfile(path):
            sha = row['sha']
            WeightRepository._download_weight(sha, path)

        return path
