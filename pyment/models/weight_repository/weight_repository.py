import base64
import os
import requests

from ...utils import MODELS_DIR

_weights = {
    ('RegressionSFCN', False, 'brain-age-2022'): {
        'filename': 'regression_sfcn_brain_age_weights_no_top.h5',
        'sha': '54b7f9545f1120cb302ff7342aaa724513f75219'
    },
    ('RegressionSFCN', True, 'brain-age-2022'): {
        'filename': 'regression_sfcn_brain_age_weights.h5',
        'sha': 'f87a66558433308bb8a5ecfb6aaa784811c5cd45'
    },
    ('RankingSFCN', False, 'brain-age-2022'): {
        'filename': 'ranking_sfcn_brain_age_weights_no_top.h5',
        'sha': '2b236896bd343e71e39569d8cc6bec77c455380f'
    },
    ('RankingSFCN', True, 'brain-age-2022'): {
        'filename': 'ranking_sfcn_brain_age_weights.h5',
        'sha': '5d1bc5fc66327eb905acf81d9956f0391277b078'
    },
    ('SoftClassificationSFCN', False, 'brain-age-2022'): {
        'filename': 'soft_classification_sfcn_brain_age_weights_no_top.h5',
        'sha': '6001748a3a9291c8544d5be8235c0ba1a8c41dbc'
    },
    ('SoftClassificationSFCN', True, 'brain-age-2022'): {
        'filename': 'soft_classification_sfcn_brain_age_weights.h5',
        'sha': '7b4f7bf4c989b80877b0bc0efe8b5125157788b5'
    }
}

def download(url: str, filename: str) -> None:
    print(f'Downloading {url} to {filename}')

    resp = requests.get(url, stream=True)
    content = resp.json()['content']

    with open(filename, 'wb') as f:
        f.write(base64.b64decode(content))

class WeightRepository:
    @staticmethod
    def get_weights(classname: str, include_top: bool, name: str,
                    root: str = MODELS_DIR):
        key = (classname, include_top, name)

        if not key in _weights:
            raise ValueError(f'Invalid name {name}')

        path = os.path.join(root, _weights[key]['filename'])

        if not os.path.isfile(path):
            sha = _weights[key]['sha']
            url = ('https://api.github.com/repos/estenhl/pyment-public/git'
                   f'/blobs/{sha}')
            download(url, path)

        return path
