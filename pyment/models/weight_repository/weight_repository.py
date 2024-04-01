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
    },
    ('BinarySFCN', True, 'dementia-2024'): {
        'filename': 'binary_sfcn_dementia_2024_fold_0_weights.h5',
        'sha': '1f43aafd2461d7e5b4f9ebb6d62e0f2ab363e1b8'
    },
    ('BinarySFCN', True, 'dementia-2024-fold-0'): {
        'filename': 'binary_sfcn_dementia_2024_fold_0_weights.h5',
        'sha': '1f43aafd2461d7e5b4f9ebb6d62e0f2ab363e1b8'
    },
    ('BinarySFCN', True, 'dementia-2024-fold-1'): {
        'filename': 'binary_sfcn_dementia_2024_fold_1_weights.h5',
        'sha': 'a0da6b724f3c1477ae2f461c49a91b7d2f46ac72'
    },
    ('BinarySFCN', True, 'dementia-2024-fold-2'): {
        'filename': 'binary_sfcn_dementia_2024_fold_2_weights.h5',
        'sha': 'cec0eb79f043a3415f5ab13977dfda24e1f7dc30'
    },
    ('BinarySFCN', True, 'dementia-2024-fold-3'): {
        'filename': 'binary_sfcn_dementia_2024_fold_3_weights.h5',
        'sha': 'c885fee44d4839d37d8bcdfd970391788ee85004'
    },
    ('BinarySFCN', True, 'dementia-2024-fold-4'): {
        'filename': 'binary_sfcn_dementia_2024_fold_4_weights.h5',
        'sha': '35d3b0343b83a9851a140cab7baed2dd36e35185'
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
