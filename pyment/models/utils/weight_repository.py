import os

from ...utils import download


_mapping = {
    ('RegressionSFCN', 'brain_age', True): {
        'url': ('https://drive.google.com/u/0/'
                'uc?id=196e6ddCaUfdVWdeEaYVRQOEgE5_414b7&export=download'),
        'filename': 'regression_sfcn_brain_age_weights.hdf5'
    },
    ('RegressionSFCN', 'brain_age', False): {
        'url': ('https://drive.google.com/u/0/'
                'uc?id=1Mev57Rdst5TAxGEt3OUY97ToKOWq0uCC&export=download'),
        'filename': 'regression_sfcn_brain_age_weights_no_top.hdf5'
    }
}


class WeightRepository:
    root = os.path.join(os.path.expanduser('~'), '.pyment', 'models')

    @staticmethod
    def get_path(model: str, weights: str, include_top: bool):
        key = (model, weights, include_top)

        if not key in _mapping:
            raise ValueError((f'Weights \'{weights}\' does not exist for model '
                              f'{model} with include_top={include_top}'))

        filename = _mapping[key]['filename']
        path = os.path.join(WeightRepository.root, filename)

        if not os.path.isfile(path):
            if not os.path.isdir(WeightRepository.root):
                os.makedirs(WeightRepository.root)

            url = _mapping[key]['url']
            download(url, path)

        return path

