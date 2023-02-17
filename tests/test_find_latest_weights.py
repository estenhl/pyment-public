import os

from shutil import rmtree

from pyment.models.utils import find_latest_weights


def test_find_latest_weights():
    checkpoints = 'tmp'

    try:
        os.mkdir(checkpoints)
        folders = [
            'epoch=1-loss=1.23-val_loss=123',
            'epoch=15-loss=2.34-val_loss=234',
            'epoch=105-loss=0.12-val_loss=12',
            'random_folder'
        ]

        for folder in folders:
            os.mkdir(os.path.join(checkpoints, folder))

        latest_weights = find_latest_weights(checkpoints)

        assert 'epoch=105-loss=0.12-val_loss=12' == latest_weights, \
            ('find_latest_weights does not return the weights from the latest '
             'epoch')
    finally:
        rmtree(checkpoints)
