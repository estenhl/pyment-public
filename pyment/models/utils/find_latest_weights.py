import os
import re
import numpy as np


def find_latest_weights(checkpoints: str) -> str:
    folders = [folder for folder in os.listdir(checkpoints) \
               if os.path.isdir(os.path.join(checkpoints, folder))]
    folders = [folder for folder in folders \
               if re.fullmatch(r'epoch=\d+-.*', folder)]

    epochs = [int(re.fullmatch(r'epoch=(\d+)-.*', folder).groups(0)[0]) \
              for folder in folders]

    return folders[np.argmax(epochs)]
