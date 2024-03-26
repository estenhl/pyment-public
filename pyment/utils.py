import math
import os
import requests

from tqdm import tqdm

ROOT_DIR = os.path.join(os.path.expanduser('~'), '.pyment')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
