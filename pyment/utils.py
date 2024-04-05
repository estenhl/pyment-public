import os

DATA_DIR = os.path.join(os.path.expanduser('~'), '.pyment')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
METADATA_DIR = os.path.join(ROOT_DIR, '.data')
