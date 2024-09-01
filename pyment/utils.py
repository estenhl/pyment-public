import os
import importlib.resources as pkg_resources

DATA_DIR = os.path.join(os.path.expanduser('~'), '.pyment')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

try:
    METADATA_DIR = pkg_resources.files('pyment.data')
except:
    METADATA_DIR = os.path.join(ROOT_DIR, 'data')
