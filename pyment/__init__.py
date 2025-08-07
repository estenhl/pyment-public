import os
import tomllib

def _get_version():
    """Get version from pyproject.toml"""
    pyproject_path = os.path.join(
        os.path.dirname(__file__), os.pardir, 'pyproject.toml'
    )

    with open(pyproject_path, 'rb') as f:
        data = tomllib.load(f)

    return data['project']['version']

__version__ = _get_version()
