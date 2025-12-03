def _get_version():
    """Get version from package metadata (generated from pyproject.toml during installation)"""
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version('pyment')
    except PackageNotFoundError:
        import os
        import tomli

        pyproject_path = os.path.join(
            os.path.dirname(__file__), os.pardir, 'pyproject.toml'
        )
        if os.path.exists(pyproject_path):
            with open(pyproject_path, 'rb') as f:
                data = tomli.load(f)

            return data['project']['version']


__version__ = _get_version()
