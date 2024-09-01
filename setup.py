from distutils.core import setup
from setuptools import find_packages

setup(
    name='pyment-public',
    version='3.0.0',
    author='Esten Høyland Leonardsen',
    author_email='estenleonardsen@gmail.com',
    packages=find_packages(),
    url='https://github.com/estenhl/pyment-public',
    install_requires=[
        'jupyter',
        'pytest',
        'tqdm',
        'nibabel',
        'pandas',
        'xlrd',
        'plotly',
        'scikit-learn',
        'tensorflow',
        'matplotlib'
    ],
    include_package_data=True,
    package_data={
        'pyment': ['data/*'],
    }
)
