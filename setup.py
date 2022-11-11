from distutils.core import setup

setup(
    name="pyment-public",
    version="1.0.1",
    author="Esten Høyland Leonardsen",
    author_email="estenleonardsen@gmail.com",
    packages=["pyment"],
    url="https://github.com/estenhl/pyment-public",
    install_requires=[
        "jupyterlab",
        "matplotlib",
        "mock",
        "nibabel",
        "numpy",
        "pandas",
        "pytest",
        "requests",
        "scikit-learn",
        "tqdm",
        "xlrd"
    ]
)