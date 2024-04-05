import os
import sys

from shutil import rmtree


TESTS_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.join(TESTS_FOLDER, os.pardir)
METADATA_FOLDER = os.path.join(ROOT_FOLDER, '.data')

# Appends the root pyment directory to the path
def pytest_configure(config):
    testpath = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(testpath, os.pardir)
    sys.path.append(libpath)

class TemporaryFolder:
    def __init__(self, path: str = os.path.join(TESTS_FOLDER, 'tmp')):
        self.path = path

    def __enter__(self):
        os.mkdir(self.path)

        return self.path

    def __exit__(self, *args):
        rmtree(self.path)
