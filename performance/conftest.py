import os
import sys


def pytest_configure(config):
    testpath = os.path.dirname(os.path.abspath(__file__))
    libpath = os.path.join(testpath, os.pardir)
    sys.path.append(libpath)

    results_folder = os.path.join(os.path.dirname(__file__), 'results')

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
