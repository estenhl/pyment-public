import os
import sys


def configure_environment():
    """Adds the pyment-library to the system path. Convenient for
    running scripts without properly setting up the environment"""
    current_directory = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.join(current_directory, os.pardir)

    sys.path.append(root_directory)
