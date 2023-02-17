import argparse
import os

from typing import List


def _parse_requirements(requirements: str) -> List[str]:
    with open(requirements, 'r') as f:
        requirements = [x.strip() for x in f.readlines()]

    return requirements

def generate_setup_py(version: str, author: str, email: str,
                      destination: str, requirements: str):
        if os.path.isfile(destination):
            raise ValueError((f'File {destination} already exists. This script '
                              'will not overwrite, so if this is intentional, '
                              'remove the old file and rerun the script'))

        requirements = _parse_requirements(requirements)
        requirements = [f'"{package}"' for package in requirements]
        requirements = ',\n        '.join(requirements)

        s = ('from distutils.core import setup\n' + \
             '\n' + \
             'setup(\n' + \
             '    name="pyment-public",\n' + \
            f'    version="{version}",\n' + \
            f'    author="{author}",\n' + \
            f'    author_email="{email}",\n' + \
             '    packages=["pyment"],\n' + \
             '    url="https://github.com/estenhl/pyment-public",\n' + \
             '    install_requires=[\n' + \
            f'        {requirements}\n' + \
             '    ]\n' + \
             ')')

        print(s)

        with open(destination, 'w') as f:
            f.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Generates a setup.py-file for '
                                      'pyment-public'))

    parser.add_argument('-v', '--version', required=True,
                        help='Version of the repository')
    parser.add_argument('-a', '--author', required=False,
                        default='Esten Høyland Leonardsen',
                        help='Name of the author')
    parser.add_argument('-e', '--email', required=False,
                        default='estenleonardsen@gmail.com',
                        help='Email of the author')
    parser.add_argument('-d', '--destination', required=False,
                        default='setup.py',
                        help='Destination where setup-file is written')
    parser.add_argument('-r', '--requirements', required=False,
                        default='requirements.txt',
                        help='Path to requirements file')

    args = parser.parse_args()

    generate_setup_py(version=args.version, author=args.author,
                      email=args.email, destination=args.destination,
                      requirements=args.requirements)
