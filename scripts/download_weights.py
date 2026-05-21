"""Downloads pretrained weights by identifier.

Used by CI workflows before building Docker images, and locally to
populate either the default cache or the checkpoints folder for a
manual docker build.

Examples
--------
# populate default cache (~/.pyment/weights/), all identifiers
python scripts/download_weights.py

# populate checkpoints folder for docker build
python scripts/download_weights.py --destination checkpoints/pyment

# download a specific identifier only
python scripts/download_weights.py --identifiers multi-2025
"""

import argparse

from pyment.models.utils.ensure_weights import (
    DEFAULT_CACHE,
    IDENTIFIERS,
    ensure_weights,
)


def download_weights(
    identifiers: list[str] | None = None,
    destination: str = DEFAULT_CACHE,
) -> None:
    if identifiers is None:
        identifiers = list(IDENTIFIERS)

    for identifier in identifiers:
        ensure_weights(identifier, local_cache=destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download pretrained pyment weights'
    )
    parser.add_argument(
        '-i',
        '--identifiers',
        nargs='+',
        choices=list(IDENTIFIERS),
        default=None,
        metavar='IDENTIFIER',
        help=(
            f'One or more weight identifiers to download. '
            f'Choices: {", ".join(IDENTIFIERS)}. '
            f'Defaults to all.'
        ),
    )
    parser.add_argument(
        '-d',
        '--destination',
        default=DEFAULT_CACHE,
        help=(
            'Directory to write weight files into. Defaults to '
            f'{DEFAULT_CACHE}.'
        ),
    )
    args = parser.parse_args()

    download_weights(identifiers=args.identifiers, destination=args.destination)
