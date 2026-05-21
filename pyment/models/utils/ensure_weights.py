import os

from pyment.utils.download_file import download_file

IDENTIFIERS = {
    'multi-2025': 'c18be19f64b4e398c446d871a7e9d35ec8c88f1d',
    'multi-2025-no-abcd': '1c8e9bc7c33af1625044fb1a54a03500ecc050c4',
    'reg-2025': 'a8baaed43082ebac70427a8bf122ffd7c63b51e2',
}
BASE_URL = 'https://api.github.com/repos/estenhl/pyment-public/git/blobs'
DEFAULT_CACHE = os.path.join(os.path.expanduser('~'), '.pyment', 'weights')


def _lookup_identifier(identifier: str, local_cache: str) -> str:
    path = os.path.join(local_cache, f'{identifier}.h5')
    if not os.path.isfile(path):
        if not os.path.isdir(local_cache):
            os.makedirs(local_cache, exist_ok=True)
        download_file(
            url=BASE_URL + '/' + IDENTIFIERS[identifier],
            destination=path,
            description=f'Downloading {identifier}',
            decode_github=True,
        )
    return path


def ensure_weights(
    identifier: str,
    local_cache: str = DEFAULT_CACHE,
) -> str:
    """Takes either a path or an identifier for a valid weight
    configuration as an argument, and returns a path to a file
    containing the weights. If necessary, the weights are downloaded.

    Parameters
    ----------
    identifier : str
        Points to either a local .h5 file or a known weight identifier.

    Returns
    -------
    str
        A path to an .h5 file containing the weights.

    Raises
    ------
    NotImplementedError
        If the identifier is not a valid file path nor a known
        identifier.
    """
    if os.path.isfile(identifier):
        return identifier
    elif identifier in IDENTIFIERS:
        return _lookup_identifier(identifier, local_cache)
    else:
        raise NotImplementedError(
            f'{identifier} is not a valid file path nor a known identifier'
        )
