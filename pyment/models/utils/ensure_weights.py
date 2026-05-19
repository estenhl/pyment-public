import os

from pyment.utils.download_file import download_file

IDENTIFIERS = {
    'multi-2025': {
        'data': 'f4054d701fa59971fb7000d38cf9e63a202bd66a',
        'index': '9c208ca0bcc3969ceb281ba63a8cee4944a63157',
    },
    'multi-2025-no-abcd': {
        'data': '564cd7b3b89f280f41abf126d063279def15c828',
        'index': 'a6c1c4c7216212dbd4c901c2f229de9a15d23c46',
    },
    'reg-2025': {
        'data': 'a876e76e382b0da7375f81686a20620415817b7f',
        'index': '0748f0c6e83272d74735ba7b1a796a43519df66f',
    },
}
BASE_URL = 'https://api.github.com/repos/estenhl/pyment-public/git/blobs'


def _lookup_identifier(identifier: str, local_cache: str) -> str:
    if not (
        os.path.isfile(os.path.join(local_cache, f'{identifier}.index'))
        and os.path.isfile(
            os.path.join(local_cache, f'{identifier}.data-00000-of-00001')
        )
    ):
        if not os.path.isdir(local_cache):
            os.makedirs(local_cache, exist_ok=True)

        download_file(
            url=BASE_URL + '/' + IDENTIFIERS[identifier]['data'],
            destination=os.path.join(
                local_cache, f'{identifier}.data-00000-of-00001'
            ),
            description=f'Downloading {identifier} data',
            decode_github=True,
        )
        download_file(
            url=BASE_URL + '/' + IDENTIFIERS[identifier]['index'],
            destination=os.path.join(local_cache, f'{identifier}.index'),
            description=f'Downloading {identifier} index',
            decode_github=True,
        )

    return os.path.join(local_cache, identifier)


def ensure_weights(
    identifier: str,
    local_cache: str = os.path.join(
        os.path.expanduser('~'), '.pyment', 'weights'
    ),
) -> str:
    """Takes either a path or an identifier for a valid weight
    configuration as an argument, and returns a path-prefix to files
    containing the weights. If necessary, the weights are downloaded.

    Parameters
    ----------
    identifier : str
        Points to either a filename or a valid keyword identifiying a
        weight file.

    Returns
    -------
    str
        A path that prefixes files containing the weights.

    Raises
    ------
    KeyError
        If the identifier is not a valid identifier and there does not
        exist either a single file <identifier> or files
        <identifier>.index and <identifier>.data-00000-of-00001 on the
        local file system.
    """
    if (
        os.path.isfile(f'{identifier}.index')
        and os.path.isfile(f'{identifier}.data-00000-of-00001')
    ) or (os.path.isfile(identifier)):
        return identifier
    elif identifier in IDENTIFIERS:
        return _lookup_identifier(identifier, local_cache)
    else:
        raise NotImplementedError(
            f'{identifier} is not a valid file prefix nor a known identifier'
        )
