import os


def ensure_weights(identifier: str) -> str:
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
        exist files <identifier>.index and
        <identifier>.data-00000-of-00001 on the local file system.
    """
    if not (
        os.path.isfile(f'{identifier}.index') and 
        os.path.isfile(f'{identifier}.data-00000-of-00001')
    ):
        raise NotImplementedError(
            f'Identifier-based lookups are not supported'
        )

    return identifier