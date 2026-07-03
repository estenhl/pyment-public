"""Utility for stripping known image file extensions from filenames."""

import os


def strip_extension(filename: str) -> str:
    """Strips the extension from filename.

    Handles the double extension of ``.nii.gz`` as a single unit;
    any other extension is stripped via ``os.path.splitext``.

    Parameters
    ----------
    filename : str
        A filename, with or without a leading path.

    Returns
    -------
    str
        filename with its extension removed.
    """
    if filename.endswith('.nii.gz'):
        return filename[: -len('.nii.gz')]

    return os.path.splitext(filename)[0]
