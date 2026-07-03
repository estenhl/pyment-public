from pyment.utils.strip_extension import strip_extension


def test_strip_extension_nii_gz():
    assert strip_extension('sub-01.nii.gz') == 'sub-01', (
        'Expected strip_extension to strip .nii.gz as a single unit'
    )


def test_strip_extension_single_extension():
    assert strip_extension('sub-01.mgz') == 'sub-01', (
        'Expected strip_extension to strip a single extension'
    )


def test_strip_extension_no_extension():
    assert strip_extension('sub-01') == 'sub-01', (
        'Expected strip_extension to return filename unchanged when it '
        'has no extension'
    )


def test_strip_extension_leaves_path_intact():
    assert strip_extension('a/b/sub-01.nii.gz') == 'a/b/sub-01', (
        'Expected strip_extension to only strip the extension, leaving '
        'any leading path unchanged'
    )
