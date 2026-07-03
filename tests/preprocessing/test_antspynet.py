from unittest.mock import patch

import ants
import numpy as np

from pyment.preprocessing import rescale_intensity, skullstrip_with_antspynet

SHAPE = (10, 12, 14)
RAMP_SHAPE = (7, 11, 13)


def make_ants_image(
    shape: tuple = SHAPE,
    spacing: tuple = (1.0, 1.0, 1.0),
    value: float = 1.0,
) -> ants.ANTsImage:
    data = np.full(shape, value, dtype='float32')
    return ants.from_numpy(data, spacing=spacing)


def make_mask(reference: ants.ANTsImage, region: tuple) -> ants.ANTsImage:
    data = np.zeros(reference.shape, dtype='float32')
    data[region] = 1.0
    return ants.from_numpy(
        data,
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


def make_ramp_image(shape: tuple = RAMP_SHAPE) -> ants.ANTsImage:
    data = np.arange(np.prod(shape), dtype='float32').reshape(shape)
    return ants.from_numpy(data, spacing=(1.0, 1.0, 1.0))


def test_rescale_intensity_maps_f_low_percentile_to_dst_min():
    image = make_ramp_image()

    result = rescale_intensity(image)

    assert result.numpy().min() == 0.0, (
        'Expected rescale_intensity to map the f_low percentile of image '
        'to dst_min'
    )


def test_rescale_intensity_maps_f_high_percentile_to_dst_max():
    image = make_ramp_image()

    result = rescale_intensity(image)

    # RAMP_SHAPE holds 1001 distinct sorted values 0..1000, so the
    # default f_high=0.999 percentile lands exactly on value 999.
    assert result.numpy()[image.numpy() == 999.0][0] == 255.0, (
        'Expected rescale_intensity to map the f_high percentile of image '
        'to dst_max'
    )


def test_rescale_intensity_clips_values_above_f_high_percentile():
    image = make_ramp_image()

    result = rescale_intensity(image)

    assert result.numpy()[image.numpy() == 1000.0][0] == 255.0, (
        'Expected rescale_intensity to clip values above the f_high '
        'percentile to dst_max'
    )


def test_rescale_intensity_respects_custom_dst_range():
    image = make_ramp_image()

    result = rescale_intensity(image, dst_min=10.0, dst_max=20.0)

    data = result.numpy()
    assert data.min() == 10.0 and data.max() == 20.0, (
        'Expected rescale_intensity to map onto a custom [dst_min, dst_max] '
        'range when given'
    )


def test_rescale_intensity_handles_constant_image():
    image = make_ants_image(value=5.0)

    result = rescale_intensity(image)

    assert np.all(result.numpy() == 0.0), (
        'Expected rescale_intensity to yield dst_min throughout for a '
        'constant image, for which there is no contrast to stretch'
    )


def test_rescale_intensity_preserves_image_geometry():
    image = make_ramp_image()

    result = rescale_intensity(image)

    assert result.spacing == image.spacing, (
        'Expected rescale_intensity to leave spacing unchanged'
    )
    assert result.shape == image.shape, (
        'Expected rescale_intensity to leave shape unchanged'
    )


def test_skullstrip_with_antspynet_reorients_image():
    image = make_ants_image()

    with patch(
        'pyment.preprocessing.antspynet.antspynet.brain_extraction'
    ) as mock_brain_extraction:
        mock_brain_extraction.side_effect = lambda resampled, modality: (
            make_mask(resampled, tuple(slice(0, s) for s in resampled.shape))
        )
        result = skullstrip_with_antspynet(image, orientation='RSP')

    assert result.orientation == 'RSP', (
        'Expected skullstrip_with_antspynet to yield an image with '
        'orientation matching the orientation parameter'
    )


def test_skullstrip_with_antspynet_resamples_to_target_resolution():
    image = make_ants_image(spacing=(2.0, 2.0, 2.0))

    with patch(
        'pyment.preprocessing.antspynet.antspynet.brain_extraction'
    ) as mock_brain_extraction:
        mock_brain_extraction.side_effect = lambda resampled, modality: (
            make_mask(resampled, tuple(slice(0, s) for s in resampled.shape))
        )
        result = skullstrip_with_antspynet(
            image, target_resolution=(1.0, 1.0, 1.0)
        )

    assert np.allclose(result.spacing, 1.0), (
        'Expected skullstrip_with_antspynet to yield an image with spacing '
        'matching the target_resolution parameter'
    )


def test_skullstrip_with_antspynet_crops_to_mask_bounding_box():
    image = make_ants_image()

    with patch(
        'pyment.preprocessing.antspynet.antspynet.brain_extraction'
    ) as mock_brain_extraction:
        mock_brain_extraction.side_effect = lambda resampled, modality: (
            make_mask(resampled, np.s_[2:5, 2:5, 2:5])
        )
        result = skullstrip_with_antspynet(image)

    assert result.shape == (3, 3, 3), (
        'Expected skullstrip_with_antspynet to crop the output to the '
        'bounding box of the brain mask'
    )


def test_skullstrip_with_antspynet_masks_out_non_brain_voxels():
    region = np.s_[2:5, 2:5, 2:5]
    data = np.zeros(SHAPE, dtype='float32')
    data[region] = 5.0
    image = ants.from_numpy(data, spacing=(1.0, 1.0, 1.0))

    with patch(
        'pyment.preprocessing.antspynet.antspynet.brain_extraction'
    ) as mock_brain_extraction:
        mock_brain_extraction.side_effect = lambda resampled, modality: (
            make_mask(resampled, region)
        )
        result = skullstrip_with_antspynet(image, orientation=image.orientation)

    assert np.all(result.numpy() == 255.0), (
        'Expected skullstrip_with_antspynet to preserve brain voxels, '
        'rescaled to dst_max by rescale_intensity and cropped to the brain '
        'mask bounding box'
    )


def test_skullstrip_with_antspynet_passes_modality_to_brain_extraction():
    image = make_ants_image()

    with patch(
        'pyment.preprocessing.antspynet.antspynet.brain_extraction'
    ) as mock_brain_extraction:
        mock_brain_extraction.side_effect = lambda resampled, modality: (
            make_mask(resampled, tuple(slice(0, s) for s in resampled.shape))
        )
        skullstrip_with_antspynet(image, modality='t2')

    assert mock_brain_extraction.call_args.kwargs['modality'] == 't2', (
        'Expected skullstrip_with_antspynet to forward the modality '
        'parameter to antspynet.brain_extraction'
    )
