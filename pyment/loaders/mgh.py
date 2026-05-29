import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

_MGH_DTYPES = {
    0: {'dtype': tf.uint8, 'bytes': 1},
    1: {'dtype': tf.int32, 'bytes': 4},
    3: {'dtype': tf.float32, 'bytes': 4},
    4: {'dtype': tf.int16, 'bytes': 2},
}
_VALID_MGH_DTYPES = list(_MGH_DTYPES.keys())
_MGH_BYTE_SIZES = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=_VALID_MGH_DTYPES,
        values=[_MGH_DTYPES[key]['bytes'] for key in _VALID_MGH_DTYPES],
    ),
    default_value=tf.constant(-1, dtype=tf.int32),
    name='nifti_dtypes',
)


def _decode(imagebytes: tf.Tensor, dtype: tf.Tensor) -> tf.Tensor:
    cases = [
        (
            tf.equal(dtype, key),
            lambda k=key: tf.cast(
                tf.io.decode_raw(
                    imagebytes, _MGH_DTYPES[k]['dtype'], little_endian=False
                ),
                tf.float32,
            ),
        )
        for key in _VALID_MGH_DTYPES
    ]

    return tf.case(cases, exclusive=True)


def _parse(
    buffer: tf.Tensor, start: int, size: int, dtype: tf.dtypes.DType
) -> tf.Tensor:
    bytestring = tf.strings.substr(buffer, start, size)

    decoded = tf.io.decode_raw(bytestring, dtype, little_endian=False)
    decoded = tf.cond(
        tf.equal(tf.rank(decoded), 1),
        lambda: tf.squeeze(decoded),
        lambda: decoded,
    )
    return decoded


def _parse_header(tfbytes: tf.Tensor) -> dict[str, tf.Tensor]:
    version = _parse(tfbytes, 0, 4, tf.int32)
    tf.debugging.assert_equal(
        version,
        tf.constant(1, dtype=tf.int32),
        message='Unknown MGH version',
    )

    width = _parse(tfbytes, 4, 4, tf.int32)
    height = _parse(tfbytes, 8, 4, tf.int32)
    depth = _parse(tfbytes, 12, 4, tf.int32)
    shape = tf.stack([width, height, depth], axis=0)

    dtype = _parse(tfbytes, 20, 4, tf.int32)
    bytes_per_voxel = _MGH_BYTE_SIZES.lookup(dtype)

    rasflag = _parse(tfbytes, 28, 2, tf.int16)
    tf.debugging.assert_equal(
        rasflag,
        tf.constant(1, dtype=tf.int16),
        message='Load not implemented when goodRASflag is not set',
    )

    return {
        'version': version,
        'width': width,
        'height': height,
        'depth': depth,
        'shape': shape,
        'dtype': dtype,
        'bytes_per_voxel': bytes_per_voxel,
        'rasflag': rasflag,
    }


def load_mgh(path: str | tf.Tensor) -> tf.Tensor:
    mghdata = tf.io.read_file(path)
    mghdata = tf.cond(
        tf.strings.regex_full_match(path, r'.*\.mgz$'),
        lambda: tf.io.decode_compressed(mghdata, 'GZIP'),
        lambda: mghdata,
    )

    header = _parse_header(mghdata)
    imagesize = tf.reduce_prod(header['shape']) * header['bytes_per_voxel']
    imagesize = tf.squeeze(imagesize)
    imagebytes = tf.strings.substr(mghdata, 284, imagesize)
    image = _decode(imagebytes, header['dtype'])
    image = tf.reshape(image, header['shape'])
    image = tf.cast(image, tf.float32)
    image = tf.transpose(image, [2, 1, 0])

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Loads an MGH image')

    parser.add_argument('-p', '--path', type=str, help='Path to the MGH image')

    args = parser.parse_args()

    path = tf.constant(args.path)
    image = load_mgh(path=tf.constant(args.path))
    image = image.numpy()
    print(image.shape)
    center = np.asarray(image.shape) // 2

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].imshow(image[112, :, :], cmap='Greys_r')
    ax[0].axis('off')
    ax[1].imshow(image[:, 112, :], cmap='Greys_r')
    ax[1].axis('off')
    ax[2].imshow(image[:, :, 112], cmap='Greys_r')
    ax[2].axis('off')

    plt.show()
