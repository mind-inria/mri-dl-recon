import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def show_coils(data, slice_nums, cmap=None):
    """
    Display image slices from a 3D dataset.

    Parameters:
    - data: 3D array representing the dataset.
    - slice_nums: List of integers specifying the indices of slices to be displayed.
    - cmap: Colormap for image display (optional). 
    
    """

    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)



def adjust_image_size(image, target_image_size, multicoil=False):
    height = tf.shape(image)[-2]
    width = tf.shape(image)[-1]
    n_slices = tf.shape(image)[0]
    transpose_axis = [1, 2, 0] if not multicoil else [2, 3, 0, 1]
    transposed_image = tf.transpose(image, transpose_axis)
    reshaped_image = tf.reshape(transposed_image, [height, width, -1])  # 3D tensors accepted
    # with channels dimension last
    target_height = target_image_size[0]
    target_width = target_image_size[1]
    padded_image = tf.image.resize_with_crop_or_pad(
        reshaped_image,
        target_height,
        target_width,
    )
    if multicoil:
        final_shape = [target_height, target_width, n_slices, -1]
    else:
        final_shape = [target_height, target_width, n_slices]
    reshaped_padded_image = tf.reshape(padded_image, final_shape)
    transpose_axis = [2, 0, 1] if not multicoil else [2, 3, 0, 1]
    transpose_padded_image = tf.transpose(reshaped_padded_image, transpose_axis)
    return transpose_padded_image




def virtual_coil_reconstruction(imgs):
    """
    Calculate the combination of all coils using virtual coil reconstruction

    Parameters
    ----------
    imgs: np.ndarray
        The images reconstructed channel by channel
        in shape [batch_size, Nch, Nx, Ny, Nz]

    Returns
    -------
    img_comb: np.ndarray
        The combination of all the channels in a complex valued
        in shape [batch_size, Nx, Ny]
    """
    imgs = tf.constant(imgs, dtype=tf.complex64)
    img_sh = imgs.shape
    dimension = len(img_sh)-2

    weights = tf.math.reduce_sum(tf.abs(imgs), axis=1) + 1e-16
    phase_reference = tf.cast(
        tf.math.angle(tf.math.reduce_sum(
            imgs,
            axis=(2+np.arange(len(img_sh)-2))
        )),
        tf.complex64
    )

    expand = [..., *((None, ) * (len(img_sh)-2))]
    reference = imgs / tf.cast(weights[:, None, ...], tf.complex64) / \
        tf.math.exp(1j * phase_reference)[expand]
    virtual_coil = tf.math.reduce_sum(reference, axis=1)
    difference_original_vs_virtual = tf.math.conj(imgs) * virtual_coil[:, None]

    hanning = tf.signal.hann_window(img_sh[-dimension])
    for d in range(dimension-1):
        hanning = tf.expand_dims(hanning, axis=-1) * tf.signal.hann_window(img_sh[dimension + d])
    hanning = tf.cast(hanning, tf.complex64)

    if dimension == 3:    
        difference_original_vs_virtual = tf.signal.ifft3d(
            tf.signal.fft3d(difference_original_vs_virtual) * tf.signal.fftshift(hanning)
        )
    else:
        fft_result = tf.signal.fft2d(difference_original_vs_virtual)
        shape_want = fft_result.shape[-1]
        hanning = tf.slice(hanning, begin=[0, 0], size=[-1, shape_want])

        difference_original_vs_virtual = tf.signal.ifft2d( fft_result * hanning )
    
    img_comb = tf.math.reduce_sum(
        imgs *
        tf.math.exp(
            1j * tf.cast(tf.math.angle(difference_original_vs_virtual), tf.complex64)),
        axis=1
    )
    return img_comb


