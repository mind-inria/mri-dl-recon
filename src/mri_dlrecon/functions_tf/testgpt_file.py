import tensorflow as tf


def TFV(imgs):
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
    # Compute first the virtual coil
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
    # Hanning filtering in readout and phase direction
    hanning = tf.signal.hann_window(img_sh[-dimension])
    for d in range(dimension-1):
        hanning = tf.expand_dims(hanning, axis=-1) * tf.signal.hann_window(img_sh[dimension + d])
    hanning = tf.cast(hanning, tf.complex64)
    # Removing the background noise via low pass filtering
    if dimension == 3:    
        difference_original_vs_virtual = tf.signal.ifft3d(
            tf.signal.fft3d(difference_original_vs_virtual) * tf.signal.fftshift(hanning)
        )
    else:
        difference_original_vs_virtual = tf.signal.ifft2d(
            tf.signal.fft2d(difference_original_vs_virtual) * hanning
        )
    img_comb = tf.math.reduce_sum(
        imgs *
        tf.math.exp(
            1j * tf.cast(tf.math.angle(difference_original_vs_virtual), tf.complex64)),
        axis=1
    )
    return img_comb


import numpy as np

# Créer des données d'exemple
batch_size = 1
num_channels = 4
image_size = (32, 32, 32)  # Dimensions de l'image en 3D

# Créer des images d'exemple
imgs_example = np.random.randn(batch_size, num_channels, *image_size)
print(imgs_example.shape)

# Appeler la fonction de reconstruction de bobine virtuelle
combined_image = TFV(imgs_example)

# Afficher les résultats
print("Forme de l'image combinée :", combined_image.shape)