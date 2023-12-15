"""
Test for all functions reliated to the data. 

The tests compare functions in tensorflow and pytorch.
"""

import sys
sys.path.append('/home/lo276838/Modèles/mri-dl-recon/src')


from mri_dlrecon.functions_tf.utils_tf import virtual_coil_reconstruction as mtos_tf
from mri_dlrecon.functions_torch.utils_torch import virtual_coil_reconstruction as mtos_torch

import numpy as np
import torch
import pytest
import h5py

from fastmri.data import transforms as T
import fastmri


@pytest.fixture
def sample_data():

    # Récuperation des données
    # file_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"
    file_path = "/home/lo276838/Modèles/mri-dl-recon/src/mri_dlrecon/data/brain_data/multicoil_test/file1.h5"
    hf = h5py.File(file_path)

    volume_kspace = hf['kspace'][()]
    slice_kspace = volume_kspace[volume_kspace.shape[0]-1]

    slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
    slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
    slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image

    return slice_image_abs


def test_combine_images(sample_data):
    # TensorFlow
    tf_output = mtos_tf(sample_data)
    tf_output = torch.tensor(tf_output.numpy())

    # PyTorch
    pt_output = mtos_torch(sample_data)

    # RSS de fastmri 
    rss_output = fastmri.rss(sample_data, dim=0)

    # Assurez-vous que les formes sont correctes
    assert tf_output.shape == pt_output.shape

    # Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
    np.testing.assert_almost_equal(tf_output.numpy(), pt_output.numpy(), decimal=1)

    # Ces fonction sont differente de fastmri.rss car complex.
    # assert tf_output.shape == rss_output.shape
    # np.testing.assert_almost_equal(rss_output.numpy(), tf_output.numpy(), decimal=4)
