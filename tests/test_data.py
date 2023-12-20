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
import tensorflow as tf

# from fastmri.data import transforms as T
# import fastmri


@pytest.fixture
def download_data() :
    # Récuperation des données
    file_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"
    # file_path = "/home/lo276838/Modèles/mri-dl-recon/src/mri_dlrecon/data/brain_data/multicoil_test/file1.h5"
    hf = h5py.File(file_path)

    volume_kspace = hf['kspace'][()]

    return volume_kspace

@pytest.fixture
def sample_data_torch(download_data):

    volume_kspace= torch.tensor(download_data, dtype=torch.complex64)

    dimension = len(volume_kspace.shape) -2
    if dimension == 2 :
        images = torch.fft.ifft2(volume_kspace)
    elif dimension == 3 : 
        images = torch.fft.ifftn(volume_kspace)

    images = torch.fft.fftshift(images)

    return volume_kspace


@pytest.fixture
def sample_data_tf(download_data):

    volume_kspace = download_data

    dimension = len(volume_kspace.shape)-2
    if dimension == 2 :
        images = tf.convert_to_tensor(tf.signal.ifft2d(volume_kspace), dtype=tf.complex64)
    elif dimension == 3 : 
        images = tf.signal.ifft3d(volume_kspace)
    images = tf.signal.fftshift(images)

    return volume_kspace


def test_combine_images(sample_data_tf, sample_data_torch):
    # TensorFlow
    tf_output = mtos_tf(sample_data_tf)
    tf_output = torch.tensor(tf_output.numpy())

    # PyTorch
    pt_output = mtos_torch(sample_data_torch)




    # RSS de fastmri 
    rss_output = fastmri.rss(sample_data_torch, dim=0)

    # Assurez-vous que les formes sont correctes
    assert tf_output.shape == pt_output.shape

    # Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
    # np.testing.assert_almost_equal(tf_output.numpy(), pt_output.numpy(), decimal=1)

    # Ces fonction sont differente de fastmri.rss car complex.
    # assert tf_output.shape == rss_output.shape
    # np.testing.assert_almost_equal(rss_output.numpy(), tf_output.numpy(), decimal=4)
