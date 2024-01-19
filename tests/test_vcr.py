"""
Test for all functions reliated to the data. 

The tests compare functions in tensorflow and pytorch.
"""

import sys
sys.path.append('/home/lo276838/Modèles/mri-dl-recon/src')

from mri_dlrecon.models.utils.vcr import virtual_coil_reconstruction as mtos_torch
from mri_dlrecon.models.utils.data_transform import load_and_transform
import fastmri

import numpy as np
import pytest
import torch


@pytest.fixture
def download_data() :
    file_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"
    kspace_multicoil = load_and_transform(file_path)
    images_multicoil = torch.fft.fftshift(torch.fft.ifft2(kspace_multicoil))

    return images_multicoil




def test_combine_images(download_data):

    # PyTorch
    pt_output = mtos_torch(download_data)

    # RSS de fastmri 
    rss_output = fastmri.rss(download_data, dim=0)


    # Assurez-vous que les formes sont correctes
    assert rss_output.shape == pt_output.shape

    # Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
    np.testing.assert_almost_equal(rss_output.numpy(), pt_output.numpy(), decimal=1)
