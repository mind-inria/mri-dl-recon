"""
Test for all functions reliated to the data. 

The tests compare functions in tensorflow and pytorch.
"""

import sys
sys.path.append('/home/lo276838/Modèles/mri-dl-recon/src')

from mri_dlrecon.models.functions_tf.utils_tf import virtual_coil_reconstruction as mtos_tf
from mri_dlrecon.models.utils.utils_torch import virtual_coil_reconstruction as mtos_torch
import fastmri

import numpy as np
import pytest
import h5py



@pytest.fixture
def download_data() :
    file_path = "/volatile/FastMRI/brain_multicoil_train/multicoil_train/file_brain_AXT1POST_201_6002780.h5"
    
    hf = h5py.File(file_path)
    volume_kspace = hf['kspace'][()]
    return volume_kspace


def test_combine_images(download_data):
    # TensorFlow
    tf_output = mtos_tf(download_data)

    # PyTorch
    pt_output = mtos_torch(download_data)

    # RSS de fastmri 
    # rss_output = fastmri.rss(download_data, dim=0)


    # Assurez-vous que les formes sont correctes
    assert tf_output.shape == pt_output.shape

    # Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
    np.testing.assert_almost_equal(tf_output.numpy(), pt_output.numpy(), decimal=1)

    # # Ces fonction sont differente de fastmri.rss car complex.
    # assert tf_output.shape == rss_output.shape
    # np.testing.assert_almost_equal(rss_output.numpy(), tf_output.numpy(), decimal=4)
