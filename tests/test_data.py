"""
Test for all functions reliated to the data. 

The tests compare functions in tensorflow and pytorch.
"""

from utils import show_coils
# from functions_torch.utils import show_coils

import numpy as np
import pytest
import torch
# import tensorflow as tf
# from your_module import combine_images_tf, combine_images_pt 


# @pytest.fixture
# def sample_data():
#     # Générer des données de test
#     batch_size = 2
#     Nch = 3
#     Nx, Ny, Nz = 32, 32, 1

#     # Créer des images multicoils
#     imgs = np.random.randn(batch_size, Nch, Nx, Ny, Nz)

#     return imgs

# def test_combine_images(sample_data):
#     # TensorFlow
#     tf_output = combine_images_tf(sample_data)

#     # PyTorch
#     pt_input = torch.from_numpy(sample_data)
#     pt_output = combine_images_pt(pt_input.numpy())

#     # Assurez-vous que les formes sont correctes
#     assert tf_output.shape == pt_output.shape

#     # Assurez-vous que les valeurs sont proches (tolérance peut être ajustée)
#     assert np.allclose(tf_output, pt_output, rtol=1e-4, atol=1e-6)
