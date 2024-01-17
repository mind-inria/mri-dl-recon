import sys
sys.path.append('/home/lo276838/Modèles/mri-dl-recon/src')

import pytest
import torch
from mri_dlrecon.data.gen_mask_torch import gen_mask_torch

@pytest.mark.parametrize('fixed_masks', [True, False])
@pytest.mark.parametrize('multicoil', [True, False])
def test_gen_mask_torch(fixed_masks, multicoil):
    kspace = torch.rand([16, 16, 640, 320], dtype=torch.float32)  
    kspace = kspace.view(16, 16, 640, 160,2)
    kspace = torch.view_as_complex(kspace) 

    accel_factor = 4
    mask = gen_mask_torch(kspace, accel_factor, multicoil, fixed_masks)

    if fixed_masks:
        mask_again = gen_mask_torch(kspace, accel_factor, multicoil, fixed_masks)
        torch.testing.assert_close(mask, mask_again)