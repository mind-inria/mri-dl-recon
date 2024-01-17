import sys
sys.path.append('/home/lo276838/Modèles/mri-dl-recon/src')

import torch
from mri_dlrecon.models.functions_torch.denoisers.didn import DIDN


def test_didn():
    n_out = 2
    model = DIDN(
        n_filters=4,
        n_dubs=2,
        n_convs_recon=2,
        n_outputs=n_out,
    )
    res = model(torch.zeros([1, 4, 32, 32]))
    assert res.shape[-1] == n_out
