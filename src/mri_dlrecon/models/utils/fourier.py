import sys
sys.path.append('/home/lo276838/Modèles/mri-dl-recon/src')

import torch
import torch.nn as nn

from mri_dlrecon.models.utils.masking import _mask


def _compute_scaling_norm(x):
    image_area = torch.prod(torch.tensor(x.shape[-2:]).float())
    scaling_norm = torch.sqrt(image_area)
    scaling_norm = scaling_norm.type(x.dtype)
    return scaling_norm


class FFTBase(nn.Module):
    def __init__(self, masked, multicoil=False, use_smaps=True, **kwargs):
        super(FFTBase, self).__init__(**kwargs)
        self.masked = masked
        self.multicoil = multicoil
        self.use_smaps = use_smaps
        if self.multicoil:
            self.shift_axes = [2, 3]
        else:
            self.shift_axes = [1, 2]

    def get_config(self):
        config = super(FFTBase, self).get_config()
        config.update({'masked': self.masked})
        config.update({'multicoil': self.multicoil})
        config.update({'use_smaps': self.use_smaps})
        return config

    def op(self, inputs):
        if self.multicoil:
            if self.masked:
                image, mask, smaps = inputs
            else:
                image, smaps = inputs
        else:
            if self.masked:
                image, mask = inputs
            else:
                image = inputs
        image = image[..., 0]
        scaling_norm = _compute_scaling_norm(image)
        if self.multicoil and self.use_smaps:
            image = image[:, None, ...]
            image = image * smaps
        shifted_image = torch.fft.fftshift(image, axes=self.shift_axes)
        kspace_shifted = torch.fft.fft2(shifted_image)
        kspace_unnormed = torch.fft.ifftshift(kspace_shifted, axes=self.shift_axes)
        kspace = kspace_unnormed[..., None] / scaling_norm
        if self.masked:
            kspace = _mask([kspace, mask])
        return kspace

    def adj_op(self, inputs):
        if self.masked:
            if self.multicoil:
                kspace, mask, smaps = inputs
            else:
                kspace, mask = inputs
            kspace = _mask([kspace, mask])
        else:
            if self.multicoil:
                kspace, smaps = inputs
            else:
                kspace = inputs
        kspace = kspace[..., 0]
        scaling_norm = _compute_scaling_norm(kspace)
        shifted_kspace = torch.fft.ifftshift(kspace, axes=self.shift_axes)
        image_shifted = torch.fft.fft2(shifted_kspace)
        image_unnormed = torch.fft.fftshift(image_shifted, axes=self.shift_axes)
        image = image_unnormed * scaling_norm
        if self.multicoil and self.use_smaps:
            image = torch.sum(image * torch.conj(smaps), axis=1)
        image = image[..., None]
        return image


class FFT(FFTBase):
    def forward(self, inputs):
        return self.op(inputs)
    
class IFFT(FFTBase):
    def forward(self, inputs):
        return self.adj_op(inputs)