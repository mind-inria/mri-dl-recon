import sys
sys.path.append('/home/lo276838/Modèles/mri-dl-recon/src/mri_dlrecon')

import torch
from models.utils.data_transform import ortho_fft2d, ortho_ifft2d


def virtual_coil_reconstruction(imgs):
    """
        Reconstruct a virtual coil image from a set of input images. (multi coil to single coil)

        Parameters:
        - imgs: Input images from different coils, represented as a complex tensor.
        with : shape [batch_size, Nch, Nx, Ny, Nz]

        Returns:
        - img_comb: Reconstructed virtual coil image.
        with shape [batch_size, Nx, Ny]
    """

    # imgs = imgs.clone().detach()
    img_sh = imgs.shape 
    dimension = len(img_sh) - 2
    # Compute first the virtual coil
    weights = torch.sum(torch.abs(imgs), dim=1) + 1e-16
    phase_reference = torch.angle(torch.sum(imgs, 
                                            dim=tuple(2+torch.arange(len(img_sh)-2))
                                            )).clone().detach()
    expand = [Ellipsis, *((None, ) * (len(img_sh) - 2))]

    reference = imgs / weights[:, None, ...].to(torch.complex64) / \
        torch.exp(1j * phase_reference)[expand]
    virtual_coil = torch.sum(reference, dim=1)
    
    difference_original_vs_virtual = torch.conj(imgs) * virtual_coil.unsqueeze(1)
    hanning = torch.hann_window(img_sh[-dimension])
    for d in range(dimension - 1):
        hanning = hanning.unsqueeze(-1) * torch.hann_window(img_sh[dimension + d])
    hanning = hanning.to(torch.complex64)

    if dimension == 3:
        fft_result = torch.fft.fftn(difference_original_vs_virtual) 
        hanning = torch.fft.fftshift(hanning) 

        difference_original_vs_virtual = torch.fft.ifftn( fft_result * hanning )

    else:
        fft_result = ortho_fft2d(difference_original_vs_virtual)
        shape_want = fft_result.shape[-1]
        hanning = hanning[:, :shape_want]

        difference_original_vs_virtual = ortho_ifft2d( fft_result * hanning )
    
    img_comb = torch.sum(
        imgs *
        torch.exp(
            1j * torch.angle(difference_original_vs_virtual.to(torch.complex64))
        ),
        dim=1
    )

    return img_comb
