from matplotlib import pyplot as plt
import torch
import h5py

def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)



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
        fft_result = torch.fft.fft2(difference_original_vs_virtual)
        shape_want = fft_result.shape[-1]
        hanning = hanning[:, :shape_want]

        difference_original_vs_virtual = torch.fft.ifft2( fft_result * hanning )
    
    img_comb = torch.sum(
        imgs *
        torch.exp(
            1j * torch.angle(difference_original_vs_virtual.to(torch.complex64))
        ),
        dim=-3
    )

    return img_comb


def load_and_transform(path):
    hf = h5py.File(path)
    kspace = hf['kspace'][()]
    kspace = torch.tensor(kspace , dtype=torch.complex64)
    return kspace

def create_masked_Image(mask, kspace,):
    masked_data , _ = mask(kspace.shape) 
    masked_kspace = kspace * masked_data
    masked_image = torch.fft.fftshift(torch.fft.ifft2(masked_kspace))

    masked_image = masked_image.unsqueeze(1)
    masked_image = masked_image.abs()
    return masked_image