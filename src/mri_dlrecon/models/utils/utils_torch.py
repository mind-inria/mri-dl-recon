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


def load_and_transform(path):
    hf = h5py.File(path)
    kspace = hf['kspace'][()]
    kspace = torch.tensor(kspace , dtype=torch.complex64)
    return kspace

def create_zero_filled_reconstruction(mask, kspace):
    masked_data , _ = mask(kspace.shape) 
    masked_kspace = kspace * masked_data
    masked_image = torch.fft.fftshift(torch.fft.ifft2(masked_kspace))

    masked_image = masked_image.unsqueeze(1)
    return masked_image, masked_data



##Zach
def ortho_fft2d(image):
    image = image.to(dtype=torch.complex64)
    axes = [len(image.shape) - 2, len(image.shape) - 1]
    scaling_norm = torch.sqrt(torch.tensor(image.size(-2) * image.size(-1), dtype=torch.float32)).to(image.dtype)
    if len(image.shape) == 4:
        # multicoil case
        ncoils = image.shape[1]
    n_slices = image.shape[0]
    i_shape_x = image.shape[-2]
    i_shape_y = image.shape[-1]
    shifted_image = torch.fft.fftshift(image)
    batched_shifted_image = shifted_image.view(-1, i_shape_x, i_shape_y)
    batched_shifted_kspace = torch.stack([torch.fft.fft2(img) for img in batched_shifted_image])
    if len(image.shape) == 4:
        # multicoil case
        kspace_shape = [n_slices, ncoils, i_shape_x, i_shape_y]
    elif len(image.shape) == 3:
        kspace_shape = [n_slices, i_shape_x, i_shape_y]
    else:
        kspace_shape = [i_shape_x, i_shape_y]
    shifted_kspace = batched_shifted_kspace.view(kspace_shape)
    kspace = torch.fft.ifftshift(shifted_kspace)
    return kspace / scaling_norm

def ortho_ifft2d(kspace):
    axes = [len(kspace.shape) - 2, len(kspace.shape) - 1]
    scaling_norm = torch.sqrt(torch.tensor(kspace.size(-2) * kspace.size(-1), dtype=torch.float32)).to(kspace.dtype)
    if len(kspace.shape) == 4:
        # multicoil case
        ncoils = kspace.shape[1]
    n_slices = kspace.shape[0]
    k_shape_x = kspace.shape[-2]
    k_shape_y = kspace.shape[-1]
    shifted_kspace = torch.fft.ifftshift(kspace)
    batched_shifted_kspace = shifted_kspace.view((-1, k_shape_x, k_shape_y))
    batched_shifted_image = torch.stack([torch.fft.ifft2(ksp) for ksp in batched_shifted_kspace])
    if len(kspace.shape) == 4:
        # multicoil case
        image_shape = [n_slices, ncoils, k_shape_x, k_shape_y]
    elif len(kspace.shape) == 3:
        image_shape = [n_slices, k_shape_x, k_shape_y]
    else:
        image_shape = [k_shape_x, k_shape_y]
    shifted_image = batched_shifted_image.view(image_shape)
    image = torch.fft.fftshift(shifted_image)
    return scaling_norm * image
 