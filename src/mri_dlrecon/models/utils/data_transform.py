import torch
import h5py


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
 