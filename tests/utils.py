from matplotlib import pyplot as plt
import torch

# def show_coils(data, slice_nums, cmap=None):
#     """
#     Display image slices from a 3D dataset.

#     Parameters:
#     - data: 3D array representing the dataset.
#     - slice_nums: List of integers specifying the indices of slices to be displayed.
#     - cmap: Colormap for image display (optional). 
    
#     """

#     fig = plt.figure()
#     for i, num in enumerate(slice_nums):
#         plt.subplot(1, len(slice_nums), i + 1)
#         plt.imshow(data[num], cmap=cmap)


def virtual_coil_reconstruction(imgs):
    imgs = torch.tensor(imgs, dtype=torch.complex64)
    img_sh = imgs.shape 
    # print(img_sh)

    dimension = len(img_sh) - 2
    # print(dimension)

    # Compute d'abord la bobine virtuelle
    weights = torch.sum(torch.abs(imgs), dim=1) + 1e-16
    # print(weights)

    phase_reference = torch.tensor(
        torch.angle(torch.sum(
            imgs,
            dim=tuple(2 + torch.arange(len(img_sh) - 2))
        )),
        dtype=torch.complex64
    )
    # print(phase_reference)

    expand = [Ellipsis, *((None, ) * (len(img_sh) - 2))]
    # print(expand)

    reference = imgs / weights[:, None, ...].to(torch.complex64) / \
        torch.exp(1j * phase_reference)[expand]
    # print(reference)
    # print(reference.shape)

    virtual_coil = torch.sum(reference, dim=1)
    # print(virtual_coil)
    # print(virtual_coil.shape)
    
    difference_original_vs_virtual = torch.conj(imgs) * virtual_coil.unsqueeze(1)
    # print(difference_original_vs_virtual)
    # print(difference_original_vs_virtual.shape)

    # Hanning filtering in readout and phase direction
    hanning = torch.hann_window(img_sh[-dimension])
    # print(hanning)

    for d in range(dimension - 1):
        hanning = hanning.unsqueeze(-1) * torch.hann_window(img_sh[dimension + d])
        # print(hanning)
    hanning = hanning.to(torch.complex64)
    # print(hanning)

    # dimension= 2
    # print(dimension)
    
    # Removing the background noise via low pass filtering
    if dimension == 3:
        difference_original_vs_virtual = torch.fft.ifftn(
            torch.fft.fftn(difference_original_vs_virtual) * torch.fft.fftshift(hanning)
        )
        # print(difference_original_vs_virtual)
    else:
        difference_original_vs_virtual = torch.fft.ifft2(
            torch.fft.fft2(difference_original_vs_virtual) * hanning
        )
        # print(difference_original_vs_virtual)
    
    img_comb = torch.sum(
        imgs *
        torch.exp(
            1j * torch.angle(difference_original_vs_virtual.to(torch.complex64))
        ),
        dim=-3
    )
    return img_comb