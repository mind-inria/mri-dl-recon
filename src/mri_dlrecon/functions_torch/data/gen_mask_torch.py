import torch

def gen_mask_torch(kspace, accel_factor, multicoil=False, fixed_masks=False):
    shape = torch.tensor(kspace.shape)
    num_cols = shape[-1]
    center_fraction = (32 // accel_factor) / 100
    num_low_freqs = int(round(num_cols.item() * center_fraction))
    
    prob = (num_cols / accel_factor - num_low_freqs) / (num_cols - num_low_freqs)
    
    if fixed_masks:
        torch.manual_seed(0)
        seed = 0
    else:
        seed = None
    
    mask = torch.rand((1, num_cols), dtype=torch.float64).to(device='cpu') < prob
    
    pad = (num_cols - num_low_freqs + 1) // 2
    final_mask = torch.cat([
        mask[:, :pad],
        torch.ones((1, num_low_freqs), dtype=torch.bool),
        mask[:, pad + num_low_freqs:],
    ], dim=1)

    # Reshape the mask
    mask_shape = torch.ones_like(shape)
    if multicoil:
        mask_shape = mask_shape[:3]
    else:
        mask_shape = mask_shape[:2]
    
    final_mask_shape = torch.cat([
        mask_shape,
        torch.unsqueeze(num_cols, dim=0),
    ], dim=0)
    
    final_mask_reshaped = final_mask.view(final_mask_shape.tolist())
    
    # Add batch dimension for cases where the batch is split across multiple GPUs
    if multicoil:
        final_mask_reshaped = final_mask_reshaped.repeat(shape[0], 1, 1, 1)
    else:
        final_mask_reshaped = final_mask_reshaped.repeat(shape[0], 1, 1)
    
    fourier_mask = final_mask_reshaped.to(dtype=torch.uint8)
    return fourier_mask


# # Example k-space data (you can replace this with your own data)
# kspace_data = torch.randn(1, 256, 256, 2) 

# # Acceleration factor
# acceleration_factor = 4

# # Generate Fourier domain mask
# fourier_mask = gen_mask_torch(kspace_data, acceleration_factor, multicoil=True)

# # Print the generated mask
# print("Generated Fourier Domain Mask:")
# print(fourier_mask)

# # Verify the shape of the generated mask
# print("Mask Shape:", fourier_mask.shape)
