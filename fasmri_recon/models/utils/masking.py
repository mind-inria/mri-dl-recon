import torch 

def _mask_torch(x):
    k_data, mask = x
    mask = mask.unsqueeze(-1).type(k_data.dtype)
    masked_k_data = torch.mul(mask, k_data)
    return masked_k_data
