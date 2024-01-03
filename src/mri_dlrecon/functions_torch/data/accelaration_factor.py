import torch

def torch_af(mask):
    mask_int = mask.to(torch.int32)
    return mask_int.shape[0] / torch.sum(mask_int)

