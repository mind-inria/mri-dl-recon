import torch
import torch.nn as nn

# _torch_crop, prend une image en entrée (im) 
# effectue un recadrage sur cette image en fonction des paramètres spécifiés (cropx et cropy)

def _torch_crop(im, cropx=320, cropy=None):
    if cropy is None:
        cropy = cropx

    y, x = im.shape[1], im.shape[2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)

    if cropy % 2 != 0:
        starty -= 1

    im = im[:, starty:starty+cropy, startx:startx+cropx, :]
    return im


def torch_fastmri_format(image):
    image = _torch_crop(torch.abs(image))
    return image

def general_fastmri_format(image, output_shape=None):
    abs_image = torch.abs(image)
    if output_shape is None:
        cropy = cropx = 320
    else:
        # We make the assumption that all output images have the same shape
        # for this batch (or volume)
        cropx = output_shape[0][1].squeeze().item()
        cropy = output_shape[0][0].squeeze().item()
    cropped_image = _torch_crop(abs_image, cropx=cropx, cropy=cropy)
    return cropped_image

# image = torch.randn(3, 256, 256, 3)
# cropped_image = general_fastmri_format(image)
# print("Image originale :", image.shape)
# print("Image recadrée :", cropped_image.shape)

