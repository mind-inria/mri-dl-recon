import torch
import torch.nn as nn
import torch.nn.functional as F

def to_complex(x, n):
    real_part = torch.tensor(x[..., :n], dtype=torch.float32)
    imag_part = torch.tensor(x[..., n:], dtype=torch.float32)
    return torch.view_as_complex(torch.stack([real_part, imag_part], dim=-1))


def to_real(x):
    real_part = torch.real(x)
    imag_part = torch.imag(x)
    return torch.cat([real_part, imag_part], dim=-1)


def _concatenate_real_imag(x):
    x_real = x.real
    x_imag = x.imag
    return torch.cat([x_real, x_imag], dim=-1)


def _complex_from_half(x, n, output_shape):
    return to_complex(x, n)


def conv2d_complex(x, n_filters, n_convs, activation='relu', output_shape=None, res=False, last_kernel_size=3):
    x_real_imag = _concatenate_real_imag(x)
    n_complex = output_shape[-1]

    for j in range(n_convs):
        x_real_imag = nn.Conv2d(
            in_channels=n_filters if j == 0 else 2 * n_complex,
            out_channels=n_filters,
            kernel_size=3,
            padding='same'
        )(x_real_imag)
        x_real_imag = F.relu(x_real_imag) if activation == 'relu' else x_real_imag

    x_real_imag = nn.Conv2d(
        in_channels=2 * n_complex,
        out_channels=2 * n_complex,
        kernel_size=last_kernel_size,
        padding='same'
    )(x_real_imag)

    x_real_imag = to_complex(x_real_imag, n_complex)

    if res:
        x_final = x + x_real_imag
    else:
        x_final = x_real_imag

    return x_final


# input_array = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
# input_tensor = torch.tensor(input_array)

# # Utiliser la fonction to_complex
# complex_tensor = to_complex(input_tensor, n=2)
# b = to_real(complex_tensor)
# a = _concatenate_real_imag(complex_tensor)
# c = _complex_from_half(input_tensor,n=2, output_shape=(3, 2))

# # Afficher le résultat
# print("\nSortie (tensor complexe):")
# print(c)

