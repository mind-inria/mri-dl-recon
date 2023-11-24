import torch 
from itertools import chain

def pad_for_pool(inputs, n_pools):
    inputs_padded, paddings = pad_for_pool_whole_plane(inputs, n_pools)
    return inputs_padded, paddings[-1]

def pad_for_pool_whole_plane(inputs, n_pools):

    problematic_dims = torch.tensor(inputs.shape[-3:-1])
    # print(problematic_dims)
    k = problematic_dims // 2 ** n_pools
    # print(k)

    n_pad = torch.where(
            torch.eq(torch.fmod(problematic_dims, 2 ** n_pools), 0),
            torch.zeros_like(problematic_dims),
            (k + 1) * 2 ** n_pools - problematic_dims
    )
    # print(n_pad)

    padding_left = torch.where(
        torch.logical_or(
            torch.eq(torch.fmod(problematic_dims, 2), 0),
            torch.eq(n_pad, 0),
        ),
        n_pad // 2,
        n_pad // 2 + 1,
    )
    print("left", padding_left)

    padding_right = n_pad//2
    print("right", padding_right)
    paddings_short = [(padding_left[i].item(), padding_right[i].item()) for i in range(2)]
    print("short", paddings_short)

    paddings = [
        (0, 0),
        paddings_short[0],
        paddings_short[1],
        (0, 0),
    ]
    print("list", paddings)
    new_padding = tuple(chain(*paddings))
    print("final", new_padding)


    inputs_padded = torch.nn.functional.pad(inputs, new_padding)
    print(inputs_padded)
    return(inputs_padded, paddings_short)

inputs = torch.rand((4, 13, 8, 3))
n_pools = 2

a,b = pad_for_pool(inputs, n_pools)

print("Données d'entrée PT:\n", inputs)
print("\nDonnées d'entrée rembourrées :\n", a)
print("\nPaddings utilisés :\n", b)



# inputs = tf.random.normal((4, 13, 8, 3))
# n_pools = 2

# a,b = pad_for_pool(inputs, n_pools)

# print("Données d'entrée PT:\n", inputs)
# print("\nDonnées d'entrée rembourrées :\n", a)
# print("\nPaddings utilisés :\n", b)