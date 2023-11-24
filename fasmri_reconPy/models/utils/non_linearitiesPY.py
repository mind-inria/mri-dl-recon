from torch.nn.functional import leaky_relu

def lrelu(x, alpha=0.1):
    return leaky_relu(x, negative_slope=alpha)
