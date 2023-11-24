import torch
from masking import _mask_torch
import torch.nn as nn

class MultiplyScalar(nn.Module):
    def __init__(self, **kwargs):
        super(MultiplyScalar, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.sample_weight = self.add_weight(
            name='sample_weight',
            shape=(1,),
            initializer='ones',
            trainable=True,
        )
        super(MultiplyScalar, self).build(input_shape)  # Be sure to call this at the end
        # self.sample_weight = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)

    
    def forward(self, x):
        # return torch.complex(self.sample_weight, torch.tensor(0.0)) * x
        return torch.view_as_complex(self.sample_weight) * x
    



# Example usage:
# Create an instance of the MultiplyScalar layer
multiply_layer = MultiplyScalar()

# Dummy input tensor
input_tensor = torch.randn(5, 5)

# Forward pass
output_tensor = multiply_layer(input_tensor)