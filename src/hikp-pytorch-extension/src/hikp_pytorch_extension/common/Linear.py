import torch
from torch import nn
import hikp_pytorch_extension._C as kernel

def Hikp_Linear(self, input, block_size=128):
    assert block_size > 0
    hidden_size = self.weight.size(1)
    assert input.size(-1) == hidden_size
    return kernel.common.hikp_linear_fwd(input, self.weight, self.bias)

