import torch
from torch import nn
import hikp_pytorch_extension._C as kernel

def OuterProductMean_forward(self, act, mask, block_size=256):
    assert block_size > 0
    mask = mask[..., None]
    epsilon = 1e-3
    norm = torch.einsum('abc,adc->bdc', mask, mask)
    act = self.layer_norm_input(act)
    Nseq = mask.size(1)
    act = kernel.alphafold.outer_product_mean_fwd(
        act,
        mask,
        Nseq,
        self.layer_norm_input.weight,
        self.layer_norm_input.bias,
        self.left_projection.weight,
        self.left_projection.bias,
        self.right_projection.weight,
        self.right_projection.bias,
        self.output_w,
        self.output_b,
        block_size)
    act /= epsilon + norm
    return act