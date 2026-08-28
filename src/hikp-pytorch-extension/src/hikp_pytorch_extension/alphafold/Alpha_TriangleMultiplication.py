import torch
from torch import nn
import hikp_pytorch_extension._C as kernel

def TriangleMultiplication_forward(self, act, mask, block_size=256):
    assert block_size > 0
    mask = mask[..., None]
    if self.c_equation == "ikc,jkc->ijc":  # "Outgoing" edges equation
        equation_flag = 0
    else:  # "Incoming" edges equation
        equation_flag = 1
    act = kernel.alphafold.triangle_multiplication_fwd(
        act,
        mask,
        equation_flag,
        self.layer_norm_input.weight,
        self.layer_norm_input.bias,
        self.left_projection.weight,
        self.left_projection.bias,
        self.left_gate.weight,
        self.left_gate.bias,
        self.right_projection.weight,
        self.right_projection.bias,
        self.right_gate.weight,
        self.right_gate.bias,
        self.center_layer_norm.weight,
        self.center_layer_norm.bias,
        self.output_projection.weight,
        self.output_projection.bias,
        self.gating_linear.weight,
        self.gating_linear.bias,
        block_size
    )
    return act