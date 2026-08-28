import torch
from torch import nn
import hikp_pytorch_extension._C as kernel

def GatingAttention_forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor(), block_size=128):
    assert block_size > 0
    bias = bias.contiguous()
    nonbatched_bias = nonbatched_bias.contiguous()
    out = kernel.alphafold.gating_attention_fwd(
        q_data,
        m_data,
        self.query_w,
        self.key_w,
        self.value_w,
        bias,
        nonbatched_bias,
        self.gating_w,
        self.gating_b,
        self.output_w,
        self.output_b,
        block_size,
    )
    return out