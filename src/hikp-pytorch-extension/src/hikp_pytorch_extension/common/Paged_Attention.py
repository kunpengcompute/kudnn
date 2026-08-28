import torch.autograd as autograd
import hikp_pytorch_extension._C as kernel

def scaled_dot_product_paged_attention(q, k_caches, v_caches, block_tables, seq_lens):
    output = kernel.common.scaled_dot_product_paged_attention_cpu(q, k_caches, v_caches, block_tables, seq_lens)
    return output