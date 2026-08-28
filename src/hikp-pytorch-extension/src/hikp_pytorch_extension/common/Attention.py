import torch.autograd as autograd
import hikp_pytorch_extension._C as kernel

def scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False):
    output, logsumexp = kernel.common.scaled_dot_product_attention_cpu(Q, K, V,
        dropout_p, is_causal, attn_mask)
    return output