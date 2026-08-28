import re
import time
import torch
from torch import nn
import torch.nn.functional as F
from operator_test_base import OperatorTest
from hikp_pytorch_extension.common.Flash_Attention import scaled_dot_product_flash_attention

class FlashAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask, is_causal=False):
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask, 0.0, is_causal)
        return output

class FlashAttentionTest(OperatorTest):
    def __init__(self, para_str, N, item):
        super().__init__(para_str, N, item)
        self.ref = F.scaled_dot_product_attention

    def gen_input_data(self):
        self.q = torch.randn(self.batch, self.nHeads, self.qSeqLen, self.headSize, requires_grad=False, dtype=self.data_type)
        self.k = torch.randn(self.batch, self.nHeads, self.kvSeqLen, self.headSize, requires_grad=False, dtype=self.data_type)
        self.v = torch.randn(self.batch, self.nHeads, self.kvSeqLen, self.headSize, requires_grad=False, dtype=self.data_type)
        if self.apply_mask == 0:
            self.attn_mask = None
            self.attn_mask_ref = None
        else:
            self.attn_mask = torch.randn(self.batch, self.nHeads, self.qSeqLen, self.kvSeqLen, requires_grad=False, dtype=self.mask_type)
            self.attn_mask_ref = self.attn_mask.clone().to(self.data_type)
    
    def build_reference_net(self):
        self.net = FlashAttention()

    def reference_forward(self):
        return self.net(self.q, self.k, self.v, self.attn_mask_ref, self.is_causal)
    
    def hikpex_opti_forward(self):
        return self.net(self.q, self.k, self.v, self.attn_mask, self.is_causal)

    def replace_model_with_hikpex_fwd(self):
        F.scaled_dot_product_attention = scaled_dot_product_flash_attention
    
    def replace_model_with_init_fwd(self):
        F.scaled_dot_product_attention = self.ref
    
    def get_para_with_str(self):
        paras = re.split("[=:,%\[\]\s]+", self.para_str)
        assert len(paras) == 9
        self.batch = int(paras[0])
        self.nHeads = int(paras[1])
        self.qSeqLen = int(paras[2])
        self.kvSeqLen = int(paras[3])
        self.headSize = int(paras[4])
        self.apply_mask = int(paras[5])
        self.is_causal = True if int(paras[6]) == 1 else False
        if paras[7] == "fp16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float
        if paras[8] == "fp16":
            self.mask_type = torch.float16
        else:
            self.mask_type = torch.float

    def judge_results_close(self, ref, result):
        deviation = 1e-10 if self.data_type == torch.float else 1e-6
        return torch.sum((ref - result) ** 2).item() < ref.numel() * deviation

    def performance_test(self):
        forward = 0
        self.replace_model_with_hikpex_fwd()
        self.hikpex_opti_forward()
        for _ in range(self.iteration):  # Optimized PyTorch layer Forward and Backward pass timing
            start = time.time()
            self.hikpex_opti_forward()
            forward += time.time() - start
        print("Hikp Forward pass time (Optimized layer): {:.3f} ms".format(forward * 1e3 / self.iteration))
        self.replace_model_with_init_fwd()

def get_test_obj(para_str, test_item, iteration):
    return FlashAttentionTest(para_str, iteration, test_item)
