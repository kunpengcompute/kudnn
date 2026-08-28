import re
import torch
from torch import nn
import torch.nn.functional as F
import time
import random
from operator_test_base import OperatorTest
from hikp_pytorch_extension.common.Paged_Attention import scaled_dot_product_paged_attention

class PagedAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def calculate_attention(self, Q, K, V):
        head_size = Q.size(-1)
        scale = 1.0 / (head_size**0.5)
        attn_weights = scale * torch.einsum("qhd,khd->hqk", Q, K).float()
        attn_weights = torch.softmax(attn_weights, dim=-1).to(Q.dtype)
        out = torch.einsum("hqk,khd->qhd", attn_weights, V)
        return out

    def forward(self, Q, k_caches, v_caches, block_tables, seq_lens):
        num_seqs = Q.shape[0]
        num_q_head = Q.shape[1]
        head_size = Q.shape[2]
        num_kv_head = k_caches.shape[1]
        block_size = k_caches.shape[3]
        output = torch.empty_like(Q)
        for i in range(num_seqs):
            q = Q[i].unsqueeze(0)
            block_table = block_tables[i]
            seq_len = int(seq_lens[i])
            keys_lst: list[torch.Tensor] = []
            values_lst: list[torch.Tensor] = []
            for j in range(seq_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = k_caches[block_number, :, :, block_offset, :]
                k = k.reshape(num_kv_head, head_size)
                keys_lst.append(k)
                v = v_caches[block_number, :, :, block_offset]
                values_lst.append(v)
            keys = torch.stack(keys_lst, dim=0)
            values = torch.stack(values_lst, dim=0)
            num_queries_per_kv = num_q_head // num_kv_head
            if num_queries_per_kv > 1:
                # Handle MQA and GQA
                keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
                values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)
            out = self.calculate_attention(q, keys, values)
            out = out.view(num_q_head, head_size)
            output[i].copy_(out, non_blocking=True)
        return output

def hikpex_scaled_dot_product_paged_attention_fwd(self, Q, k_caches, v_caches, block_tables, seq_lens):
    return scaled_dot_product_paged_attention(Q, k_caches, v_caches, block_tables, seq_lens)


class PagedAttenTest(OperatorTest):
    def __init__(self, para_str, N, item):
        super().__init__(para_str, N, item)
        self.ref = PagedAttention.forward

    def gen_input_data(self):
        NUM_BLOCKS = 32
        MAX_NUM_BLOCKS_PER_SEQ = 65
        BLOCK_SIZE = 16
        if self.data_type == torch.float16:
            split_size = 8
        else:
            split_size = 4
        self.Q = torch.randn(self.num_seqs, self.num_q_head, self.head_size, dtype=self.data_type)
        self.k_caches = torch.randn(NUM_BLOCKS, self.num_kv_head, self.head_size // split_size, BLOCK_SIZE, split_size, dtype=self.data_type)
        self.v_caches = torch.randn(NUM_BLOCKS, self.num_kv_head, self.head_size , BLOCK_SIZE, dtype=self.data_type)
        seq_lens_lst = [self.max_seq_len for _ in range(self.num_seqs)]
        block_tables_lst: list[list[int]] = []
        for _ in range(self.num_seqs):
            block_table = [
                random.randint(0, NUM_BLOCKS - 1)
                for i in range(MAX_NUM_BLOCKS_PER_SEQ)
            ]
            block_tables_lst.append(block_table)
        self.block_tables = torch.tensor(block_tables_lst, dtype=torch.int)
        self.seq_lens = torch.tensor(seq_lens_lst, dtype=torch.int)

    def build_reference_net(self):
        self.net = PagedAttention()

    def reference_forward(self):
        return self.net(self.Q, self.k_caches, self.v_caches, self.block_tables, self.seq_lens)
    
    def hikpex_opti_forward(self):
        return self.net(self.Q, self.k_caches, self.v_caches, self.block_tables, self.seq_lens)

    def replace_model_with_hikpex_fwd(self):
        PagedAttention.forward = hikpex_scaled_dot_product_paged_attention_fwd
    
    def replace_model_with_init_fwd(self):
        PagedAttention.forward = self.ref
    
    def get_para_with_str(self):
        paras = re.split("[=:,%\[\]\s]+", self.para_str)
        assert len(paras) == 7
        self.num_seqs = int(paras[0])
        self.num_q_head = int(paras[1])
        self.num_kv_head = int(paras[2])
        self.max_seq_len = int(paras[3])
        self.head_size = int(paras[4])
        self.block_size = int(paras[5])
        if paras[6] == "fp16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float

    def judge_results_close(self, ref, result):
        return torch.sum((ref - result) ** 2).item() < ref.numel() * 1e-6

    def performance_test(self):
        forward = 0
        validate = 0
        retry = 0

        self.replace_model_with_hikpex_fwd()
        while validate < self.iteration:
            start = time.time()
            self.hikpex_opti_forward()
            interval = time.time() - start
            if interval >= 0.00025:
                time.sleep(0.2)
                retry += 1
                if retry > 1000:
                    print("Hikp Forward pass time (Optimized layer): {:.3f} ms".format(interval * 1e3))
                    return
                continue
            validate += 1
            forward += interval

        print("Hikp Forward pass time (Optimized layer): {:.3f} ms".format(forward * 1e3 / self.iteration))
        self.replace_model_with_init_fwd()

def get_test_obj(para_str, test_item, iteration):
    return PagedAttenTest(para_str, iteration, test_item)