import re
import torch
from torch import nn
from operator_test_base import OperatorTest
from hikp_pytorch_extension.alphafold.Alpha_Attention import GatingAttention_forward

class GatingAttention(nn.Module):
    def __init__(self, num_head, a_dim, m_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        # k,v dim
        self.key_dim = int(a_dim)
        self.value_dim = int(m_dim)
        self.num_head = num_head
        assert self.key_dim % self.num_head == 0
        assert self.value_dim % self.num_head == 0
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head
        # q,k,v weights
        self.query_w = nn.Parameter(
            torch.Tensor(a_dim, self.num_head, self.key_dim), requires_grad=False
        )
        self.key_w = nn.Parameter(
            torch.Tensor(m_dim, self.num_head, self.key_dim), requires_grad=False
        )
        self.value_w = nn.Parameter(
            torch.Tensor(m_dim, self.num_head, self.value_dim), requires_grad=False
        )
        self.gating_w = nn.Parameter(
            torch.Tensor(a_dim, self.num_head, self.value_dim), requires_grad=False
        )
        self.gating_b = nn.Parameter(
            torch.Tensor(self.num_head, self.value_dim), requires_grad=False
        )
        self.output_w = nn.Parameter(
            torch.Tensor(self.num_head, self.value_dim, self.output_dim),
            requires_grad=False,
        )
        self.output_b = nn.Parameter(torch.Tensor(self.output_dim), requires_grad=False)
        # softmax & act fn
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor(), block_size=64):
        """Builds Attention module.
        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
          nonbatched_bias: Shared bias, shape [N_queries, N_keys].
        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """
        # get query, key, value
        q = torch.einsum("bqa,ahc->bqhc", q_data, self.query_w) * self.key_dim ** (-0.5)
        k = torch.einsum("bka,ahc->bkhc", m_data, self.key_w)
        v = torch.einsum("bka,ahc->bkhc", m_data, self.value_w)

        logits = torch.einsum("bqhc,bkhc->bhqk", q, k) + bias

        if nonbatched_bias.shape[0] > 0:
            logits += torch.unsqueeze(nonbatched_bias, dim=0)
        weights = self.softmax(logits)

        weighted_avg = torch.einsum("bhqk,bkhc->bqhc", weights, v)

        gate_values = (
            torch.einsum("bqc,chv->bqhv", q_data, self.gating_w) + self.gating_b
        )
        gate_values = self.sigmoid(gate_values)
        weighted_avg *= gate_values

        output = (
            torch.einsum("bqhc,hco->bqo", weighted_avg, self.output_w) + self.output_b
        )
        return output

class GatingAttentionTest(OperatorTest):
    def __init__(self, para_str, N, item):
        super().__init__(para_str, N, item)
        self.ref = GatingAttention.forward

    def gen_input_data(self):
        self.q_data = torch.randn(self.B, self.S, self.HS, requires_grad=False)
        self.m_data = torch.randn(self.B, self.S, self.HS, requires_grad=False)
        self.bias = torch.randn(self.B, 1, 1, self.S, requires_grad=False)
        if self.flag == 0:
            self.nonbatched_bias = torch.Tensor()
        else:
            self.nonbatched_bias = torch.randn(self.N, self.S, self.S, requires_grad=False)
    
    def build_reference_net(self):
        self.H = self.HS // self.N
        attention = GatingAttention(self.N, self.HS, self.HS, self.HS)
        attention.query_w.data = torch.randn(self.HS, self.N, self.H)
        attention.key_w.data = torch.randn(self.HS, self.N, self.H)
        attention.value_w.data = torch.randn(self.HS, self.N, self.H)
        attention.gating_w.data = torch.randn(self.HS, self.N, self.H)
        attention.gating_b.data = torch.randn(self.N, self.H)
        attention.output_w.data = torch.randn(self.N, self.H, self.HS)
        attention.output_b.data = torch.randn(self.HS)
        self.net = attention

    def reference_forward(self):
        return self.net(self.q_data, self.m_data, self.bias, self.nonbatched_bias)
    
    def hikpex_opti_forward(self):
        return self.net(self.q_data, self.m_data, self.bias, self.nonbatched_bias)

    def replace_model_with_hikpex_fwd(self):
        GatingAttention.forward = GatingAttention_forward
    
    def replace_model_with_init_fwd(self):
        GatingAttention.forward = self.ref
    
    def get_para_with_str(self):
        paras = re.split("[=:,%\[\]\s]+", self.para_str)
        assert len(paras) == 5
        self.B = int(paras[0])
        self.S = int(paras[1])
        self.HS = int(paras[2])
        self.N = int(paras[3])
        self.flag = int(paras[4])
        assert self.N != 0

def get_test_obj(para_str, test_item, iteration):
    return GatingAttentionTest(para_str, iteration, test_item)
