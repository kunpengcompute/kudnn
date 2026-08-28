import re
import torch
from torch import nn
from operator_test_base import OperatorTest
from hikp_pytorch_extension.alphafold.Alpha_OuterProductMean import OuterProductMean_forward

class OuterProductMean(nn.Module):

    def __init__(self,num_outer_channel,act_dim,num_output_channel):
        super().__init__()
        """Builds OuterProductMean module.

        Arguments:
        act: MSA representation, shape [N_seq, N_res, c_m].
        mask: MSA mask, shape [N_seq, N_res].
        is_training: Whether the module is in training mode.

        Returns:
        Update to pair representation, shape [N_res, N_res, c_z].
        """
        self.num_output_channel = num_output_channel
        self.num_outer_channel = num_outer_channel
        self.layer_norm_input = nn.LayerNorm(normalized_shape=act_dim,elementwise_affine=True)
        self.left_projection = nn.Linear(act_dim,self.num_outer_channel)
        self.right_projection = nn.Linear(act_dim,self.num_outer_channel)
        self.output_w = nn.Parameter(torch.Tensor(
        self.num_outer_channel, self.num_outer_channel,self.num_output_channel))
        self.output_b = nn.Parameter(torch.Tensor(self.num_output_channel,))

    def compute_chunk(self,left_act,right_act):
        left_act = torch.transpose(left_act, 2, 1)
        act = torch.einsum('acb,ade->dceb', left_act, right_act)
        act = torch.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b
        return torch.transpose(act, 1, 0)

    def forward(self, act, mask):
        mask = mask[..., None]
        act = self.layer_norm_input(act)
        left_act = mask * self.left_projection(act)
        right_act = mask * self.right_projection(act)
        act = self.compute_chunk(left_act, right_act)
        epsilon = 1e-3
        norm = torch.einsum('abc,adc->bdc', mask, mask)
        act /= epsilon + norm
        return act

class OuterProductMeanTest(OperatorTest):
    def __init__(self, para_str, N, item):
        super().__init__(para_str, N, item)
        self.ref = OuterProductMean.forward

    def gen_input_data(self):
        self.act = torch.randn(self.B, self.S, self.HS, requires_grad=False)
        self.mask = torch.randn(self.B, self.S, requires_grad=False)
    
    def build_reference_net(self):
        net = OuterProductMean(
            num_outer_channel=self.num_outer_channel, act_dim=self.HS, num_output_channel=self.num_output_channel
        )

        layer_norm_input_weight = torch.randn(self.HS)
        layer_norm_input_bias = torch.randn(self.HS)
        left_projection_weight = torch.randn(self.num_outer_channel, self.HS, requires_grad=False)
        right_projection_weight = torch.randn(self.num_outer_channel, self.HS, requires_grad=False)
        left_projection_bias = torch.randn(self.num_outer_channel, requires_grad=False)
        right_projection_bias = torch.randn(self.num_outer_channel, requires_grad=False)
        output_w = torch.randn(self.num_outer_channel, self.num_outer_channel, self.num_output_channel)
        output_b = torch.randn(self.num_output_channel)

        net.layer_norm_input.weight.data = layer_norm_input_weight
        net.layer_norm_input.bias.data = layer_norm_input_bias
        net.left_projection.weight.data = left_projection_weight
        net.left_projection.bias.data = left_projection_bias
        net.right_projection.weight.data = right_projection_weight
        net.right_projection.bias.data = right_projection_bias
        net.output_w.data = output_w
        net.output_b.data = output_b
        self.net = net

    def reference_forward(self):
        act = self.act.detach().clone()
        return self.net(act, self.mask)
    
    def hikpex_opti_forward(self):
        act = self.act.detach().clone()
        return self.net(act, self.mask)

    def replace_model_with_hikpex_fwd(self):
        OuterProductMean.forward = OuterProductMean_forward
    
    def replace_model_with_init_fwd(self):
        OuterProductMean.forward = self.ref
    
    def get_para_with_str(self):
        paras = re.split("[=:,%\[\]\s]+", self.para_str)
        assert len(paras) == 5
        self.B = int(paras[0])
        self.S = int(paras[1])
        self.HS = int(paras[2])
        self.num_outer_channel = int(paras[3])
        self.num_output_channel = int(paras[4])

def get_test_obj(para_str, test_item, iteration):
    return OuterProductMeanTest(para_str, iteration, test_item)
