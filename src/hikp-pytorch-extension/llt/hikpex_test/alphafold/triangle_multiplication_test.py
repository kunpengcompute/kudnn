import re
import torch
from torch import nn
from operator_test_base import OperatorTest
from hikp_pytorch_extension.alphafold.Alpha_TriangleMultiplication import TriangleMultiplication_forward

class TriangleMultiplication(nn.Module):
    def __init__(self, equation, num_intermediate_channel, act_dim):
        """Builds TriangleMultiplication module.

        Arguments:
        act: Pair activations, shape [N_res, N_res, c_z]
        mask: Pair mask, shape [N_res, N_res].
        is_training: Whether the module is in training mode.

        Returns:
        Outputs, same shape/type as act.
        """
        super().__init__()
        self.c_equation = equation
        self.layer_norm_input = nn.LayerNorm(
            normalized_shape=act_dim, elementwise_affine=True
        )
        self.left_projection = nn.Linear(act_dim, num_intermediate_channel)
        self.right_projection = nn.Linear(act_dim, num_intermediate_channel)
        self.left_gate = nn.Linear(act_dim, num_intermediate_channel)
        self.right_gate = nn.Linear(act_dim, num_intermediate_channel)
        self.center_layer_norm = nn.LayerNorm(
            normalized_shape=num_intermediate_channel, elementwise_affine=True
        )
        self.output_projection = nn.Linear(num_intermediate_channel, act_dim)
        self.gating_linear = nn.Linear(num_intermediate_channel, act_dim)

    def forward(self, act, mask):
        mask = mask[..., None]
        act = self.layer_norm_input(act)
        input_act = act  # For gate
        
        left_proj_act = mask * self.left_projection(act)
        right_proj_act = mask * self.right_projection(act)
        left_proj_act *= torch.sigmoid(self.left_gate(act))
        right_proj_act *= torch.sigmoid(self.right_gate(act))
        # "Outgoing" edges equation: 'ikc,jkc->ijc'
        # "Incoming" edges equation: 'kjc,kic->ijc'
        # Note on the Suppl. Alg. 11 & 12 notation:
        # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
        # For the "incoming" edges, it's swapped:
        #   b = left_proj_act and a = right_proj_act
        
        act = torch.einsum(self.c_equation, left_proj_act, right_proj_act)
        act = self.center_layer_norm(act)
        act = self.output_projection(act)

        act *= torch.sigmoid(self.gating_linear(input_act))

        return act

class TriangleMultiplicationTest(OperatorTest):
    def __init__(self, para_str, N, item):
        super().__init__(para_str, N, item)
        self.ref = TriangleMultiplication.forward

    def gen_input_data(self):
        self.act = torch.randn(self.S, self.S, self.HS, requires_grad=False)
        self.mask = torch.rand(self.S, self.S, requires_grad=False)
    
    def build_reference_net(self):
        net = TriangleMultiplication(self.equation, self.num_intermediate_channel, self.HS)

        layer_norm_input_weight = torch.randn(self.HS)
        layer_norm_input_bias = torch.randn(self.HS)
        left_projection_weight = torch.randn(self.num_intermediate_channel, self.HS)
        left_projection_bias = torch.randn(self.num_intermediate_channel)
        right_projection_weight = torch.randn(self.num_intermediate_channel, self.HS)
        right_projection_bias = torch.randn(self.num_intermediate_channel)
        left_gate_weight = torch.randn(self.num_intermediate_channel, self.HS)
        left_gate_bias = torch.randn(self.num_intermediate_channel)
        right_gate_weight = torch.randn(self.num_intermediate_channel, self.HS)
        right_gate_bias = torch.randn(self.num_intermediate_channel)
        output_projection_weight = torch.randn(self.HS, self.num_intermediate_channel)
        output_projection_bias = torch.randn(self.HS)
        center_layer_norm_weight = torch.randn(self.num_intermediate_channel)
        center_layer_norm_bias = torch.randn(self.num_intermediate_channel)
        gating_linear_weight = torch.randn(self.HS, self.HS)
        gating_linear_bias = torch.randn(self.HS)

        net.layer_norm_input.weight = torch.nn.Parameter(layer_norm_input_weight)
        net.layer_norm_input.bias = torch.nn.Parameter(layer_norm_input_bias)
        net.left_projection.weight = torch.nn.Parameter(left_projection_weight)
        net.left_projection.bias = torch.nn.Parameter(left_projection_bias)
        net.right_projection.weight = torch.nn.Parameter(right_projection_weight)
        net.right_projection.bias = torch.nn.Parameter(right_projection_bias)
        net.left_gate.weight = torch.nn.Parameter(left_gate_weight)
        net.left_gate.bias = torch.nn.Parameter(left_gate_bias)
        net.right_gate.weight = torch.nn.Parameter(right_gate_weight)
        net.right_gate.bias = torch.nn.Parameter(right_gate_bias)
        net.center_layer_norm.weight = torch.nn.Parameter(center_layer_norm_weight)
        net.center_layer_norm.bias = torch.nn.Parameter(center_layer_norm_bias)
        net.output_projection.weight = torch.nn.Parameter(output_projection_weight)
        net.output_projection.bias = torch.nn.Parameter(output_projection_bias)
        net.gating_linear.weight = torch.nn.Parameter(gating_linear_weight)
        net.gating_linear.bias = torch.nn.Parameter(gating_linear_bias)
        self.net = net

    def reference_forward(self):
        act = self.act.detach().clone()
        return self.net(act, self.mask)
    
    def hikpex_opti_forward(self):
        act = self.act.detach().clone()
        return self.net(act, self.mask)

    def replace_model_with_hikpex_fwd(self):
        TriangleMultiplication.forward = TriangleMultiplication_forward
    
    def replace_model_with_init_fwd(self):
        TriangleMultiplication.forward = self.ref
    
    def get_para_with_str(self):
        paras = re.split("[=:,%\[\]\s]+", self.para_str)
        assert len(paras) == 4
        self.S = int(paras[0])
        self.HS = int(paras[1])
        self.num_intermediate_channel = int(paras[2])
        self.equation_flag = int(paras[3])
        if self.equation_flag == 0:
            self.equation = "ikc,jkc->ijc"
        else:
            self.equation = "kjc,kic->ijc"

def get_test_obj(para_str, test_item, iteration):
    return TriangleMultiplicationTest(para_str, iteration, test_item)