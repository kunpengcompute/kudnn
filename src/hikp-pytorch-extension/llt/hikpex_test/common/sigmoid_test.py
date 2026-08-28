import re
import torch
from torch import nn
from operator_test_base import OperatorTest
from hikp_pytorch_extension.common.Sigmoid import Hikp_Sigmoid

class SigmoidTest(OperatorTest):
    def __init__(self, para_str, N, item):
        super().__init__(para_str, N, item)
        self.ref = nn.Sigmoid.forward

    def gen_input_data(self):
        B, S = 512, 779 # typical length of tensor in application
        self.input = torch.randn(B, S, self.input_num, requires_grad=False)
    
    def build_reference_net(self):
        self.net = nn.Sigmoid()

    def reference_forward(self):
        return self.net(self.input)
    
    def hikpex_opti_forward(self):
        return self.net(self.input)

    def replace_model_with_hikpex_fwd(self):
        nn.Sigmoid.forward = Hikp_Sigmoid
    
    def replace_model_with_init_fwd(self):
        nn.Sigmoid.forward = self.ref
    
    def get_para_with_str(self):
        paras = re.split("[=:,%\[\]\s]+", self.para_str)
        assert len(paras) == 1
        self.input_num = int(paras[0])

def get_test_obj(para_str, test_item, iteration):
    return SigmoidTest(para_str, iteration, test_item)
