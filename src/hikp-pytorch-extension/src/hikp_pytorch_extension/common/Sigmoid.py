import torch
from torch import nn
import hikp_pytorch_extension._C as kernel

def Hikp_Sigmoid(self, input):
    return kernel.common.hikp_sigmoid_fwd(input)