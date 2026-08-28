import re
import time
import torch
from torch import nn
from operator_test_base import OperatorTest
from hikp_pytorch_extension.common.Linear import Hikp_Linear

class LinearTest(OperatorTest):
    def __init__(self, para_str, N, item):
        super().__init__(para_str, N, item)
        self.ref = nn.Linear.forward

    def gen_input_data(self):
        S = 512 # typical length of sequence in application
        self.input = torch.randn(S, S, self.input_feat, requires_grad=False)
    
    def build_reference_net(self):
        if self.bias == 0:
            self.net = nn.Linear(self.input_feat, self.output_feat, bias=False)
        else:
            self.net = nn.Linear(self.input_feat, self.output_feat, bias=True)

    def reference_forward(self):
        return self.net(self.input)
    
    def hikpex_opti_forward(self):
        return self.net(self.input)

    def replace_model_with_hikpex_fwd(self):
        nn.Linear.forward = Hikp_Linear
    
    def replace_model_with_init_fwd(self):
        nn.Linear.forward = self.ref
    
    def get_para_with_str(self):
        paras = re.split("[=:,%\[\]\s]+", self.para_str)
        assert len(paras) == 3
        self.input_feat = int(paras[0])
        self.output_feat = int(paras[1])
        self.bias = int(paras[2])

    def performance_test(self):
        forward1 = 0
        forward2 = 0
        retry = 0
        validate = 0

        self.reference_forward()
        for _ in range(self.iteration):
            start = time.time()
            Y1 = self.reference_forward()
            forward1 += time.time() - start

        reference_time = forward1 * 1e3 / self.iteration

        self.replace_model_with_hikpex_fwd()
        while validate < self.iteration:
            start = time.time()
            self.hikpex_opti_forward()
            interval = time.time() - start
            singe_time = interval * 1e3
            if singe_time >= reference_time / 10:
                time.sleep(0.5)
                retry += 1
                if retry > 10:
                    print("Forward pass time (PyTorch layer): {:.3f} ms | Hikp Forward pass time (Optimized layer): {:.3f} ms".format(reference_time, singe_time))
                    print("Improve by %f times"%(reference_time / singe_time))
                    return
                continue
            validate += 1
            forward2 += interval

        opt_time = forward2 * 1e3 / self.iteration
        print("Forward pass time (PyTorch layer): {:.3f} ms | Hikp Forward pass time (Optimized layer): {:.3f} ms"
            .format(reference_time, opt_time))
        print("Improve by %f times"%(reference_time / opt_time))
        self.replace_model_with_init_fwd()

def get_test_obj(para_str, test_item, iteration):
    return LinearTest(para_str, iteration, test_item)
