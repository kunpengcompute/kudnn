import time
from abc import ABC, abstractmethod
import torch

torch.set_default_dtype(torch.float32)
torch.manual_seed(11)
class OperatorTest:
    def __init__(self, para_str, N, item):
        self.iteration = N
        self.item = item
        self.para_str = para_str

    @abstractmethod
    def gen_input_data(self):
        pass

    @abstractmethod
    def build_reference_net(self):
        pass

    @abstractmethod
    def reference_forward(self):
        pass

    @abstractmethod
    def hikpex_opti_forward(self):
        pass

    @abstractmethod
    def replace_model_with_hikpex_fwd(self):
        pass

    @abstractmethod
    def replace_model_with_init_fwd(self):
        pass

    @abstractmethod
    def get_para_with_str(self):
        pass

    def judge_results_close(self, ref, result):
        diff = (ref.max() - ref.min()).item()
        r = 1 if diff == 0 else diff
        return ((torch.abs(ref - result) / r < 1e-3).sum() == ref.numel()).item()

    def function_test(self):
        Y1 = self.reference_forward()

        self.replace_model_with_hikpex_fwd()
        Y2 = self.hikpex_opti_forward()

        test_result = self.judge_results_close(Y1, Y2)
        print("Forward pass check: ", test_result)
        if test_result == False:
            print("ref result:", Y1)
            print("our result:", Y2)
        self.replace_model_with_init_fwd()
    
    def performance_test(self):
        forward1 = 0  # variables to store time values
        forward2 = 0

        self.reference_forward()
        for _ in range(self.iteration):  # MKLDNN PyTorch layer Forward and Backward pass timing
            start = time.time()
            Y1 = self.reference_forward()
            forward1 += time.time() - start

        self.replace_model_with_hikpex_fwd()
        self.hikpex_opti_forward()
        for _ in range(self.iteration):  # Optimized PyTorch layer Forward and Backward pass timing
            start = time.time()
            Y2 = self.hikpex_opti_forward()
            forward2 += time.time() - start

        print(
            "Forward pass time (PyTorch layer): {:.3f} ms | Hikp Forward pass time (Optimized layer): {:.3f} ms".format(
            forward1 * 1e3 / self.iteration, forward2 * 1e3 / self.iteration
            )
        )
        print("Improve by %f times"%(forward1 / forward2))
        self.replace_model_with_init_fwd()

    def test(self):
        self.get_para_with_str()
        self.gen_input_data()
        self.build_reference_net()
        if self.item != "performance":
            self.function_test()
        else:
            self.performance_test()
