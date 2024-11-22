import numpy as np

class ReLULayer:
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)
    
    def backward(self, d_out):
        d_input = d_out.copy()
        d_input[self.last_input <= 0] = 0
        return d_input
