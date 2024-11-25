import numpy as np
from scipy import signal
from .layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        self.input_shape = input_shape
        self.input_depth, self.input_height, self.input_width = input_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.output_height = self.input_height - kernel_size + 1
        self.output_width = self.input_width - kernel_size + 1
        self.output_shape = (depth, self.output_height, self.output_width)
        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2.0 / np.prod(self.kernels_shape[1:]))
        self.biases = np.zeros(self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], mode='valid')
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] += signal.correlate2d(self.input[j], output_gradient[i], mode='valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], mode='full')
        # Update parameters
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient