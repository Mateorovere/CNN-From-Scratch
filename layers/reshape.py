import numpy as np
from .layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input = input
        return input.reshape(self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)