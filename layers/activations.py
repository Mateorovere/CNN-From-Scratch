import numpy as np
from .layer import Layer
from .activation import Activation

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)

class Softmax(Layer):
    def forward(self, input):
        self.input = input
        self.output = softmax(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Gradient is simplified when combined with cross-entropy loss
        return output_gradient
