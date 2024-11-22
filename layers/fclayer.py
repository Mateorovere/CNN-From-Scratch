import numpy as np

class FullyConnectedLayer:
    def __init__(self, input_len, output_len):
        # Initialize weights and biases
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.biases = np.zeros(output_len)
    
    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, d_out, learning_rate):
        """
        Performs a backward pass of the fully connected layer.
        """
        # Gradient of loss with respect to weights and biases
        d_weights = np.dot(self.last_input[:, np.newaxis], d_out[np.newaxis, :])
        d_biases = d_out

        # Gradient of loss with respect to input
        d_input = np.dot(d_out, self.weights.T)
        d_input = d_input.reshape(self.last_input_shape)

        # Update weights and biases
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input