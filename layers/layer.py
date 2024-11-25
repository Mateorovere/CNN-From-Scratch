class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass  # To be implemented in subclasses

    def backward(self, output_gradient, learning_rate):
        pass  # To be implemented in subclasses