import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
    
    def iterate_regions(self, image):
        """
        Generates all possible regions of the input image that the filters can cover.
        """
        h, w = image.shape
        filter_size = self.filters.shape[1]
        for i in range(h - filter_size + 1):
            for j in range(w - filter_size + 1):
                image_region = image[i:(i + filter_size), j:(j + filter_size)]
                yield image_region, i, j
    
    def forward(self, input):
        """
        Performs a forward pass of the convolution layer.
        """
        self.last_input = input
        h, w = input.shape
        filter_size = self.filters.shape[1]
        output = np.zeros((h - filter_size + 1, w - filter_size + 1, self.num_filters))
        
        for image_region, i, j in self.iterate_regions(input):
            for f in range(self.num_filters):
                output[i, j, f] = np.sum(image_region * self.filters[f])
        return output
    
    def backward(self, d_out, learning_rate):
        """
        Performs a backward pass of the convolution layer.
        """
        filter_size = self.filters.shape[1]
        d_filters = np.zeros(self.filters.shape)
        d_input = np.zeros(self.last_input.shape)

        for image_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # Gradient of the loss w.r.t. the filter
                d_filters[f] += d_out[i, j, f] * image_region
                # Gradient of the loss w.r.t. the input
                d_input[i:(i + filter_size), j:(j + filter_size)] += d_out[i, j, f] * self.filters[f]

        # Update the filters using gradient descent
        self.filters -= learning_rate * d_filters

        return d_input

