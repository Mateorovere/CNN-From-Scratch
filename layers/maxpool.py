import numpy as np

class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def iterate_regions(self, image):
        h, w, num_filters = image.shape
        pool_size = self.pool_size
        for i in range(0, h, pool_size):
            for j in range(0, w, pool_size):
                image_region = image[i:i+pool_size, j:j+pool_size]
                yield image_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        pool_size = self.pool_size
        output = np.zeros((h // pool_size, w // pool_size, num_filters))

        for image_region, i, j in self.iterate_regions(input):
            output[i // pool_size, j // pool_size] = np.amax(image_region, axis=(0, 1))
        
        return output

    def backward(self, d_out):
        """
        Backpropagation for max pooling layer.
        d_out is the gradient of the loss with respect to the output of this layer.
        """
        d_input = np.zeros_like(self.last_input)

        for image_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = image_region.shape
            max_vals = np.amax(image_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # Only propagate gradients where the max value was selected
                        if image_region[i2, j2, f2] == max_vals[f2]:
                            d_input[i + i2, j + j2, f2] += d_out[i // self.pool_size, j // self.pool_size, f2]
        
        return d_input
