import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    # For numerical stability
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred))

def categorical_cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true  # Simplified gradient when using softmax activation