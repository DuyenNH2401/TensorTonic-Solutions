import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype=np.float64)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    
    return x * sigmoid(x)