import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    dot_prod = np.sum(a * b)
    norm = lambda x: np.sqrt(np.sum([i**2 for i in x]))
    n_a = norm(a)
    n_b = norm(b)
    if n_a == 0 or n_b == 0: return 0
    
    return dot_prod / (n_a * n_b)