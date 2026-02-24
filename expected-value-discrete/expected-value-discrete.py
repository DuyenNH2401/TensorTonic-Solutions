import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x)
    p = np.array(p)
    
    if not np.allclose(np.sum(p), 1, rtol= 10e-6):
        raise ValueError()
        
    return np.sum(x*p)