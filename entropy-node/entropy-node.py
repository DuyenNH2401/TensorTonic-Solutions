import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)
    values, counts = np.unique(y, return_counts=True)

    res = 0.0
    p = counts / len(y)
    for i in range(len(p)):
        if p[i] == 0:
            continue
        res += -p[i] * np.log2(p[i])

    return res
        