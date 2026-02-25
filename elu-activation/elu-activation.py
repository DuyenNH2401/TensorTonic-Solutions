from math import exp
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    res = []
    elu = lambda x: x if x > 0 else alpha * (exp(x)-1)
    for i in x:
        res.append(elu(i))

    return res