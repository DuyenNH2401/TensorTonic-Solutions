def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    res = []

    for j in range(len(values)):
        res.append([values[j] ** i for i in range(degree+1)])
    return res