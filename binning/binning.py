def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    from math import floor
    
    max_val = max(values)
    min_val = min(values)
    ran = max_val-min_val
    
    res = [0 for _ in range(len(values))]
    w = ran / num_bins
    
    if w == 0:
        return res

    for i in range(len(values)):
        res[i] = min(floor((values[i]-min_val) / w) , num_bins-1)

    return res
    