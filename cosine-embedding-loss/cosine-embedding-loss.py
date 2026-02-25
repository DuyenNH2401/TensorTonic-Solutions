def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    import math
    dot_prod = sum(a*b for a, b in zip(x1, x2))
    norm = lambda x: sum(i*i for i in x)

    norm1 = math.sqrt(norm(x1))
    norm2 = math.sqrt(norm(x2))

    cosphi = dot_prod / (norm1 * norm2)

    if label == 1:
        return 1 - cosphi
    else:
        return max(0, cosphi - margin)