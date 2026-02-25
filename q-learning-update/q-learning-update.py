import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q = np.asarray(Q, dtype=float)
    Q_new = Q.copy()
    
    target = r + gamma * np.max(Q[s_next, :])
    td_error =  target - Q[s,a]

    Q_new[s, a] = Q[s, a] + alpha * td_error

    return Q_new