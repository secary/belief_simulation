import numpy as np
import matplotlib.pyplot as plt

def simulate(W, C, X0, T=30):
    """
    Simulate belief dynamics following:
        x_i(t+1) = C_i * (sum_j w_ij x_j(t))
    """
    X = X0.copy()
    traj = [X]

    for t in range(T):
        influence = W @ X              # shape (n, m)

        X_next = np.einsum("nij,nj->ni", C, influence)

        traj.append(X_next)
        X = X_next

    return np.array(traj)
    

    
    
def inject_kol(C_list, C_base, alpha=0.3, kol_index=0):
    C_new = C_list.copy()
    n, m, _ = C_new.shape

    for idx in range(n):
        if idx == kol_index:
            C_new[idx] = C_base
        else:
            C_new[idx] = (1 - alpha) * C_new[idx] + alpha * C_base

            # 归一化
            for i in range(m):
                row_abs_sum = np.sum(np.abs(C_new[idx][i, :i+1]))
                if row_abs_sum == 0:
                    C_new[idx][i, i] = 1.0
                else:
                    C_new[idx][i, :i+1] /= row_abs_sum

    return C_new

