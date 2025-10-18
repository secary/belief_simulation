# torch_theorem2.py
# 2 agents (n=2), 2 topics (m=2)
# Case 1: non-competing dependencies -> nonzero consensus
# Case 2: competing dependencies -> consensus to 0 (for the competing topic)

import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.double)

def build_A_from_C_and_W_torch(C_list, W):
    """
    C_list: list of length n, each item is an (m x m) torch tensor (C_i)
    W: (n x n) nonnegative row-stochastic torch tensor
    return: A in (8), shape (nm x nm)
    """
    n = len(C_list)
    m = C_list[0].shape[0]

    # Γ_{kj} = diag(c_{kj,1}, ..., c_{kj,n})
    Gamma = [[None for _ in range(m)] for _ in range(m)]
    for k in range(m):
        for j in range(m):
            diag_entries = torch.tensor([C_list[i][k, j].item() for i in range(n)], dtype=torch.double)
            Gamma[k][j] = torch.diag(diag_entries)

    # Assemble block matrix A (m x m blocks, each n x n)
    A = torch.zeros(n*m, n*m, dtype=torch.double)
    for k in range(m):
        for j in range(m):
            block = Gamma[k][j] @ W
            A[k*n:(k+1)*n, j*n:(j+1)*n] = block
    return A

def iterate_y_torch(A, y0, T, clip_to_unit=True):
    """y(t+1) = A y(t)"""
    nm = y0.numel()
    Y_hist = torch.zeros(nm, T+1, dtype=torch.double)
    Y_hist[:, 0] = y0
    for t in range(T):
        y_next = A @ Y_hist[:, t]
        if clip_to_unit:
            y_next = torch.clamp(y_next, -1.0, 1.0)
        Y_hist[:, t+1] = y_next
    return Y_hist, Y_hist[:, -1]

def run_case(title, C_list, W, y0, T, ax):
    A = build_A_from_C_and_W_torch(C_list, W)
    Y_hist, y_lim = iterate_y_torch(A, y0, T, clip_to_unit=True)

    # plot: y = [y1(agents=1..n); y2(agents=1..n); ...]
    t = torch.arange(T+1).numpy()
    for r in range(Y_hist.shape[0]):
        ax.plot(t, Y_hist[r, :].numpy(), linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel('t')
    ax.set_ylabel('opinion')
    ax.set_ylim([-1.05, 1.05])
    ax.grid(True)

    # consensus estimates per topic (mean over agents at final time)
    n = W.shape[0]
    m = int(Y_hist.shape[0] / n)
    alphas = []
    for k in range(m):
        idx = slice(k*n, (k+1)*n)
        alphas.append(Y_hist[idx, -1].mean().item())

    return alphas

if __name__ == "__main__":
    # common settings
    n, m = 2, 2
    W = torch.tensor([[0.5, 0.5],
                      [0.5, 0.5]], dtype=torch.double)   # row-stochastic
    y0 = torch.tensor([0.8, -0.3,   # y1(0): agent1=0.8, agent2=-0.3
                       0.2,  0.6],  # y2(0): agent1=0.2, agent2=0.6
                      dtype=torch.double)
    T = 40

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # ---- Case 1: non-competing dependencies (no negative sign conflict)
    C1 = torch.tensor([[1.0, 0.0],
                       [0.5, 0.5]], dtype=torch.double)
    C2 = torch.tensor([[1.0, 0.0],
                       [0.5, 0.5]], dtype=torch.double)
    alphas1 = run_case('Non-competing dependencies',
                       [C1, C2], W, y0, T, axes[0])
    print(f'Case 1 consensus: alpha_1 ≈ {alphas1[0]:.4f}, alpha_2 ≈ {alphas1[1]:.4f}')

    # ---- Case 2: competing interdependencies (opposite signs on 1->2)
    C1c = torch.tensor([[1.0, 0.0],
                        [0.5, 0.5]], dtype=torch.double)
    C2c = torch.tensor([[1.0, 0.0],
                        [-0.5, 0.5]], dtype=torch.double)
    alphas2 = run_case('Competing interdependencies',
                       [C1c, C2c], W, y0, T, axes[1])
    print(f'Case 2 consensus: alpha_1 ≈ {alphas2[0]:.4f}, alpha_2 ≈ {alphas2[1]:.4f}')

    # legend (same order as MATLAB demo)
    axes[0].legend(['y1^(1)','y1^(2)','y2^(1)','y2^(2)'], loc='best')
    axes[1].legend(['y1^(1)','y1^(2)','y2^(1)','y2^(2)'], loc='best')
    plt.show()
