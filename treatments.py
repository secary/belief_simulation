from social_network import SocialNetwork
from logic_matrix import LogicMatrix
from init_belief import InitialBelief
from functions import simulate, inject_kol

import numpy as np, matplotlib.pyplot as plt


def alpha_controll(
    AGENT_NUM: int,
    TOPIC_NUM: int,
    T: int,
    SEED:int = 1942340,
    alpha_list: list = [0.3, 0.5, 0.7, 0.9]

):
    
    W = SocialNetwork(
        n=AGENT_NUM,
        random_beta=True,
        seed=SEED
    ).generate()

    W_matrix = W.W
    
    C = LogicMatrix(
        n=AGENT_NUM,
        m=TOPIC_NUM,
        random_beta=True,
        seed=SEED
    ).generate()
    C_base = C.C_base
    C_tensor = np.array(C.C_list)   # shape: (n, m, m)


    X0 = InitialBelief(
        n=AGENT_NUM,
        m=TOPIC_NUM,
        mode="uniform",
        seed=SEED
    ).generate()

    X0_vector = X0.X0

    
    traj = simulate(W_matrix, C_tensor, X0_vector, T=T)

    traj_kol_list = []
    

    for alpha in alpha_list:
        C_kol = inject_kol(C_tensor, C_base, alpha=alpha, kol_index=0)
        traj_kol = simulate(W_matrix, C_kol, X0_vector, T=T)
        traj_kol_list.append(traj_kol)



    n = len(alpha_list)

    fig, axes = plt.subplots(2, n+1, figsize=(5*(n+1), 8), sharex=True)

    # =========================
    # 第一行：trajectory
    # =========================

    for i in range(traj.shape[1]):
        for k in range(TOPIC_NUM):
            axes[0, 0].plot(traj[:, i, k], alpha=0.3)

    axes[0, 0].set_title("Baseline")
    axes[0, 0].set_ylabel("Trajectory")

    for idx, (traj_kol, alpha) in enumerate(zip(traj_kol_list, alpha_list)):
        ax = axes[0, idx+1]

        for i in range(traj_kol.shape[1]):
            for k in range(TOPIC_NUM):
                ax.plot(traj_kol[:, i, k], alpha=0.3)

        ax.set_title(f"alpha = {alpha}")

    # =========================
    # 第二行：mean
    # =========================

    # baseline mean
    mean_base = traj.mean(axis=1)
    for k in range(TOPIC_NUM):
        axes[1, 0].plot(mean_base[:, k], linewidth=3, label=f"Topic {k}")

    axes[1, 0].set_ylabel("Mean")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].legend()

    # KOL mean
    for idx, (traj_kol, alpha) in enumerate(zip(traj_kol_list, alpha_list)):
        ax = axes[1, idx+1]

        mean = traj_kol.mean(axis=1)
        for k in range(TOPIC_NUM):
            ax.plot(mean[:, k], linewidth=3, label=f"Topic {k}")

        ax.set_xlabel("Time")
        ax.legend()
        
    # =========================
    # 🔥 统一纵轴范围
    # =========================
    for ax in axes.flatten():
        ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.show()

    
def topology_control(
AGENT_NUM: int,
TOPIC_NUM: int,
T: int,
alpha: float = 0.7,
SEED: int = 1942340,
models: list = ["ER", "WS", "BA", "RR"]
):

    # =========================
    # 1️⃣ 固定 C 和 X0（关键！）
    # =========================
    C = LogicMatrix(
        n=AGENT_NUM,
        m=TOPIC_NUM,
        random_beta=True,
        seed=SEED
    ).generate()

    C_base = C.C_base
    C_tensor = np.array(C.C_list)

    X0 = InitialBelief(
        n=AGENT_NUM,
        m=TOPIC_NUM,
        mode="uniform",
        seed=SEED
    ).generate()

    X0_vector = X0.X0

    # =========================
    # 2️⃣ baseline & KOL 结果
    # =========================
    traj_base_list = []
    traj_kol_list = []

    # =========================
    # 3️⃣ 遍历不同拓扑
    # =========================
    # KOL
    C_kol = inject_kol(C_tensor, C_base, alpha=alpha, kol_index=0)
    for model in models:

        W = SocialNetwork(
            n=AGENT_NUM,
            model=model,          # 🔥 固定拓扑类型
            random_beta=True,
            seed=SEED
        ).generate()

        W_matrix = W.W

        # baseline
        traj_base = simulate(W_matrix, C_tensor, X0_vector, T=T)

  
        traj_kol = simulate(W_matrix, C_kol, X0_vector, T=T)

        traj_base_list.append(traj_base)
        traj_kol_list.append(traj_kol)

    n = len(models)

    fig, axes = plt.subplots(2, n, figsize=(5*n, 8), sharex=True)

    # =========================
    # 第一行：baseline
    # =========================
    for idx, traj in enumerate(traj_base_list):
        ax = axes[0, idx]

        for i in range(traj.shape[1]):
            for k in range(TOPIC_NUM):
                ax.plot(traj[:, i, k], alpha=0.3)

        ax.set_title(f"{models[idx]} (Baseline)")
        ax.set_ylim(-1, 1)

    axes[0, 0].set_ylabel("Baseline")

    # =========================
    # 第二行：KOL trajectory（🔥改这里）
    # =========================
    for idx, traj in enumerate(traj_kol_list):
        ax = axes[1, idx]

        for i in range(traj.shape[1]):
            for k in range(TOPIC_NUM):
                ax.plot(traj[:, i, k], alpha=0.3)

        ax.set_title(f"{models[idx]} (KOL)")
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Time")

    axes[1, 0].set_ylabel("KOL")

    # =========================
    # 统一样式
    # =========================
    for ax in axes.flatten():
        ax.axhline(0, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
                
if __name__ == "__main__":
    alpha_controll(
    AGENT_NUM=30,
    TOPIC_NUM=3,
    T=20,
    SEED=1942340,
    alpha_list = [0.3, 0.5, 0.7, 0.9]
    )
    
    topology_control(
    AGENT_NUM=30,
    TOPIC_NUM=3,
    T=20,
    alpha=0.7,
    SEED=1942340,
    models=["ER", "WS", "BA", "RR"]
    )