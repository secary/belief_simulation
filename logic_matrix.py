import numpy as np
from scipy.stats import beta
import json, os, time
import matplotlib.pyplot as plt
import seaborn as sns

class LogicMatrix:
    """
    Class to generate individual logic matrices C_i using Beta-distributed
    weights for internal topic relationships.

    Each C_i is an m×m matrix describing how topics influence each other
    for a given agent.
    """

    def __init__(
        self,
        n=100,                # 个体数量
        m=3,                  # 主题数量
        beta_params=(2, 5),   # Beta 分布参数 (α, β)
        random_beta=False,    # 是否随机 Beta 参数
        seed=None
    ):
        if seed is None:
            seed = int(time.time() * 1e6) % (2**32)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.n = n
        self.m = m
        self.beta_params = beta_params
        self.heterogeneity = float(np.random.default_rng(seed).uniform(0.05, 0.2))
        self.random_beta = random_beta
        self.C_base = None
        self.C_list = None

    # ----------------------------------------------------------
    def generate(self):
        """Generate baseline logic matrix C_base and heterogeneous C_i list."""
        rng = self.rng

        # 1) Beta 参数
        if self.random_beta:
            a = rng.uniform(0.5, 5)
            b = rng.uniform(0.5, 5)
            self.beta_params = (a, b)
        a, b = self.beta_params

        # 2) 生成 C_base（下三角 + 允许正负号 + 绝对值行随机）
        C_base = np.zeros((self.m, self.m))

        for i in range(self.m):
            # 对角线（始终保留 inertia）
            C_base[i, i] = 1.0

            if i > 0:
                # 下三角随机（Beta × ±1）
                raw = []
                for j in range(i):
                    val = beta.rvs(a, b, random_state=rng)
                    sign = rng.choice([-1, 1])
                    raw.append(sign * val)

                # 拼回行向量
                C_base[i, :i] = raw

                # ======== 🔥 行绝对值归一化 (关键!) ========
                row_abs_sum = np.sum(np.abs(C_base[i, :i+1]))
                C_base[i, :i+1] = C_base[i, :i+1] / row_abs_sum

        # 3) 个体异质性（并保持 absolute row-stochastic）
        C_list = []
        for _ in range(self.n):
            C_i = C_base + self.heterogeneity * rng.normal(0, 0.02, size=(self.m, self.m))

            # 保持下三角
            C_i = np.tril(C_i)

            # 保持每行 absolute row stochastic
            for i in range(self.m):
                row_abs_sum = np.sum(np.abs(C_i[i, :i+1]))
                if row_abs_sum == 0:
                    C_i[i, i] = 1.0
                else:
                    C_i[i, :i+1] /= row_abs_sum

            C_list.append(C_i)

        self.C_base = C_base
        self.C_list = C_list
        return self

    # ----------------------------------------------------------
    def describe(self):
        """Summarize the generated logic matrices."""
        if self.C_base is None:
            print("⚠️ No logic matrices generated. Call .generate() first.")
            return

        abs_vals = np.abs(self.C_base)
        non_zero_ratio = np.mean(abs_vals > 1e-6)
        avg_offdiag = np.mean(np.abs(self.C_base - np.eye(self.m)))

        print("\n=== Logic Matrix Summary ===")
        print(f"Agents (n): {self.n}")
        print(f"Topics (m): {self.m}")
        print(f"Beta(α,β): {tuple(round(x,3) for x in self.beta_params)}")
        print(f"Heterogeneity: {self.heterogeneity:.2f}")
        print(f"Non-zero ratio: {non_zero_ratio:.3f}")
        print(f"Average off-diagonal weight: {avg_offdiag:.3f}")
        print(f"Seed: {self.seed}")
        print("==============================\n")

    # ----------------------------------------------------------
    def export(self, prefix="logic_matrix", directory="./"):
        """Export C_base and generation parameters."""
        if self.C_base is None:
            print("⚠️ Please call .generate() first.")
            return

        os.makedirs(directory, exist_ok=True)

        # 导出 C_base 为 CSV
        cbase_path = os.path.join(directory, f"{prefix}_base.csv")
        np.savetxt(cbase_path, self.C_base, delimiter=",", fmt="%.6f")
        print(f"✅ Saved baseline C_base to: {cbase_path}")

        # 导出元信息 JSON
        info = {
            "n": self.n,
            "m": self.m,
            "beta_params": tuple(round(x,3) for x in self.beta_params),
            "heterogeneity": self.heterogeneity,
            "sparsity": self.sparsity,
            "random_beta": self.random_beta,
            "seed": int(self.seed)
        }
        
        return info
    

    def visualize(self, show_individual=False):
        """
        Visualize:
        1) Heatmap of C_base
        2) Histogram of non-zero off-diagonal entries
        3) (Optional) Histogram for all C_i

        Parameters
        ----------
        show_individual : bool
            If True, also show histograms for all C_i combined.
        """
        if self.C_base is None:
            raise ValueError("Please call .generate() before visualize().")

        C_base = self.C_base
        abs_vals = np.abs(C_base)
        nonzero_vals = abs_vals[np.tril(np.ones_like(C_base), k=-1) == 1]  # 仅下三角非零

        # -----------------------------
        # 1️⃣ Heatmap of C_base
        # -----------------------------
        plt.figure(figsize=(6, 4))
        sns.heatmap(C_base, annot=False, cmap="RdBu", center=0)
        plt.title("Heatmap of C_base (Lower Triangular Structure)")
        plt.xlabel("Topic j")
        plt.ylabel("Topic i")
        plt.tight_layout()
        plt.show()

        # -----------------------------
        # 2️⃣ Histogram of non-zero weights
        # -----------------------------
        plt.figure(figsize=(6, 4))
        plt.hist(nonzero_vals.flatten(), bins=20, color='steelblue', alpha=0.8)
        plt.title("Histogram of |C_base| Off-diagonal Weights")
        plt.xlabel("Absolute Weight")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # -----------------------------
        # 3️⃣ All C_i combined (optional)
        # -----------------------------
        if show_individual:
            all_vals = []
            for C_i in self.C_list:
                Ci_abs = np.abs(C_i)
                Ci_vals = Ci_abs[np.tril(np.ones_like(C_i), k=-1) == 1]
                all_vals.extend(Ci_vals.flatten())

            plt.figure(figsize=(6, 4))
            plt.hist(all_vals, bins=20, color='orange', alpha=0.75)
            plt.title("Histogram of All C_i Off-diagonal Weights")
            plt.xlabel("Absolute Weight")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

        
