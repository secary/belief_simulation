import numpy as np
from scipy.stats import beta
import json, os, time

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
        heterogeneity=0.1,    # 个体间差异强度
        sparsity=0.5,         # 非零元素比例
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
        self.heterogeneity = heterogeneity
        self.sparsity = sparsity
        self.random_beta = random_beta

        self.C_base = None
        self.C_list = None

    # ----------------------------------------------------------
    def generate(self):
        """Generate baseline logic matrix C_base and heterogeneous C_i list."""
        rng = self.rng

        # 1️⃣ 确定 Beta 分布参数
        if self.random_beta:
            a = rng.uniform(0.5, 5)
            b = rng.uniform(0.5, 5)
            self.beta_params = (a, b)
        a, b = self.beta_params

        # 2️⃣ 生成基准逻辑矩阵 C_base
        C_base = np.eye(self.m)
        for i in range(self.m):
            for j in range(self.m):
                if i != j and rng.random() < self.sparsity:
                    val = beta.rvs(a, b, random_state=rng)   # Beta 分布抽样
                    sign = rng.choice([-1, 1])               # 随机正负符号
                    C_base[i, j] = sign * val * 0.5          # 控制幅度在 ±0.5

        # 行归一化，防止过大权重
        row_norm = np.sum(np.abs(C_base), axis=1, keepdims=True)
        row_norm[row_norm == 0] = 1
        C_base = C_base / row_norm

        # 3️⃣ 添加个体异质性
        C_list = []
        for _ in range(self.n):
            noise = self.heterogeneity * rng.uniform(-0.05, 0.05, size=(self.m, self.m))
            C_i = C_base + noise
            C_i = C_i / np.max(np.sum(np.abs(C_i), axis=1))
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
        print(f"Sparsity: {self.sparsity:.2f}")
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
        
