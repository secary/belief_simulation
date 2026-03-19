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
        n=100,               
        m=3,                 
        beta_params=(2, 5),   
        random_beta=False,    
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
        """Generate logic matrices using Eq.(19) with Beta distribution."""
        rng = self.rng

        if self.random_beta:
            a = rng.uniform(0.5, 5)
            b = rng.uniform(0.5, 5)
            self.beta_params = (a, b)

        a, b = self.beta_params

        C_list = []

        # ===== 可控符号概率（你可以调）=====
        def project_pair(x, y):
            s = abs(x) + abs(y)
            if s > 1:
                x = x / s
                y = y / s
            return x, y
        
        p_sign = {
            "eta": 0.7,
            "beta": 0.5,
            "mu": 0.3,
            "delta": 0.5
        }

        def sample_param(name):
            val = beta.rvs(a, b, random_state=rng)
            sign = 1 if rng.random() < p_sign[name] else -1
            return sign * val

        for _ in range(self.n):

            # ===== 1️⃣ 采样参数 =====
            eta   = sample_param("eta")
            beta_i = sample_param("beta")
            eta, beta_i = project_pair(eta, beta_i)

            mu    = sample_param("mu")
            delta = sample_param("delta")
            mu, delta = project_pair(mu, delta)
            
            # ===== 2️⃣ 构造 C_i（核心！）=====
            C_i = np.zeros((self.m, self.m))

            # 第一行（root）
            C_i[0, 0] = 1.0

            if self.m >= 2:
                C_i[1, 0] = eta
                C_i[1, 1] = 1 - (abs(eta) + abs(beta_i))
                C_i[1, 2] = -beta_i if self.m >= 3 else 0

            if self.m >= 3:
                C_i[2, 0] = mu
                C_i[2, 1] = -delta
                C_i[2, 2] = 1 - (abs(mu) + abs(delta))

            # ===== 3️⃣ 高维扩展（如果 m > 3）=====
            for i in range(3, self.m):
                row_vals = []

                for j in range(i):
                    val = beta.rvs(a, b, random_state=rng)
                    sign = rng.choice([-1, 1])
                    row_vals.append(sign * val)

                row_vals = np.array(row_vals)

                row_abs_sum = np.sum(np.abs(row_vals))
                if row_abs_sum > 0:
                    row_vals /= row_abs_sum

                C_i[i, :i] = row_vals
                C_i[i, i] = 1 - np.sum(np.abs(row_vals))

            C_list.append(C_i)

        self.C_list = C_list
        self.C_base = C_list[0]

        return self


    def describe(self):
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
        if self.C_base is None:
            print("⚠️ Please call .generate() first.")
            return

        os.makedirs(directory, exist_ok=True)

        cbase_path = os.path.join(directory, f"{prefix}_base.csv")
        np.savetxt(cbase_path, self.C_base, delimiter=",", fmt="%.6f")
        print(f"✅ Saved baseline C_base to: {cbase_path}")

        info = {
            "n": self.n,
            "m": self.m,
            "beta_params": tuple(round(x,3) for x in self.beta_params),
            "heterogeneity": self.heterogeneity,
            "random_beta": self.random_beta,
            "seed": int(self.seed)
        }
        
        return info
    
        
