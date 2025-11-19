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
        """Generate baseline logic matrix C_base and heterogeneous C_i list."""
        rng = self.rng

        # 1) Beta 参数
        if self.random_beta:
            a = rng.uniform(0.5, 5)
            b = rng.uniform(0.5, 5)
            self.beta_params = (a, b)
        a, b = self.beta_params


        C_base = np.zeros((self.m, self.m))

        for i in range(self.m):
            C_base[i, i] = 1.0

            if i > 0:
                raw = []
                for j in range(i):
                    val = beta.rvs(a, b, random_state=rng)
                    sign = rng.choice([-1, 1])
                    raw.append(sign * val)

                C_base[i, :i] = raw

                row_abs_sum = np.sum(np.abs(C_base[i, :i+1]))
                C_base[i, :i+1] = C_base[i, :i+1] / row_abs_sum

        C_list = []
        for _ in range(self.n):
            C_i = C_base + self.heterogeneity * rng.normal(0, 0.02, size=(self.m, self.m))
            C_i = np.tril(C_i)
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
    
        
