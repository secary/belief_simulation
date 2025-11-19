import numpy as np
import time
from scipy.stats import beta
import matplotlib.pyplot as plt

class InitialBelief:
    """
    Generate initial belief states X0 for all agents.
    Supports independent per-agent distributions:
        - uniform: each agent has its own (low_i, high_i) within [-1,1]
        - beta: each agent has its own (a_i, b_i), mapped to [-1,1]
    """

    def __init__(
        self,
        n=100,
        m=3,
        mode="uniform",          # "uniform" or "beta"
        seed=None
    ):
        if seed is None:
            seed = int(time.time() * 1e6) % (2**32)

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.n = n
        self.m = m
        self.mode = mode

        self.X0 = None

    def generate(self):
        rng = self.rng

        X0 = np.zeros((self.n, self.m))
        if self.mode == "uniform":
            for i in range(self.n):
                low = rng.uniform(-1, 1)
                high = rng.uniform(low, 1)
                X0[i] = rng.uniform(low, high, size=self.m)

        elif self.mode == "beta":
            for i in range(self.n):
                a_i = rng.uniform(1, 5)
                b_i = rng.uniform(1, 5)

                y = beta.rvs(a_i, b_i, size=self.m, random_state=rng)

                X0[i] = 2 * y - 1

        else:
            raise ValueError("mode must be 'uniform' or 'beta'")

        self.X0 = X0
        return self
    

    def plot_hist(self, bins=30):
        if self.X0 is None:
            raise ValueError("Please call .generate() first.")

        plt.figure(figsize=(8, 4))
        plt.hist(self.X0.flatten(), bins=bins,
                 color="steelblue", alpha=0.8, edgecolor="black")
        plt.title(f"Histogram of Initial Beliefs")
        plt.xlabel("Belief value")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
