import numpy as np
from scipy.stats import beta
import time

class InitialBelief:
    """
    Generate initial belief states X0 for all agents.
    Each agent has an m-dimensional belief vector in [-1,1].
    """

    def __init__(self, n=100, m=3, mode="uniform", beta_params=(2,5), seed=None):
        if seed is None:
            seed = int(time.time() * 1e6) % (2**32)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.m = m
        self.mode = mode
        self.beta_params = beta_params
        self.X0 = None

    def generate(self):
        rng = self.rng
        a, b = self.beta_params

        if self.mode == "uniform":
            X0 = rng.uniform(-1, 1, size=(self.n, self.m))
        elif self.mode == "normal":
            X0 = np.clip(rng.normal(0, 0.5, size=(self.n, self.m)), -1, 1)
        elif self.mode == "beta":
            Y = beta.rvs(a, b, size=(self.n, self.m), random_state=rng)
            X0 = 2 * Y - 1
        elif self.mode == "polarized":
            mask = rng.random(size=(self.n, self.m)) < 0.5
            X0 = np.where(mask, rng.uniform(0.5, 1.0, size=(self.n, self.m)),
                                rng.uniform(-1.0, -0.5, size=(self.n, self.m)))
        else:
            raise ValueError("mode must be one of ['uniform', 'normal', 'beta', 'polarized']")

        self.X0 = X0
        return self

    def describe(self):
        if self.X0 is None:
            print("⚠️ No initial beliefs generated.")
            return
        print("\n=== Initial Belief Summary ===")
        print(f"Agents (n): {self.n}, Topics (m): {self.m}")
        print(f"Mode: {self.mode}")
        if self.mode == "beta":
            print(f"Beta(α,β): {self.beta_params}")
        print(f"Mean: {np.mean(self.X0):.3f}, Std: {np.std(self.X0):.3f}")
        print(f"Min: {np.min(self.X0):.2f}, Max: {np.max(self.X0):.2f}")
        print(f"Seed: {self.seed}")
        print("==============================\n")
