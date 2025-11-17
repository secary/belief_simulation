import numpy as np
from scipy.stats import beta
import time
import matplotlib.pyplot as plt

class InitialBelief:
    """
    Generate initial belief states X0 for all agents.
    Modes:
        - uniform        : U[-1, 1]
        - clustered      : each cluster uses its own uniform(a,b)
        - agent_uniform  : each agent uses its own uniform(a,b)
    """

    def __init__(self, n=100, m=3, mode="uniform",
                 clusters=None,              # for clustered mode
                 agent_range="random",       # for agent_uniform
                 seed=None):
        if seed is None:
            seed = int(time.time() * 1e6) % (2**32)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.n = n
        self.m = m
        self.mode = mode

        # cluster configs: list of tuples [(size, low, high), ...]
        self.clusters = clusters

        # agent_uniform configs:
        # "random" or list of (low_i, high_i)
        self.agent_range = agent_range

        self.X0 = None

    # ----------------------------------------------------------
    def generate(self):
        rng = self.rng

        # -------------------------
        # 1) simple uniform
        # -------------------------
        if self.mode == "uniform":
            X0 = rng.uniform(-1, 1, size=(self.n, self.m))

        # -------------------------
        # 2) clustered uniform
        #    clusters = [(size, low, high), ...]
        # -------------------------
        elif self.mode == "clustered":
            if self.clusters is None:
                raise ValueError("For 'clustered' mode, provide clusters=[(size,low,high), ...].")

            X0 = np.zeros((self.n, self.m))
            idx = 0
            for size, low, high in self.clusters:
                X0[idx:idx+size] = rng.uniform(low, high, size=(size, self.m))
                idx += size

        # -------------------------
        # 3) agent-uniform
        #    each agent has its own uniform(low_i, high_i)
        # -------------------------
        elif self.mode == "agent_uniform":

            X0 = np.zeros((self.n, self.m))

            # A) auto-generate random ranges for each agent
            if self.agent_range == "random":
                for i in range(self.n):
                    low = rng.uniform(-1, 1)
                    high = rng.uniform(low, 1)
                    X0[i] = rng.uniform(low, high, size=self.m)

            # B) user-provided ranges
            else:
                if len(self.agent_range) != self.n:
                    raise ValueError("agent_range must have length n.")
                for i in range(self.n):
                    low, high = self.agent_range[i]
                    X0[i] = rng.uniform(low, high, size=self.m)

        else:
            raise ValueError("mode must be one of ['uniform', 'clustered', 'agent_uniform']")

        self.X0 = X0
        return self

    # ----------------------------------------------------------
    def plot_hist(self, bins=30):
        """Plot histogram of all initial belief values."""
        if self.X0 is None:
            raise ValueError("Please call .generate() first.")

        plt.figure(figsize=(8, 4))
        plt.hist(self.X0.flatten(), bins=bins, color="steelblue", alpha=0.8, edgecolor="black")
        plt.title("Histogram of Initial Beliefs")
        plt.xlabel("Belief value")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
