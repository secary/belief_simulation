import numpy as np, networkx as nx, matplotlib.pyplot as plt
import time
from scipy.stats import beta

class SocialNetwork:
    def __init__(
        self,
        n=100,
        model=None,             # "ER", "WS", "BA", "RR" 或 None 随机选
        directed=True,
        random_params=True,     # 是否随机化模型参数
        random_beta=False,      # 是否随机化 Beta(α,β)
        beta_params=(2, 5),     # 固定 Beta 参数时使用
        seed=None               # 随机种子（None → 自动生成）
    ):
        if seed is None:
            seed = int(time.time() * 1e6) % (2**32)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 存储参数
        self.n = n
        self.model = model
        self.directed = directed
        self.random_params = random_params
        self.random_beta = random_beta
        self.beta_params = beta_params

        # 初始化结果属性
        self.W = None
        self.G = None
        self.summary = None   # 替代旧 info

    # --------------------------------------------
    def generate(self):
        """Generate the social influence matrix W and network G."""
        rng = self.rng
        models = ["ER", "WS", "BA", "RR"]

        # 选择模型
        if self.model is None:
            self.model = rng.choice(models)
        model = self.model.upper()

        # 随机 Beta 参数
        if self.random_beta:
            a = rng.uniform(0.5, 5)
            b = rng.uniform(0.5, 5)
            self.beta_params = (a, b)
        a, b = self.beta_params

        # 模型参数
        if self.random_params:
            er_p = rng.uniform(0.02, 0.1)
            ws_k = int(rng.integers(2, 8)) // 2 * 2
            ws_p = rng.uniform(0.01, 0.3)
            ba_m = int(rng.integers(1, 4))
            rr_d = int(rng.integers(3, 8))
            rr_d = min(rr_d, self.n - 1)
        else:
            er_p, ws_k, ws_p, ba_m, rr_d = 0.05, 4, 0.1, 2, 4

        # 生成拓扑
        if model == "ER":
            G = nx.gnp_random_graph(self.n, er_p, seed=self.seed, directed=self.directed)
        elif model == "WS":
            G = nx.watts_strogatz_graph(self.n, ws_k, ws_p, seed=self.seed)
        elif model == "BA":
            G = nx.barabasi_albert_graph(self.n, ba_m, seed=self.seed)
        elif model == "RR":
            G = nx.random_regular_graph(rr_d, self.n, seed=self.seed)
        else:
            raise ValueError("model must be one of ['ER','WS','BA','RR']")

        # 若需定向
        if self.directed and not G.is_directed():
            DG = nx.DiGraph()
            DG.add_nodes_from(G.nodes())
            for u, v in G.edges():
                DG.add_edge(u, v) if rng.random() < 0.5 else DG.add_edge(v, u)
            G = DG

        # 构造加权矩阵
        A = nx.to_numpy_array(G, dtype=float)
        W = np.zeros_like(A)
        mask = A > 0
        W[mask] = beta.rvs(a, b, size=mask.sum(), random_state=rng)

        # 行归一化（保证行随机）
        row_sum = W.sum(axis=1, keepdims=True)
        zero_rows = (row_sum[:, 0] == 0)
        W[zero_rows, zero_rows] = 1.0
        W = W / W.sum(axis=1, keepdims=True)

        # 存储结果
        self.W = W
        self.G = G
        self.summary = self._summarize(model, a, b, er_p, ws_k, ws_p, ba_m, rr_d)
        return self

    # --------------------------------------------
    def _summarize(self, model=None, a=None, b=None, er_p=None, ws_k=None, ws_p=None, ba_m=None, rr_d=None):
        """生成或更新网络的完整摘要信息"""
        summary = {}

        # 若已生成网络，则可补充动态特征
        if self.W is not None:
            eigvals = np.linalg.eigvals(self.W)
            mags = np.abs(eigvals)
            rho = np.max(mags)
            second = np.sort(mags)[-2] if len(mags) > 1 else None
            stability = (
                "Semi-stable (row-stochastic)" if abs(rho - 1) < 1e-6
                else "Asymptotically stable" if rho < 1
                else "Unstable"
            )
            density = np.mean(self.W > 0)
            avg_row_sum = self.W.sum(axis=1).mean()
            avg_deg = np.mean([d for _, d in self.G.degree()])
            clustering = nx.average_clustering(self.G.to_undirected())

            summary.update({
                "Density": round(float(density), 4),
                "Average row sum": round(float(avg_row_sum), 4),
                "Average degree": round(float(avg_deg), 4),
                "Average clustering": round(float(clustering), 4),
                "Spectral radius (ρ)": round(float(rho), 6),
                "Second largest eigenvalue": round(float(second), 6),
                "Stability": stability,
            })

        # 添加静态信息（模型、参数、种子等）
        summary.update({
            "Network model": model or self.model,
            "Beta(α,β)": (round(a, 3), round(b, 3)) if a and b else self.beta_params,
            "Nodes": self.n,
            "Directed": self.directed,
            "Seed": int(self.seed),
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        return summary


    # --------------------------------------------
    def describe(self):
        """Print a detailed summary of the generated social network."""
        if self.W is None:
            print("⚠️ No network generated yet. Please call .generate() first.")
            return

        print("\n=== Social Network Summary ===")
        print(f"Nodes (n): {self.n}")
        print(f"Directed: {self.directed}")
        print(f"Model: {self.summary['Network model']}")
        print(f"Beta(α,β): {self.summary['Beta(α,β)']}")
        print(f"Matrix shape: {self.W.shape}")

        # 稀疏度与行和
        density = np.mean(self.W > 0)
        avg_row_sum = self.W.sum(axis=1).mean()
        print(f"Density (non-zero ratio): {density:.3f}")
        print(f"Average row sum: {avg_row_sum:.3f}")

        # 平均度与聚类
        avg_deg = np.mean([d for _, d in self.G.degree()])
        clustering = nx.average_clustering(self.G.to_undirected())
        print(f"Average degree: {avg_deg:.3f}")
        print(f"Average clustering: {clustering:.3f}")

        # 稳定性分析
        eigvals = np.linalg.eigvals(self.W)
        mags = np.abs(eigvals)
        rho = np.max(mags)
        second = np.sort(mags)[-2] if len(mags) > 1 else None
        if abs(rho - 1) < 1e-6:
            stability = "Semi-stable (row-stochastic)"
        elif rho < 1:
            stability = "Asymptotically stable"
        else:
            stability = "Unstable"

        print(f"Spectral radius (ρ): {rho:.6f}")
        if second is not None:
            print(f"Second largest eigenvalue: {second:.6f}")
        print(f"Stability: {stability}")
        print(f"Random seed: {self.seed}")
        print("===============================\n")

    # --------------------------------------------
    def visualize(self, weight_threshold=0.0):
        """Visualize W and its network topology."""
        if self.W is None or self.G is None:
            raise ValueError("Please call .generate() before visualize().")

        W, G, model = self.W, self.G, self.summary["Network model"]
        n = W.shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle(f"Model: {model} | n={n}", fontsize=14, fontweight="bold")

        # 矩阵视图
        im = axes[0].imshow(W, cmap="Blues", interpolation="none")
        axes[0].set_title("Weight matrix $W$")
        axes[0].set_xlabel("Influencer j")
        axes[0].set_ylabel("Receiver i")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        # 网络视图
        pos = nx.spring_layout(G, seed=42)
        weights = [W[u, v] for u, v in G.edges() if W[u, v] > weight_threshold]
        nx.draw_networkx_nodes(G, pos, node_size=80, node_color="skyblue", ax=axes[1])
        nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Blues,
                               arrows=False, width=1.5, ax=axes[1])
        axes[1].set_title("Network topology")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    def export(self, prefix="network", directory="./", gephi_fmt="gexf"):
        """
        Export the influence matrix W, summary info (JSON), and Gephi-compatible file.

        Parameters
        ----------
        prefix : str
            Filename prefix for all exported files.
        directory : str
            Directory path where files will be saved.
        gephi_fmt : str
            File format for Gephi export: 'gexf' or 'graphml'.
        """
        import os, json

        if self.W is None or self.G is None:
            print("⚠️ No network generated yet. Please call .generate() first.")
            return

        os.makedirs(directory, exist_ok=True)

        # ---- 1️⃣ 导出 W 矩阵为 CSV
        w_path = os.path.join(directory, f"{prefix}.csv")
        np.savetxt(w_path, self.W, delimiter=",", fmt="%.6f")
        print(f"✅ Saved W matrix to: {w_path}")

        # ---- 2️⃣ 导出摘要信息为 JSON
        info = self._summarize()


        # ---- 3️⃣ 导出 Gephi 文件（GEXF 或 GraphML）
        # 先写入边权
        for i, j in self.G.edges():
            self.G[i][j]["weight"] = float(self.W[i, j])

        if gephi_fmt.lower() == "gexf":
            g_path = os.path.join(directory, f"{prefix}.gexf")
            nx.write_gexf(self.G, g_path)
        elif gephi_fmt.lower() == "graphml":
            g_path = os.path.join(directory, f"{prefix}.graphml")
            nx.write_graphml(self.G, g_path)
        else:
            raise ValueError("Unsupported gephi_fmt. Use 'gexf' or 'graphml'.")

        print(f"✅ Exported Gephi network to: {g_path}")

        return info