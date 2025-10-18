import numpy as np, pandas as pd, os

def init_w_random(n: int, seed: int = None, export_csv: bool = False, filename: str = "W_random.csv") -> np.ndarray:
    """
    初始化一个 n×n 的随机行随机矩阵 W。
    每一行的元素非负且和为 1。
    """
    if seed is not None:
        np.random.seed(seed)
    W = np.random.rand(n, n)
    W = W / W.sum(axis=1, keepdims=True)
    
    if export_csv:
        export_w_to_csv(W, filename)
    return W


def init_w_stable(n: int, target_radius: float = 0.95, seed: int = None, export_csv: bool = False, filename: str = "W_stable.csv"):
    """
    严格生成谱半径 < 1 的稳定矩阵。
    target_radius：目标谱半径（例如 0.95）
    export_csv: 是否导出为 CSV（用于 Gephi）
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. 随机行随机矩阵
    W = np.random.rand(n, n)
    W /= W.sum(axis=1, keepdims=True)

    # 2. 计算当前谱半径
    eigvals = np.linalg.eigvals(W)
    rho = np.max(np.abs(eigvals))

    # 3. 缩放到目标谱半径
    W = W / rho * target_radius

    # 4. 打印稳定性信息
    print("谱半径 ρ =", np.max(np.abs(np.linalg.eigvals(W))))
    if np.max(np.abs(np.linalg.eigvals(W))) < 1:
        print("✅ 系统稳定（会收敛）")
    else:
        print("⚠️ 系统不稳定（可能发散）")

    # 5. 导出为 CSV（Gephi 可直接导入）
    if export_csv:
        export_w_to_csv(W, filename)
    
    return W

def init_w_kol(n: int, alpha: float = 0.5, seed: int = None, export_csv: bool = False, filename: str = "W_kol.csv") -> np.ndarray:
    """
    初始化一个带 KOL 的影响矩阵。
    alpha 控制 KOL 的影响程度 (0~1)，越大表示越集中在 KOL。
    export_csv: 是否导出为 CSV（用于 Gephi）
    """
    if seed is not None:
        np.random.seed(seed)
    W = np.random.rand(n, n)
    W = W / W.sum(axis=1, keepdims=True)

    # 提升 KOL (节点 0) 的影响力
    W[:, 0] += alpha
    W = W / W.sum(axis=1, keepdims=True)

    if export_csv:
        export_w_to_csv(W, filename)
    return W


def export_w_to_csv(W: np.ndarray, filename: str = "W.csv", threshold: float = 0.0):
    """
    将影响矩阵 W 导出为 Gephi 可识别的边列表 CSV。
    threshold: 过滤掉权重过小的边（例如 0.01 可去掉弱影响）
    """
    n = W.shape[0]
    edges = [(i, j, W[i, j]) for i in range(n) for j in range(n) if W[i, j] > threshold]
    df = pd.DataFrame(edges, columns=["Source", "Target", "Weight"])
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"📁 已导出到 {filename} ({len(df)} 条边)")
    
    
def export_nodes(n: int, kol_index=0, filename="gephi_data/nodes.csv"):
    data = []
    for i in range(n):
        label = "KOL" if i == kol_index else f"Agent{i}"
        data.append((i, label))
    df = pd.DataFrame(data, columns=["Id", "Label"])
    df.to_csv(filename, index=False)
    print(f"📁 节点文件已导出到 {filename}")


if __name__ == "__main__":
    SEED = 1942340
    W = init_w_stable(n=6, target_radius=0.9, seed=SEED, export_csv=True, filename="gephi_data/W_stable.csv")
    W_random = init_w_random(n=6, seed=SEED, export_csv=True, filename="gephi_data/W_random.csv")
    W2 = init_w_kol(n=6, alpha=0.8, seed=SEED, export_csv=True, filename="gephi_data/W_kol.csv")
    export_nodes(n=6, kol_index=0, filename="gephi_data/nodes.csv")