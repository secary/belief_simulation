# main_system.py
from social_network import SocialNetwork
from logic_matrix import LogicMatrix
from init_belief import InitialBelief
import numpy as np
import json, os, time

def belief_generator(agent_num: int = 10,
         topic_num: int = 3,
         export: bool = False,
         save_dir: str = "./simulations",
         seed: int = None):
    """
    Generate W (social network), C (logic matrices), and X0 (initial beliefs),
    then save all related data into organized files.
    """

    # ===== 0️⃣ 基本设置 =====
    if seed is None:
        seed = int(time.time() * 1e6) % (2**32)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    print(f"🔹 Experiment seed: {seed}")

    # ===== 1️⃣ 初始信念 X0 =====
    X0_obj = InitialBelief(
        n=agent_num,
        m=topic_num,
        mode="beta",
        beta_params=(2, 2),
        seed=seed
    ).generate()
    X0 = X0_obj.X0

    # ===== 2️⃣ 社会网络 W =====
    W_obj = SocialNetwork(
        n=agent_num,
        random_beta=True,
        seed=seed
    ).generate()
    W = W_obj.W

    # ===== 3️⃣ 逻辑矩阵 C =====
    C_obj = LogicMatrix(
        n=agent_num,
        m=topic_num,
        random_beta=True,
        heterogeneity=0.2,
        sparsity=0.6,
        seed=seed
    ).generate()
    
    C_base = C_obj.C_base
    C_list = np.array(C_obj.C_list)   # shape: (n, m, m)
    
    # ===== 4️⃣ 存储元信息 =====
    if export:
        # 1. 保存初始信念
        np.savetxt(f"{save_dir}/X0.csv", X0, delimiter=",", fmt="%.4f")
        print(f"✅ Saved initial belief X0 to {save_dir}/X0.csv")

        # 2. 导出网络和逻辑矩阵
        w_info = W_obj.export(prefix="W", directory=save_dir, gephi_fmt="gexf")
        c_info = C_obj.export(prefix="C", directory=save_dir)
        np.save(f"{save_dir}/C_tensor.npy", C_list)
        print(f"✅ Saved logic tensor (C_i) to {save_dir}/C_tensor.npy")

        # 3. 写元信息
        meta = {
            "agent_num": agent_num,
            "topic_num": topic_num,
            "seed": seed,
            "X0_shape": X0.shape,
            "W_shape": W.shape,
            "C_shape": C_list.shape,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(f"{save_dir}/meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        
        # 4. 合并 JSON 文件
        summary = {
            "meta": meta,
            "network_info": w_info,
            "logic_info": c_info
        }

        summary_path = os.path.join(save_dir, "meta.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

    

    # ===== 5️⃣ 返回对象 =====
    print("\n🎯 All components generated successfully.")
    return {
        "X0": X0,
        "W": W,
        "C_list": C_list,
        "W_obj": W_obj,
        "C_obj": C_obj,
        "X0_obj": X0_obj
    }

# 直接运行测试
if __name__ == "__main__":
    results = belief_generator(agent_num=10, topic_num=3)
