# Simulation
## Degroot Model from [1]
$$
x_i(t+1) = \sum_{j=1}^{n} W_{ij} C_i x_j(t)
$$

## Pipeline
1. 初始化 $W, C, x(0)$ 
2. 按时间步模拟
3. 画图

## $W$初始生成函数
* 目的：测试不同$W$的社交网络结构的最终收敛
    Random Networks
  * Watts-Strogatz model
  * Barabasi-Albert model
  * Erdos-Renyi model
  * Random-regular model
* 小样本或灵活输入样本数
* 行随机矩阵
* 选取特征绝对值小于1的

## $C$初始生成函数
* 与研究问题相关

## 动态模拟
* 也许可以早停
* 时间步长与研究相关

## 可视化
* 制定相关metric(指标)
* 选取相关画图维度
* 选取最有说服力的图像


## Reference
[1] M. Ye, J. Liu, L. Wang, B. D. O. Anderson, and M. Cao, “Consensus and Disagreement of
Heterogeneous Belief Systems in Influence Networks,” IEEE Transactions on Automatic
Control, vol. 65, no. 11, pp. 4679–4694, Nov. 2020. [Online]. Available:
https://ieeexplore.ieee.org/document/8941271/
