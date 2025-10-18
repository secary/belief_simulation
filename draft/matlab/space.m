% space.m
% Consensus and Disagreement of  Heterogeneous Belief Systems  in Influence Networks 中太空案例
% 参数配置文件

% 逻辑矩阵
C1  = [1 0;
      0.5 0.5];


C2  = [1 0;
      -0.5 0.5];

% 初始意见
x0 = [1; -0.2];

% 最大迭代次数
max_iters = 100;

% 收敛阈值
tol = 1e-8;

% 是否裁剪到 [-1,1]
clip_to_unit = true;

% 运行单个个体的信念系统迭代
[Xs, x_limit] = single_belief(C1, x0, max_iters, tol, clip_to_unit);    %[1, 1]^T

[Xs2, x_limit2] = single_belief(C2, x0, max_iters, tol, clip_to_unit);  %[1, -1]^T