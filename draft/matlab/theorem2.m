%% theorem2.m
% 2个个体(n=2)、2个话题(m=2) 的两组对比实验
% 例1：无竞争依赖 + 结构平衡 -> 非零共识
% 例2：有竞争依赖(符号相反) -> 收敛到 0

clear; clc; close all;

%% 公共设置
n = 2; m = 2;
W = [0.5 0.5;   % 行随机的人际影响矩阵
     0.5 0.5];

% 初始意见(按话题分组)：y = [y1; y2]，每个 yk 是 n×1（所有人对话题k的意见）
y0 = [ 0.8;  -0.3;   % y1(0): 个体1=0.8, 个体2=-0.3
       0.2;   0.6 ]; % y2(0): 个体1=0.2, 个体2=0.6

T  = 40;    % 迭代步数
tol_clip = true;   % 是否每步裁剪到 [-1,1]

%% ===== 例 1：无竞争依赖 + 平衡 =====
% 两个个体的逻辑矩阵 C1, C2 完全一致，无负边
C1 = [1 0; 0.5 0.5];
C2 = [1 0; 0.5 0.5];

A1 = build_A_from_C_and_W({C1,C2}, W);   % 构造(8)中的块矩阵A
[Y_hist1, y_lim1] = iterate_y(A1, y0, T, tol_clip);

figure('Color','w'); 
subplot(1,2,1); hold on; grid on;
plot(0:T, Y_hist1(1,:), 'LineWidth',1.8);   % 话题1-个体1
plot(0:T, Y_hist1(2,:), 'LineWidth',1.8);   % 话题1-个体2
plot(0:T, Y_hist1(3,:), 'LineWidth',1.8);   % 话题2-个体1
plot(0:T, Y_hist1(4,:), 'LineWidth',1.8);   % 话题2-个体2
title('Non competing dependencies' );
xlabel('t'); ylabel('opinion'); ylim([-1.05 1.05]);
legend('y_1^{(1)}','y_1^{(2)}','y_2^{(1)}','y_2^{(2)}','Location','best');

% 计算每个话题的共识值 α_k（最后一步的均值）
alpha1_case1 = mean(Y_hist1(1:2,end));  % 话题1
alpha2_case1 = mean(Y_hist1(3:4,end));  % 话题2
fprintf('alpha_1 ≈ %.4f, alpha_2 ≈ %.4f\n', alpha1_case1, alpha2_case1);

%% ===== 例 2：存在竞争依赖 =====
% 个体1与个体2在 (话题1 -> 话题2) 的符号相反
C1 = [1 0; +0.5 0.5];
C2 = [1 0; -0.5 0.5];

A2 = build_A_from_C_and_W({C1,C2}, W);
[Y_hist2, y_lim2] = iterate_y(A2, y0, T, tol_clip);

subplot(1,2,2); hold on; grid on;
plot(0:T, Y_hist2(1,:), 'LineWidth',1.8);
plot(0:T, Y_hist2(2,:), 'LineWidth',1.8);
plot(0:T, Y_hist2(3,:), 'LineWidth',1.8);
plot(0:T, Y_hist2(4,:), 'LineWidth',1.8);
title('Competing Interdependencies ');
xlabel('t'); ylabel('opinion'); ylim([-1.05 1.05]);
legend('y_1^{(1)}','y_1^{(2)}','y_2^{(1)}','y_2^{(2)}','Location','best');

alpha1_case2 = mean(Y_hist2(1:2,end));
alpha2_case2 = mean(Y_hist2(3:4,end));
fprintf('alpha_1 ≈ %.4f, alpha_2 ≈ %.4f\n', alpha1_case2, alpha2_case2);

%% ===== 函数：由 {C_i} 与 W 构造 (8) 的 A =====
function A = build_A_from_C_and_W(Ccells, W)
    % Ccells: 1×n 的 cell，每个元素是 m×m 的 C_i
    % W: n×n 行随机非负矩阵
    n = numel(Ccells);
    [m, ~] = size(Ccells{1});
    % 先构造 Γ_{kj} = diag(c_{kj,1},...,c_{kj,n})
    Gamma = cell(m,m);
    for k = 1:m
        for j = 1:m
            diagEntries = zeros(n,1);
            for i = 1:n
                diagEntries(i) = Ccells{i}(k,j);
            end
            Gamma{k,j} = diag(diagEntries);
        end
    end
    % 组装块矩阵 A (m×m 个块，每块是 n×n)
    A = zeros(n*m);
    for k = 1:m
        for j = 1:m
            A( (k-1)*n+1:k*n, (j-1)*n+1:j*n ) = Gamma{k,j} * W;
        end
    end
end

%% ===== 函数：迭代 y(t+1) = A y(t) 并记录轨迹 =====
function [Y_hist, y_lim] = iterate_y(A, y0, T, clip01)
    nm = length(y0);
    Y_hist = zeros(nm, T+1);
    Y_hist(:,1) = y0;
    for t = 1:T
        y_next = A * Y_hist(:,t);
        if clip01
            y_next = max(min(y_next,1),-1); % 限制到[-1,1]
        end
        Y_hist(:,t+1) = y_next;
    end
    y_lim = Y_hist(:,end);
end
