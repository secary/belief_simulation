function [Xs, x_limit] = single_belief(C, x0, max_iters, tol, clip_to_unit)
% single_belief 模拟单个个体的信念系统迭代: x(t+1) = C * x(t)
%
% 输入:
%   C            - 逻辑矩阵 (m×m)
%   x0           - 初始意见向量 (m×1)
%   max_iters    - 最大迭代次数 (默认 200)
%   tol          - 收敛阈值 (默认 1e-10)
%   clip_to_unit - 是否裁剪到 [-1,1] 区间 (true/false, 默认 true)
%
% 输出:
%   Xs      - 每一步的意见轨迹 (m × T)
%   x_limit - 收敛到的极限值 (m × 1)

    % 参数默认值
    if nargin < 3 || isempty(max_iters), max_iters = 200; end
    if nargin < 4 || isempty(tol), tol = 1e-10; end
    if nargin < 5 || isempty(clip_to_unit), clip_to_unit = true; end

    % 初始化
    x  = x0;
    Xs = zeros(length(x0), max_iters+1);
    Xs(:,1) = x0;

    % 迭代
    for t = 1:max_iters
        x_new = C * x;
        if clip_to_unit
            x_new = max(min(x_new, 1), -1); % 保持在 [-1,1]
        end
        Xs(:,t+1) = x_new;

        if norm(x_new - x, 2) < tol
            Xs = Xs(:,1:t+1);
            break;
        end
        x = x_new;
    end

    x_limit = Xs(:,end);

    % 作图
    T = 0:size(Xs,2)-1;
    figure('Color','w'); hold on; grid on;
    plot(T, Xs(1,:), 'LineWidth', 1.8);   % 第1个话题
    if size(Xs,1) >= 2
        plot(T, Xs(2,:), 'LineWidth', 1.8); % 第2个话题
    end
    xlabel('t (iteration)'); ylabel('opinion value');
    legend(arrayfun(@(k) sprintf('Topic %d',k), 1:size(Xs,1), 'UniformOutput',false));
    title('Belief Dynamics: x(t+1) = C x(t)');
end
