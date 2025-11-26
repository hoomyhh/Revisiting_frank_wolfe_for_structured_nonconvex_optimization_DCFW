%% Test setup for Embeddings Alignment Experiment
% Replicates the experiment and generates Figure 3 in [MHSY25].
%
% [MHSY25] H. Maskan, Y.Hou, S.Sra, A. Yurtsever
% "Revisiting Frank-Wolfe for Structured Nonconvex Optimization"
% 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
% 
% contact information: https://github.com/hoomyhh 


clc; clear; close all;
rng(0);            % Seed for reproducibility

load('fasttext_embeddings.mat');
E1 = E1(1:300,1:1e4);
E2 = E2(1:300,1:1e4);

%% Parameters

d = size(E1,1);     % [Embedding dimension, Number of words
n = size(E1,2);     % Number of words
obs_ratio = 0.1;    % Observation ratio (10% observed)
iter = 30;          % Number of gradient steps to calculate Gap_DC approximately
%% Procrustes Alignment to Find W
% This is the ideal case scenario if we had access to all embeddings. 
M_all = E2 * E1';
[U, ~, V] = svd(M_all);
W_est = U * V';

%% Generate Partial Observation Mask
M = rand(d, n) < obs_ratio;     % Binary mask with observation ratio
Y = M .* E2;                    % Partially observed embeddings

%% Problem Parameters

radius_Schatten_inf = 1;        % Radius of the problem domain
reg_par_Schatten_1 = 1e-4;      % Regularization parameter

%% Algorithm Parameters

W0 = zeros(d);   % initial point
maxit = 1e3;     % 10000 total iterations
tolerance = 0;   % stopping criterion

%% Operators

norm_nuc = @(W) norm(svd(W),1);
f = @(W) (0.5 / n) * norm((M.*(W*E1) - Y),'fro')^2;
g = @(W) reg_par_Schatten_1 * norm_nuc(W);
obj = @(W) f(W) - g(W);
gradf = @(W) (1/n) * (M.*(W*E1) - Y)*E1';
subgradg = @(W) eval_subgrad(W, reg_par_Schatten_1);
lmo = @(W) eval_lmo(W, radius_Schatten_inf);

%% Evaluate smoothness and curvature constants

norm_nuc_W_true = d;        % by definition
norm_fro_W_true = sqrt(d);  % by definition
norm_spect_W_true = 1;      % by definition

E1E1 = E1*E1';
L_in_norm_spect = norm(E1E1) / n;       % Lipschitz constants in different norms
L_in_norm_nuc = norm(E1,'fro')^2 / n;   % Lipschitz constants in different norms
L_in_norm_fro = norm(E1E1,'fro') / n;   % Lipschitz constants in different norms

C0 = min(...
    [norm_nuc_W_true^2 * L_in_norm_spect, ...
    norm_spect_W_true^2 * L_in_norm_nuc, ...
    norm_fro_W_true^2 * L_in_norm_fro]);    %Curvature constant

%% Run FW-K

[XFWK, infoFWK] = FW_K(...
    @(x) f(x), ...
    @(x) gradf(x), ...
    @(x) subgradg(x), ...
    @(x) lmo(x), ...
    maxit, tolerance, W0, C0, iter);

[U,~,V] = svd(XFWK);
XFWKrdd = U*V';
norm(XFWKrdd*E1 - E2, 'fro')

%% Run FW-M

[XFWM, infoFWM] = FW_M(...
    @(x) f(x), ...
    @(x) gradf(x), ...
    @(x) subgradg(x), ...
    @(x) lmo(x), ...
    @(x) obj(x), ...
    maxit, tolerance, W0, L_in_norm_fro, iter);

[U,~,V] = svd(XFWM);
XFWMrdd = U*V';
norm(XFWMrdd*E1 - E2, 'fro')

%% Run DCFW

[XDCFW, infoDCFW] = DCFW(...
    @(x) f(x), ...
    @(x) gradf(x), ...
    @(x) subgradg(x), ...
    @(x) lmo(x), ...
    maxit, tolerance, W0, C0, iter);

[U,~,V] = svd(XDCFW);
XDCFWrdd = U*V';
norm(XDCFWrdd*E1 - E2, 'fro')


%% Save results (run main_plot.m later to obtain plots)
save(['results.mat'],...
    'XDCFW','XDCFWrdd','infoDCFW',...
    'XFWM','XFWMrdd','infoFWM',...
    'XFWK','XFWKrdd','infoFWK')

%% Subgradient and linear minimization oracles
function out = eval_subgrad(W, reg_par)
    
    [U,~,V] = svd(W);
    out = reg_par * (U*V');

end

function out = eval_lmo(H, radius)
    
    [U,~,V] = svd(H);
    out = -radius * (U*V');

end