
% Comparing the Performance of DCFW, FW-M, FW-K, theoretical bound using
% DC_Gap and #SVD Calculations

close all

rng(0);            % Seed for reproducibility

load('fasttext_embeddings.mat');
E1 = E1(1:300,1:1e4);
E2 = E2(1:300,1:1e4);

%% Parameters

d = size(E1,1);     % [Embedding dimension, Number of words
n = size(E1,2);     % Number of words
obs_ratio = 0.1;    % Observation ratio (10% observed)
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
maxit = 1e3;     %1000 iterations
%% Operators


norm_nuc = @(W) norm(svd(W),1);
f = @(W) (0.5 / n) * norm((M.*(W*E1) - Y),'fro')^2;
g = @(W) reg_par_Schatten_1 * norm_nuc(W);
obj = @(W) f(W) - g(W);
load results

hfig = figure('Name','EmbeddingAlignment','NumberTitle','off','Position',[100,100,1000,285]);

subplot(121)
tk_sum = 1:10^3;
loglog(infoFWK.gap_DC);
hold on;
loglog(infoFWM.gap_DC);
loglog(infoDCFW.outer.tk, infoDCFW.outer.gap_DC);
loglog(tk_sum, sqrt((8*(obj(W0)-obj(XDCFW))^2)./(tk_sum)),'Linestyle',"--")

xlabel('iterations','Interpreter','latex','fontsize',14)
ylabel('gap$_{DC}$','Interpreter','latex','fontsize',14)
hleg = legend('FW-K', 'FW-M', 'DC-FW', 'Theoretical Bound');
hleg.Interpreter = 'latex';
hleg.FontSize = 14;
hleg.Location = 'SouthWest';
axis tight;
ax = gca
ax.XTick = 10.^(0:6);


subplot(122)

semilogy(infoFWK.counter_subgradg + infoFWK.counter_lmo, infoFWK.gap_DC);
hold on;
loglog(infoFWM.counter_subgradg + infoFWM.counter_lmo + infoFWM.counter_obj, infoFWM.gap_DC);
loglog(infoDCFW.outer.counter_subgradg + infoDCFW.outer.counter_lmo, infoDCFW.outer.gap_DC)  ;

xlabel('SVD calculations','Interpreter','latex','fontsize',14)
xlim([0,4e3])
axis tight;
ax = gca;
ax.XTick = (0:4)*1e3;
xlim([0,4e3])


for t = 1:2

    subplot(1,2,t)
    ax = gca;
    set(findall(ax, 'Type', 'line'),'LineWidth',3);
    ax.FontSize = 14;
    ax.TickLabelInterpreter = 'latex';
    ax.TickDir = 'out';
    grid on; grid minor; grid minor;
    set(gca,'TickDir','out')
    set(gca,'LineWidth',0.75,'TickLength',[0.02 0.02]);
    %     ax.XTick = 10.^(-10:10);
    %     ax.YTick = 10.^(-100:2:100);
    ax.Box = 'on';

end

