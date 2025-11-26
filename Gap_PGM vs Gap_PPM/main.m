%% Test setup for DC Gap comparisons
% Replicates the experiment and generates Figure 1 in [MHSY25].
%
% [MHSY25] H. Maskan, Y.Hou, S.Sra, A. Yurtsever
% "Revisiting Frank-Wolfe for Structured Nonconvex Optimization"
% 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
% 
% contact information: https://github.com/hoomyhh

close all
clc
clear
% Grid setup
x1 = -1:0.01:1;
x2 = -1:0.01:1;

[X1, X2] = meshgrid(x1, x2);

% Parameters

L = pi^2;
eta = 1/L;       % PGM step size
lambda = 1/L;    % PPM step size

% Initialize arrays
PGM_gap = zeros(size(X1));
PPM_gap = zeros(size(X1));
FW_gap  = zeros(size(X1));
F_vals  = sin(pi * X1) .* cos(pi * X2);  % function values

% Loop over grid
for i = 1:numel(X1)
    i / numel(X1)

    x = [X1(i); X2(i)];

    %% Gradient of f(x1, x2) = sin(pi*x1) * cos(pi*x2)
    grad_f = [ ...
        pi * cos(pi * x(1)) * cos(pi * x(2)); ...
        -pi * sin(pi * x(1)) * sin(pi * x(2)) ];                

    %% --- PGM gap ---
    pgm_step = x - eta * grad_f;
    pgm_proj = min(max(pgm_step, -1), 1);  % projection onto box
    PGM_gap(i) = 0.5/eta*norm(x - pgm_proj)^2;

    %% --- PPM gap ---
    obj = @(z) sin(pi * z(1)) * cos(pi * z(2)) + (1 / (2 * lambda)) * norm(z - x)^2;
    z0 = x;  % initialization
    lb = [-1; -1]; ub = [1; 1];
    opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');
    z_star = fmincon(obj, z0, [], [], [], [], lb, ub, [], opts);
    PPM_gap(i) = 0.5/lambda*norm(x - z_star)^2;

    %% --- Frank-Wolfe gap ---
    s_star = -sign(grad_f);                     % minimize ⟨∇f, s⟩
    s_star = max(min(s_star, 1), -1);           % clip to box
    FW_gap(i) = dot(grad_f, x - s_star);

    
end

%%
% Reshape for plotting
PGM_gap = reshape(PGM_gap, size(X1));
PPM_gap = reshape(PPM_gap, size(X1));
FW_gap  = reshape(FW_gap,  size(X1));

% Shared color limits
vmin_all = min([PGM_gap(:); PPM_gap(:)]);
vmax_all = max([PGM_gap(:); PPM_gap(:)]);

%% Plot: level curves of f(x1, x2)
figure;
contourf(X1, X2, F_vals, 20);
colorbar;
xlabel('x_1'); ylabel('x_2');
title('Level curves of f(x_1, x_2) = x_1^2 - x_2^2 ');
axis equal;

%% Plot: PGM gap
figure;
contourf(X1, X2, PGM_gap, 20);
colorbar;
xlabel('x_1'); ylabel('x_2');
title('PGM gap: ||x - proj(x - \eta \nabla f(x))||');
caxis([vmin_all vmax_all]);
axis equal;

%% Plot: PPM gap
figure; 
contourf(X1, X2, PPM_gap, 20);
colorbar;
xlabel('x_1'); ylabel('x_2');
title('PPM gap: ||x - prox_{F}(x)||');
caxis([vmin_all vmax_all]);
axis equal;

%% Plot: FW gap
figure;
contourf(X1, X2, FW_gap, 20);
colorbar;
xlabel('x_1'); ylabel('x_2');
title('Frank-Wolfe gap: max_{s in C} ⟨∇f(x), x - s⟩');
caxis([vmin_all vmax_all]);
axis equal;



set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

set(groot,'defaultAxesFontName','Times');
set(groot,'defaultAxesFontSize',24);

set(gcf,'Renderer','opengl');    % robust with images/transparency

% target width/height, e.g., 85 mm wide (single-column)
w = 85; h = 60;                       % in cm
set(gcf, 'Units','centimeters', 'Position',[1 3 w h]); 

f = gcf;
figure(f);
subplot(1,3,1);
contourf(X1, X2, F_vals, 20);
colorbar;
xlabel('$x_1$', 'Interpreter','latex','FontSize',28); ylabel('$x_2$', 'Interpreter','latex','FontSize',28);
title('Level curves of $ sin(\pi x_1) cos(\pi x_2)$ ', 'Interpreter','latex','FontSize',20);
axis equal;


subplot(1,3,2);
contourf(X1, X2, PGM_gap, 20);
colorbar;
xlabel('$x_1$', 'Interpreter','latex','FontSize',28); ylabel('$x_2$', 'Interpreter','latex','FontSize',28);
title('gap$_{PGM}^L(x_1,x_2)$', 'Interpreter','latex','FontSize',22);
caxis([vmin_all vmax_all]);
axis equal;

subplot(1,3,3);
contourf(X1, X2, PPM_gap, 20);
colorbar;
xlabel('$x_1$', 'Interpreter','latex','FontSize',28); ylabel('$x_2$', 'Interpreter','latex','FontSize',28);
title('gap$_{PPM}^L(x_1,x_2)$', 'Interpreter','latex','FontSize',24);
caxis([vmin_all vmax_all]);
axis equal;

% 3) Export as vector PDF
exportgraphics(gcf, 'figure.pdf', ...
    'ContentType','vector', ...
    'BackgroundColor','white', ...
    'Resolution',300); 