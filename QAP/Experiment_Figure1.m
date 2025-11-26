%% Test setup for QAP experiment 
% Replicates the experiment and generates Figure 2 in [MHSY25].
%
% [MHSY25] H. Maskan, Y.Hou, S.Sra, A. Yurtsever
% "Revisiting Frank-Wolfe for Structured Nonconvex Optimization"
% 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
% 
% contact information: https://github.com/hoomyhh

%% Preamble
clearvars
close all
rng(0,'twister');
addpath(genpath('utils'));
addpath methods;

%% Load data
% NOTE: You need to download data from QAPLIB and locate them to under the
% "./data/" folder, together with the solution files
% Links for QAPLIB:
%   http://anjos.mgi.polymtl.ca/qaplib/inst.html
%   https://www.opt.math.tugraz.at/qaplib/inst.html
%	https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/

% Insert the dataset name below
dataname = 'lipa20b';


[A,B,OPT,P] = qapread(['./data/',dataname]);
if (norm(A,'fro') ~= 0) && (norm(B,'fro') ~= 0)
A = A/norm(A,'fro');
B= B/norm(B,'fro');
OPT = iprod(A'*P,P*B);
end
n = size(A,1);

%% Operators

obj = @(X) iprod(A'*X,X*B); % = trace(X'*A*X*B); % These are same but iprod is more efficient
grad = @(X) A'*X*B' + A*X*B;
lmo = @(G) LAP(G); 

%% Sanity check for the gradient operators
% for tt = 1:10
%     Xtmp = randn(n,n);
%     if norm(gradf(Xtmp) - gradg(Xtmp) - grad(Xtmp),'fro') > 1e-12
%         error('gradient operators do not match');
%     end
% end

%% Choose initial point
% Generate a random direction with Gaussian iid entries, then project
% (approximately) onto the Birkhoff polytope
X0 = ApproxProjBirkhoff(ones(n,n)./n + 0.001 * randn(n,n),1e3);


%% DC-FW-1
gradf = @(X) 0.5 * (A*(A'*X + X*B) + (A'*X + X*B)*B');% Derivative of 1/4*||A'X+XB||_F^2
gradg = @(X) 0.5 * (A*(A'*X - X*B) - (A'*X - X*B)*B');% Derivative of 1/4*||A'X-XB||_F^2
lr_schedule = @(t,k,Xt,Xtk,Gg,Gf,Dtk) max(min(...
    ( 2 * iprod(Gg, Dtk) - iprod(A'*Dtk + Dtk*B, A'*Xtk + Xtk*B) ) /  norm(A'*Dtk + Dtk*B,'fro')^2  ...
    ,1),0);

[XDCFW1, infoDCFW1] = DCFW_orig(n,obj,gradf,gradg,lmo,'lr_schedule',lr_schedule,'x0',X0);
XDCFW1_rdd = ProjPermMatrix(XDCFW1);
infoDCFW1.rddobj = obj(XDCFW1_rdd);
errDCFW1 = (infoDCFW1.rddobj - OPT)/max(OPT,1);
%%  DC-Frank-Wolfe (var2)

% Other DC decomposition: f(X) = L/2*||X||^2 , g(X) = L/2*||X||^2 - trace(X'*A*X*B)


if issparse(A), nA = svds(A,1); else, nA = norm(A); end
if issparse(B), nB = svds(B,1); else, nB = norm(B); end
L = nA*nB;

gradf = @(X) L*X; % Derivative of L/2*||X||^2
gradg = @(X) L*X - grad(X); % Derivative of L/2*||X||^2 - trace(X'*A*X*B)

lr_schedule = @(t,k,Xt,Xtk,Gg,Gf,Dtk) max(min(...
    ( iprod(Gg - L *Xtk, Dtk) ) / ( L * norm(Dtk,'fro')^2 )  ...
    ,1),0);

[XDCFW2, infoDCFW2] = DCFW_orig(n,obj,gradf,gradg,lmo,'lr_schedule',lr_schedule,'x0',X0);
XDCFW2_rdd = ProjPermMatrix(XDCFW2);
infoDCFW2.rddobj = obj(XDCFW2_rdd);
errDCFW2 = (infoDCFW2.rddobj - OPT)/max(OPT,1);


%% DC-Frank-Wolfe (var3)

% Other DC decomposition: f(X) =  trace(X'*A*X*B) - L/2*||X||^2  , g(X) = L/2*||X||^2 

gradf = @(X) grad(X) + L*X;
gradg = @(X) L*X;


lr_schedule = @(t,k,Xt,Xtk,Gg,Gf,Dtk) max(min( -(iprod(Gf,Dtk)- L*iprod(Xt,Dtk))/(2*(iprod(A'*Dtk,Dtk*B) + L/2* norm(Dtk,'fro')^2)) ,1),0);

[XDCFW3, infoDCFW3] = DCFW_orig(n,obj,gradf,gradg,lmo,'lr_schedule',lr_schedule,'x0',X0);


XDCFW3_rdd = ProjPermMatrix(XDCFW3);
infoDCFW3.rddobj = obj(XDCFW3_rdd);
errDCFW3 = (infoDCFW3.rddobj - OPT)/max(OPT,1);


%% Non-cvx-Cnd
alpha_k = @(k) 2/(k+1);
lambda_k = 1/(2*L);

[Xnon_Cnd, infonon_Cnd] = Noncvx_CondFW(n,obj,grad,lmo,alpha_k,lambda_k,'x0',X0);


Xnon_Cnd_rdd = ProjPermMatrix(Xnon_Cnd);
infonon_Cnd.rddobj = obj(Xnon_Cnd_rdd);
errnon_Cnd = (infonon_Cnd.rddobj - OPT)/max(OPT,1);

%% FW

lr_schedule = @(t,k,Xtk,Dtk) max(min( - (iprod(Xtk , A*Dtk*B) + iprod(Dtk , A*Xtk*B)) / (2 * iprod(Dtk, A*Dtk*B)),1),0);

[XFW, infoFW] = FW_orig(n,obj,grad,lmo,A,B,'lr_schedule',lr_schedule,'x0',X0);

XFW_rdd = ProjPermMatrix(XFW);
infoFW.rddobj = obj(XFW_rdd);
errFW = (infoFW.rddobj - OPT)/max(OPT,1);



