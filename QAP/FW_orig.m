function [Xt,info] = FW_orig(n,obj,grad,lmo,A,B,varargin)
%% FW
% Implements the FW algorithm for our QAP experiments

Xt = ones(n,n)./n;
T = 1e4;
time_limit = 9.5*3600; %30 minutes margin dedicated to saving data
counter_lmo = 0;
counter_grad = 0;

lr_schedule = @(t,k,Xtk,Dtk) 2/(t+1);

if ~isempty(varargin)
    for tt = 1:2:length(varargin)
        switch lower(varargin{tt})
            case 'tol'
                tol = varargin{tt+1};
            case 'maxit'
                T = varargin{tt+1};
            case 'x0'
                Xt = varargin{tt+1};
            case 'lr_schedule'
                lr_schedule = varargin{tt+1};
            otherwise
                warning(['Unknown option: ',varargin{tt}]);
        end
    end
end

Gt = grad(Xt);
gap = iprod(Gt, Xt - lmo(Gt));
tol = 0.001 * gap;

info.time = nan(T,1);
info.lmo_calls = nan(T,1);
info.obj = nan(T,1);
info.gap = nan(T,1);
info.failFlag = false;

time_start = tic; % start timer

for t = 1:T

    Gt = grad(Xt);
    counter_grad = counter_grad + 1;

    if (2^floor(log2(t)) == t) || t == T
        fprintf('Iteration: %d | Tol: %e, Err: %e \n', t, tol, gap);
    end

    % solve the linear assignment subproblem
    Ht = LAP(Gt);
    counter_lmo = counter_lmo + 1;

    % update direction
    Dt = Ht - Xt;

    % compute inner gap
    gap = -iprod(Gt , Dt);

    % For analytic step-size
    if iprod(Dt,A*Dt*B)>0
    	eta = lr_schedule(t,[],Xt,Dt);
    else
    	eta = 1;
    end
    
    % update the decision variable
    Xt = Xt + eta * Dt;

    % measurements
    info.gap(t,1) = gap;
    info.obj(t,1) = obj(Xt);
    info.lmo_calls(t,1) = counter_lmo;
    info.time(t,1) = toc(time_start);

    if gap < tol
        break;
    end
    
    if info.time(t,1)> time_limit
       break;
    end

end

info.gap(t+1:end) = [];
info.obj(t+1:end) = [];
info.lmo_calls(t+1:end) = [];
info.time(t+1:end) = [];

end

