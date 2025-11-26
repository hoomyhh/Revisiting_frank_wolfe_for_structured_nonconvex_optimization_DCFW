function [Xtk,info] = Noncvx_CondFW(n,obj,grad,lmo,alpha_k,lambda_k,varargin)
%% Nonconvex Conditional Gradient Sliding
% Implements the Nonconvex Conditional Gradient Sliding (CGS) algorithm for
% our QAP experiments (Qu et al., 2018)

Xt = ones(n,n)./n;
time_limit = 9.5*3600; %30 minutes margin dedicated to saving data
T = 1e4;
K = 1e4;
tic
eta_k = 1/T;
beta_k = lambda_k;

counter_lmo = 0;
counter_gradf = 0;
counter_gradg = 0;

lr_schedule = @(t,k,Xt,Xtk,Gg,Gf,Dtk) 2/(k+1);

if ~isempty(varargin)
    for tt = 1:2:length(varargin)
        switch lower(varargin{tt})
            case 'tol'
                tol_outer = varargin{tt+1};
            case 'maxit'
                T = varargin{tt+1};
            case 'maxit_inner'
                K = varargin{tt+1};
            case 'x0'
                Xt = varargin{tt+1};
            case 'lr_schedule'
                lr_schedule = varargin{tt+1};
            otherwise
                warning(['Unknown option: ',varargin{tt}]);
        end
    end
end


Xag = Xt;
Gtot = grad(Xt);

gap_outer = iprod(Gtot, Xt - lmo(Gtot));
% counter_lmo = counter_lmo + 1;
tol_outer = 0.001 * gap_outer;
tol_inner_multiplier = 0.8;
tol_inner = tol_inner_multiplier * gap_outer;
tol_update_counter = 0;


info.iter = nan(T,1);
info.time = nan(T,1);
info.lmo_calls = nan(T,1);
info.obj = nan(T,1);
info.gap = nan(T,1);
info.failFlag = false;


time_start = tic; % start timer

for t = 1:T
    Xt_1 = Xt;
    Xmd = (1-alpha_k(t)) * Xag + alpha_k(t) * Xt;

    if (2^floor(log2(t)) == t) || t == T
        fprintf('Iteration: %d | Tol: %e, Err: %e \n', t, tol_outer, gap_outer);
    end

        
    Gtot = grad(Xmd);
    counter_gradg = counter_gradg + 1;

    Xtk = Xt;
    for k = 1:K

            
        % solve the linear assignment subproblem
        At = Gtot + 1/(lambda_k) * (Xtk - Xt);
        vt = LAP(At);
        counter_lmo = counter_lmo + 1;

        Dtk = vt - Xtk;

        Vt = -iprod(At , Dtk);
        
        
        % check the convergence
        % if Vt <= eta_k
        %     Xt = Xtk;
        %     break;
        % end
        if Vt <= tol_inner/2
            Xt = Xtk;
            break;
        end

        % For analytic step-size
        xi = max(min(iprod(-At , Dtk ) / (1/(lambda_k) * norm(Dtk,'fro')^2),1),0);

        % update u_t
        Xtk = Xtk + xi * Dtk;
      	timer = toc;
	if timer> time_limit
	  break;
	end
    end

    if k==K
       warning('inner iterations did not converge')
       info.failFlag = true; 
    end
    
    Xt = Xtk;
    % update the outer model and the gradient

    Xag = Xmd - beta_k * (Xt_1 - Xt) / lambda_k;

    % update the gradient
    Gtot = grad(Xt);
    counter_gradg = counter_gradg + 1;
    

    % Gap of the original problem
    Ht = LAP(Gtot);
    counter_lmo = counter_lmo + 1;

    Dt = Ht - Xt;

    gap_outer = -iprod(Gtot , Dt);

    % measurements
    info.iter(t,1) = k;
    info.gap(t,1) = gap_outer;
    info.obj(t,1) = obj(Xt);
    info.lmo_calls(t,1) = counter_lmo;
    info.time(t,1) = toc(time_start);

    if gap_outer < tol_outer
        break;
    end
    timer = toc;
    if timer> time_limit
    	break;
    end
    if gap_outer < tol_inner
        tol_inner = tol_inner_multiplier*gap_outer;
        tol_update_counter = tol_update_counter + 1;
    end

end

info.iter(t+1:end) = [];
info.gap(t+1:end) = [];
info.obj(t+1:end) = [];
info.lmo_calls(t+1:end) = [];
info.time(t+1:end) = [];

end

