function [Xtk,info] = DCFW_orig(n,obj,gradf,gradg,lmo,varargin)
%% DC-FW (var 1,var 2,var 3)
% Implements the DC-FW algorithm for our QAP experiments

Xt = ones(n,n)./n;
time_limit = 9.5*3600; %30 minutes margin dedicated to saving data
T = 1e4;
K = 1e4;
tic
counter_lmo = 0;
counter_gradf = 0;
counter_gradg = 0;

lr_schedule = @(t,k,Xtk,Gg,Gf,Dtk) 2/(k+1);

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


Gf = gradf(Xt);
% counter_gradf = counter_gradf + 1;
Gg = gradg(Xt);
% counter_gradg = counter_gradg + 1;
Gtot = Gf - Gg;
gap_outer = iprod(Gtot, Xt - lmo(Gtot));
%For FW and DCFW 0.001
tol_outer = 0.001 * gap_outer;
%For FW and DCFW 0.8
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

    Xtk = Xt;
    Gg = gradg(Xt);
    counter_gradg = counter_gradg + 1;

    if (2^floor(log2(t)) == t) || t == T
        fprintf('Iteration: %d | Tol: %e, Err: %e \n', t, tol_outer, gap_outer);
    end

    for k = 1:K

        % find the gradient
        Gf = gradf(Xtk);
        counter_gradf = counter_gradf + 1;

        Gtk = Gf - Gg;

        % solve the linear assignment subproblem
        Htk = LAP(Gtk);
        counter_lmo = counter_lmo + 1;

        % update direction
        Dtk = Htk - Xtk;

        % compute inner gap
        gap_inner = -iprod(Gtk , Dtk);

        % check stopping criterion
        if gap_inner <= tol_inner/2
            break;
        end
	


        % For analytic step-size
        eta = lr_schedule(t,k,Xt,Xtk,Gg,Gf,Dtk);

        % update the decision variable
        Xtk = Xtk + eta * Dtk;
	timer = toc;
	if timer> time_limit
	    break;
	end

    end

    if k==K, warning('inner iterations did not converge'); info.failFlag = true; end

    % update the outer model and the gradient

    Xt = Xtk;

    % update the gradient
    Gg = gradg(Xt);
    counter_gradg = counter_gradg + 1;
    Gt = Gf - Gg;

    % Gap of the original problem
    Ht = LAP(Gt);
    counter_lmo = counter_lmo + 1;

    Dt = Ht - Xt;

    gap_outer = -iprod(Gt , Dt);

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

