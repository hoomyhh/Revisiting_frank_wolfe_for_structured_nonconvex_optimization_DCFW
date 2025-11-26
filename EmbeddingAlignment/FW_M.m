function [Xt, info] = FW_M(f,gradf, subgradg, lmo, obj, maxit, tol, x0, L0, iter)
% Implementation of FW method from 
%
% [Millán, R. D., Ferreira, O., and Ugon, J. Frank-Wolfe algorithm for DC optimization problem. arXiv preprint
% arXiv:2308.16444, 2023]


Xt = x0;
obj_t = obj(Xt);
Lt = L0;

% initialize counters
counter_lmo = 0;
counter_gradf = 0;
counter_subgradg = 0;
counter_obj = 0;
timer = tic; 

for t = 1:maxit
    ut = subgradg(Xt);
    Ht = gradf(Xt) - ut;
    counter_subgradg = counter_subgradg + 1;
    counter_gradf = counter_gradf + 1;

    St = lmo(Ht);
    counter_lmo = counter_lmo + 1;
    
    Dt = Xt - St;
    gap = Ht(:)'*Dt(:); 
    
    gap_DC_maximizer = calc_gap_DC(@(x) gradf(x),Xt,ut,L0,iter);
    counter_gradf = counter_gradf + iter;
    counter_subgradg = counter_subgradg + iter;

    Dt2 = Xt - gap_DC_maximizer;
    gap_DC = f(Xt) - f(gap_DC_maximizer) - ut(:)'*Dt2(:);
    % Display progress every 10 iterations
    if mod(t, 100) == 0 || t == 1 || gap <= tol
        fprintf('Iteration %d: Gap = %.6f\n', t, gap);
    end

    if gap <= tol
        break;
    end
    
    j = 0;
    while true
        lbd = min(1, gap / (2^j * Lt * norm(Dt,'fro')^2));
        Xt_hat = Xt - lbd*Dt;
        obj_t_hat = obj(Xt_hat);
        counter_obj = counter_obj + 1;
        if obj_t_hat <= obj_t - lbd * gap + 2^j * Lt/2 * norm(Dt,'fro')^2 * lbd^2
            Xt = Xt_hat;
            obj_t = obj_t_hat;
            Lt = 2^j * Lt/2;
            break;
        else
            j = j + 1;
        end
    end

    info.t(t) = t;
    info.counter_lmo(t) = counter_lmo;
    info.counter_gradf(t) = counter_gradf;
    info.counter_subgradg(t) = counter_subgradg;
    info.gap(t) = gap;
    info.gap_DC(t) = gap_DC;
    info.counter_obj(t) = counter_obj;
    info.time(t) = toc(timer);

end

end

