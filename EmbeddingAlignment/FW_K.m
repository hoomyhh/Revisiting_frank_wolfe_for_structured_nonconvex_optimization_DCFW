function [Xt, info] = FW_K(f,gradf, subgradg, lmo, maxit, tol, x0, C0, iter)
% Implementation of FW method from 
%
% [Khamaru, K. and Wainwright, M. J. Convergence guarantees for a class of non-convex and non-smooth
% optimization problems. Journal of Machine Learning Research, 20(154):1–52, 2019]

Xt = x0;

% initialize counters
counter_lmo = 0;
counter_gradf = 0;
counter_subgradg = 0;
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

    gap_DC_maximizer = calc_gap_DC(@(x) gradf(x),Xt,ut,C0,iter);
    counter_gradf = counter_gradf + iter;
    counter_subgradg = counter_subgradg + iter;

    Dt2 = Xt - gap_DC_maximizer;


    gap_DC = f(Xt) - f(gap_DC_maximizer) - ut(:)'*Dt2(:);

    info.t(t) = t;
    info.counter_lmo(t) = counter_lmo;
    info.counter_gradf(t) = counter_gradf;
    info.counter_subgradg(t) = counter_subgradg;
    info.gap(t) = gap;
    info.gap_DC(t) = gap_DC;
    info.time(t) = toc(timer);

    % Display progress every 10 iterations
    if mod(t, 100) == 0 || t == 1 || gap <= tol
        fprintf('Iteration %d: Gap = %.6f\n', t, gap);
    end

    if gap <= tol
        break;
    end
    
    % Khamaru-Wainwright original step-size
    eta = min(gap/C0, 1);

    Xt = Xt*(1-eta) + eta*St;

end

end

