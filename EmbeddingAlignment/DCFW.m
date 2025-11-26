function [Xt, info] = DCFW(f,gradf, subgradg, lmo, maxit, tol_outer, x0, C0, iter)
%Implementation of DCFW

Xt = x0;
tk = 0;
exitflag = 0;
beta = 0.8;

% initialize counters
counter_lmo = 0;
counter_gradf = 0;
counter_subgradg = 0;
timer = tic;

Gt = subgradg(Xt);
counter_subgradg = counter_subgradg + 1;

Ft = gradf(Xt);
counter_gradf = counter_gradf + 1;
Ht = Ft - Gt;

St = lmo(Ht);
counter_lmo = counter_lmo + 1;

Dt = St - Xt;
gap = - Ht(:)'*Dt(:);

tol_inner = beta * gap;


for t = 1:maxit

    % Gt = subgradg(Xt);    % We already compute this in the previous
    % iteration to compute gap, so we can reuse it

    for k = 1:1e7

        tk = tk + 1; % increment the total iteration counter

        if k == 1
            Xtk = Xt;
            Htk = Ht;
            Stk = St;
            Dtk = Dt;
            Ftk = Ft;
            gap_tk = gap;
        else
            Ftk = gradf(Xtk);
            counter_gradf = counter_gradf + 1;
            Htk = Ftk - Gt;

            Stk = lmo(Htk);
            counter_lmo = counter_lmo + 1;

            Dtk = Stk - Xtk;
            gap_tk = - Htk(:)'*Dtk(:);
        end

        info.inner.t(tk) = t;
        info.inner.k(tk) = k;
        info.inner.counter_lmo(tk) = counter_lmo;
        info.inner.counter_gradf(tk) = counter_gradf;
        info.inner.counter_subgradg(tk) = counter_subgradg;
        info.inner.gap_tk(tk) = gap_tk;
        info.inner.gap_tk_DC(t) = gap_tk;
        info.inner.time(tk) = toc(timer);

        % Display progress every t iteration
        %         if mod(tk, 100) == 0
        %             fprintf('t: %d, tk: %d, gap: %.6f, grad: %d, lmo: %d, subgrad: %d\n', ...
        %                 t, tk, min(info.gap_tk), counter_gradf, counter_lmo, counter_subgradg);
        %         end

        if tk >= maxit
            exitflag = 1;
            break;
        end

        eta = 2/(tk+1);


        Xtk = Xtk + eta*Dtk;

        %         if gaptk <= gap0/t
        if gap_tk <= tol_inner
            break;
        end

    end


    Xt = Xtk;
    Gt = subgradg(Xt);
    counter_subgradg = counter_subgradg + 1;

    Ft = gradf(Xt);
    counter_gradf = counter_gradf + 1;
    Ht = Ft - Gt;

    St = lmo(Ht);
    counter_lmo = counter_lmo + 1;

    Dt = St - Xt;
    gap = - Ht(:)'*Dt(:);

    gap_DC_maximizer = calc_gap_DC(@(x) gradf(x),Xt,Gt,C0,iter);
    counter_gradf = counter_gradf + iter;
    counter_subgradg = counter_subgradg + iter;

    Dt2 = Xt - gap_DC_maximizer;
    gap_DC = f(Xt) - f(gap_DC_maximizer) - Gt(:)'*Dt2(:);

    info.outer.counter_lmo(t) = counter_lmo;
    info.outer.counter_gradf(t) = counter_gradf;
    info.outer.counter_subgradg(t) = counter_subgradg;
    info.outer.gap(t) = gap;
    info.outer.gap_DC(t) = gap_DC;
    info.outer.tk(t) = tk;
    info.outer.time(t) = toc(timer);

    % Display progress every t iteration
    if mod(t, 1) == 0
        fprintf('t: %d, tk: %d, gap: %.6f, grad: %d, lmo: %d, subgrad: %d\n', ...
            t, tk, gap, counter_gradf, counter_lmo, counter_subgradg);
    end

    if exitflag
        break;
    end

    if gap <= tol_outer
        exitflag = 1;
        break;
    end

    if gap < tol_inner
        tol_inner = beta*tol_inner;
    end


end

end

