function [Xt] = calc_gap_DC(gradf,Xt,ut,C0,iter)
% Calculating the DC Gap

t = 1;
while t<=iter

    Xt = proj_spec_ball_sym(Xt - 1/C0 * (gradf(Xt) - ut), 1);
    t = t + 1;
    
end
end