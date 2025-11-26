function X = proj_spec_ball_sym(A, tau)
% Projection for symmetric A onto {X : ||X||_2 <= tau}.
    if nargin < 2 || isempty(tau), tau = 1; end
    [U,S,V] = svd(A,'econ');
    s = min(diag(S), tau);
    X = U*diag(s)*V';
end