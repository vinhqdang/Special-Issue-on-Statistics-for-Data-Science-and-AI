"""Two-phase (pool + label sampling) inference for PATTERN-DIF contrasts.

Target: the POPULATION contrast theta_pop = f(tau) where the pool is a simple random sample of N units from
an infinite population and tau = E t(Y) is the population mean of the per-unit total columns. The estimator is
the same plug-in difference estimator theta_hat = f(T_dif) of tcs.labeling_dif (f is homogeneous of degree
zero in (T, N), so f(T/N) = f(T)). With T_pool = sum_U t_i, G = grad f(T) (total scale),
    theta_hat - theta_pop = [f(T_dif) - f(T_pool)] + [f(T_pool) - f(N tau)]
the first bracket has conditional mean 0 given the pool and variance V_design (the existing estimand-on-the-pool
variance), the second is a sum of N iid influence values z_i = G^T (t_i(Y_i) - tau) and has variance V_pool =
N Var(z_i), uncorrelated with the first. Hence V_total = V_design + V_pool (Sarndal and Swensson 1987).
V_pool is estimated from the labelled sample, a stratified sample of the pool, with weights N_h / n_h:
    V_pool_hat = sum_{i in s} (N_h(i) / n_h(i)) (z_i - zbar)^2,   zbar = sum_{i in s} (N_h/n_h) z_i / N,
z_i = G_hat^T t_i(Y_i) the FULL-label influence value (not the residual used for V_design), G_hat the gradient
at T_dif.
"""
import numpy as np
from tcs.labeling_dif import DifContext, dif_totals, estimate_dif_all, Z95


def estimate_two_phase(strata, ids, Nh, nh, U, V, y_lab, idx_lab, est, p, floor=None, ctx=None, alpha_z=Z95):
    """Returns dict with theta, se_design (pool-conditional), se_pool, se_total, and the Wald interval
    bounds for both (lo_design, hi_design, lo_total, hi_total). NaN if a stratum is empty or a denominator
    total is not positive."""
    ctx = ctx or DifContext(strata, ids, U, V, p, est)
    ve = ctx.ve; nt = ve.nt; Nh = np.asarray(Nh)
    T, t, r, pos, cnt = dif_totals(strata, ids, Nh, idx_lab, y_lab, U, V, p, est, ctx)
    nan = np.full(nt, np.nan)
    if T is None or not ve.den_ok(T):
        return dict(theta=nan, se_design=nan, se_pool=nan, se_total=nan, lo_design=nan, hi_design=nan, lo_total=nan, hi_total=nan)
    theta, ses = estimate_dif_all(strata, ids, Nh, nh, U, V, y_lab, idx_lab, est, p, ctx)
    sd = ses[floor or "none"]
    G = ve.grad(T); z = t @ G.T                                          # full-label influence values (n, n_t)
    w = (Nh[pos] / np.maximum(cnt[pos], 1)).astype(float)                # N_h / n_h
    N = Nh.sum(); zbar = (w[:, None] * z).sum(0) / N
    vpool = (w[:, None] * (z - zbar) ** 2).sum(0)
    sp = np.sqrt(np.maximum(vpool, 0)); st = np.sqrt(sd ** 2 + vpool)
    return dict(theta=theta, se_design=sd, se_pool=sp, se_total=st, lo_design=theta - alpha_z * sd, hi_design=theta + alpha_z * sd,
                lo_total=theta - alpha_z * st, hi_total=theta + alpha_z * st)
