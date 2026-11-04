# Two-phase inference for the population contrast

Notation as in docs/notation_v2.md. Reference for the two-phase framework: Sarndal and Swensson (1987), "A general view of
estimation for two phases of selection with applications to two-phase sampling and nonresponse", Int. Stat. Rev. 55:279-294,
doi 10.2307/1403406.

## Set-up

Units (X_i, G_i, Y_i) are iid from an infinite deployment population. Phase 1: a simple random sample (the pool) of size N;
(X_i, G_i), hence the model predictions, the proxy pi_i and the strata, are observed for the whole pool. Phase 2: stratified
simple random sampling without replacement of n_h labels in stratum h of the pool (PATTERN or any label-free design).
Per-unit total columns t_i = t_i(Y_i) (so T = sum_i t_i), tau = E t_i the population mean. Every contrast is a smooth
function f of the totals that is homogeneous of degree zero in (T, N): f(T) = f~(T/N) (ratios; the error contrast uses
(E_m - E_j)/N). The population contrast is theta_pop = f~(tau); the pool contrast is theta_U = f(T_U), T_U = sum over the pool.
The estimator is unchanged: theta_hat = f(T_dif), T_dif the difference estimator of tcs.labeling_dif.

## Variance

theta_hat - theta_pop = [f(T_dif) - f(T_U)] + [f(T_U) - f(N tau)] = D + P.

D is the design error. E[D | pool] = 0 to first order (T_dif is exactly design-unbiased for T_U, and f is smooth) and
Var(D | pool) = V_design = sum_h N_h^2 (1 - n_h/N_h) S_h^2(e) / n_h, with e_i = G^T (t_i - f_i) the residual influence value,
G = grad f(T_U) (total scale), f_i the proxy mean of t_i. P is the pool error. A first-order expansion gives
P = sum_{i in pool} G*^T (t_i - tau) + o_p(N^{-1/2}) with G* = grad f(N tau), a sum of N iid terms. Since E[D | pool] = 0,
Cov(D, P) = 0 to first order and

    V_total = E[V_design] + V_pool,    V_pool = N Var_pop(z_i) = (1/N) Var_pop(g^T t_i),
    z_i = G*^T (t_i(Y_i) - tau)  (total-scale gradient),   g = N G* = grad f~(tau) (per-unit scale).

z_i is the FULL-label influence value of the plug-in ratio contrast (not the residual e_i of the difference estimator: the
pool term does not benefit from the proxy because the pool-to-population variation of Y is not removed by labelling).
Examples: for dERR, z_i = ((|yhat_mi - Y_i|) - |yhat_ji - Y_i|)/N - const, so V_pool = Var(loss_m - loss_j)/N. For a TPR
gap, z_i = sum_g s_g (1/P_g)(Y_i yhat-difference_i - TPR-difference_g Y_i) restricted to group g.
Relative size: write V_design = Var_SRS / (L Eff) with Var_SRS the variance of the SRS estimator of the full influence values
(of order N Var(z_i) up to the finite-population factor) and Eff the design efficiency against SRS. Then
rho = V_pool/V_design is of the order of L Eff / N, the quantity quoted in the referee report. If rho is small the pool term
is negligible; when it is of order one or larger the pool-conditional interval undercovers the population contrast:
with correct V_design and normal errors its coverage is about 2 Phi(1.96 / sqrt(1 + rho)) - 1 (0.93 at rho = 0.2,
0.83 at rho = 1, 0.67 at rho = 3).

## Estimator of V_pool and the interval

The labelled sample is a stratified sample of the pool with weights w_i = N_h/n_h (sum of weights = N). With G_hat = grad f(T_dif)
and z_i = G_hat^T t_i(Y_i) for labelled i:

    zbar = (1/N) sum_{i in s} w_i z_i,    V_pool_hat = sum_{i in s} w_i (z_i - zbar)^2 .

This is N times the weighted (plug-in) population variance of z over the pool, a consistent estimator of N Var(z_i) as N, n_h
grow (it is a ratio of stratified expansion estimators; its bias is O(1/n) relative). V_design_hat is the existing estimator
(sum_h N_h^2 (1 - n_h/N_h) s_h^2(e)/n_h, with or without floors). The two-phase 95% interval is

    theta_hat +/- 1.96 sqrt( V_design_hat + V_pool_hat ).

Strata sizes are conditional on the pool in V_design_hat; if strata sizes N_h are themselves random (they are, under pool
sampling) this is second order: the N_h enter V_design as pool quantities whose randomness is already inside P.

Implementation: tcs/labeling_two_phase.py (estimate_two_phase), test in tests/test_labeling_two_phase.py,
experiment in experiments/two_phase_check.py.
