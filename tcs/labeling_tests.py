"""Hypothesis tests, multiple testing, power and power-optimal design for the K contrasts of PATTERN-DIF.

Everything here sits on top of tcs.labeling_dif (difference estimator, residual influence values) and
tcs.labeling_two_phase (pool-to-population variance). Notation: theta_hat in R^K (K = 108 contrasts for M = 9
models), Sigma_hat its estimated covariance, se_k = sqrt(Sigma_hat_kk), R_hat = D^{-1} Sigma_hat D^{-1}.

1. Joint estimate. With G = grad f(T_hat) (K x n_c) the residual influence values of the difference estimator
   are e_i = G (t_i(Y_i) - f_i) (n x K) and, stratum by stratum,
       Sigma_hat = sum_h N_h^2 (1 - n_h/N_h) / n_h * S_h(e),   S_h(e) the K x K sample covariance in stratum h.
   The diagonal is exactly the variance of tcs.labeling_dif.estimate_dif_all. Floors ('zero', 'max') act on the
   diagonal as in labeling_dif; off-diagonal entries of a floored row keep the sample correlation where it is
   defined and use the proxy-model correlation otherwise (the proxy covariance G Sigma_h G^T is label free).
   kind='plugin' gives the expansion (plug-in) estimator with e_i = G t_i (as tcs.labeling.estimate) and is
   used for the SRS and PATTERN comparators. With two_phase=True the pool covariance
       Sigma_pool = sum_{i in s} (N_h/n_h) (z_i - zbar)(z_i - zbar)^T,   z_i = G t_i,
   is added (population target, tcs.labeling_two_phase).
2. Wald tests (two-sided, one-sided), non-inferiority (H0: theta >= margin) and TOST (H0: |theta| >= margin).
   A contrast with se = 0 (exactly estimated) is rejected iff theta_hat differs from the null value.
3. Multiple testing for a family of K hypotheses H_k: theta_k = theta0_k: unadjusted, Bonferroni, Holm (step
   down on the Wald p-values) and the single-step max-|t| test whose critical value is the (1 - alpha)-quantile
   of max_k |X_k|, X ~ N(0, R_hat), by Monte Carlo (20,000 draws from a fixed seed; exact normal quantile if
   K = 1 and exact Sidak value if R_hat = I). The same critical value gives simultaneous bands
   theta_hat +/- c se. All procedures control the FWER asymptotically in the strong sense if Sigma_hat is
   consistent; Holm and the single-step max-t dominate Bonferroni.
4. Power. Asymptotic power of the two-sided level-alpha Wald test against theta - theta0 = Delta with
   variance V: Phi(-z + |Delta|/sqrt V) + Phi(-z - |Delta|/sqrt V), z = z_{1-alpha/2} (z replaced by the
   max-t or Bonferroni critical value for simultaneous tests). With V(L) = c (1/L - 1/N) the label budget for
   a given power is available in closed form. The Pitman ARE of two designs is the ratio of the variances.
5. Power-optimal allocation: minimise max_k V_k(n) / delta_k^2 over n with sum_h n_h = L, lo_h <= n_h <= N_h,
   V_k(n) = sum_h C_hk (1/n_h - 1/N_h), C_hk = N_h^2 S_hk^2 (proxy-model variance of the residual influence
   values; 'plain' variances give the plug-in estimator). This is convex (max of convex functions). Lagrangian
   duality gives min_n max_w = max_w min_n over the simplex of weights w; for fixed w the inner minimiser is
   n_h = clip( sqrt(a_h) / lambda, lo_h, N_h ),  a_h = sum_k w_k C_hk / delta_k^2,  i.e. n_h proportional
   to N_h sqrt(sum_k w_k S_hk^2 / delta_k^2) (water-filling in lambda by bisection). The outer problem is
   concave in w and is solved by entropic mirror descent; every iterate gives a lower bound D(w) and an upper
   bound max_k V_k(n(w))/delta_k^2, so the duality gap certifies the solution. Optional SLSQP polish on the
   epigraph form is available (method='slsqp' or 'auto').
"""
import numpy as np
from functools import lru_cache
from scipy.stats import norm
from scipy.optimize import minimize, brentq
from tcs.labeling_dif import DifContext, dif_totals, _vec, proxy_residual_variance, Z95
from tcs.labeling_baselines import _proxy_influence

SE_TOL = 1e-10                                                         # se below this: contrast treated as exact


# =============================================================================================== 1. joint estimate
def _floor_cov(S, G, Sig_h, mode):
    """Floor a stratum covariance S (K x K), see module docstring."""
    s2 = np.diag(S).copy(); P = G @ Sig_h @ G.T; s2p = np.maximum(np.diag(P), 0)
    d = np.where(s2 > 1e-24, s2, s2p) if mode == "zero" else np.maximum(s2, s2p)
    sd, sp = np.sqrt(np.maximum(s2, 0)), np.sqrt(s2p)
    with np.errstate(divide="ignore", invalid="ignore"):
        CS = S / np.outer(sd, sd); CP = P / np.outer(sp, sp)
    okS = np.outer(sd > 1e-12, sd > 1e-12)
    C = np.where(okS, CS, np.where(np.isfinite(CP), CP, 0.0))
    C = np.clip(C, -1, 1); np.fill_diagonal(C, 1.0)
    return C * np.outer(np.sqrt(d), np.sqrt(d))


def joint_estimate(strata, ids, Nh, nh, U, V, y_lab, idx_lab, est, p, kind="dif", floor=None, ctx=None, two_phase=False):
    """theta_hat, Sigma_hat (K x K), SEs for one labelled sample. kind in ('dif', 'plugin'); floor in
    (None, 'zero', 'max') (dif only); two_phase adds the pool covariance. Returns a dict with theta, Sigma
    (design, pool-conditional), se_design, and if two_phase Sigma_pool, Sigma_total, se_total; None if a stratum
    has no label or a denominator total is not positive."""
    ctx = ctx or DifContext(strata, ids, U, V, p, est)
    ve = ctx.ve; K = ve.nt; Nh = np.asarray(Nh); H = len(ctx.ids)
    t = y_lab[:, None] * U[idx_lab] + (1 - y_lab[:, None]) * V[idx_lab]
    pos = ctx.pos[idx_lab]; cnt = np.bincount(pos, minlength=H)
    if (cnt == 0).any(): return None
    if kind == "dif":
        T, _, r, _, _ = dif_totals(strata, ids, Nh, idx_lab, y_lab, U, V, p, est, ctx)
    elif kind == "plugin":
        if floor: raise ValueError("floors are defined for the difference estimator only")
        mu = np.zeros((H, U.shape[1])); np.add.at(mu, pos, t); mu /= cnt[:, None]
        T = (Nh[:, None] * mu).sum(0); r = t
    else:
        raise ValueError(kind)
    if not ve.den_ok(T): return None
    theta = ve.value(T); G = ve.grad(T)
    e = r @ G.T
    Sig = np.zeros((K, K))
    for k in range(H):
        n_h, N_h = int(nh[k]), Nh[k]
        if n_h < 2 or n_h >= N_h: continue
        S = np.cov(e[pos == k], rowvar=False, ddof=1)
        if floor: S = _floor_cov(S, G, ctx.Sigma[k], floor)
        Sig += N_h ** 2 * (1 - n_h / N_h) / n_h * S
    out = dict(theta=theta, Sigma=Sig, se_design=np.sqrt(np.maximum(np.diag(Sig), 0)), T=T)
    if two_phase:
        z = t @ G.T; w = (Nh[pos] / cnt[pos]).astype(float); zc = z - (w[:, None] * z).sum(0) / Nh.sum()
        Sp = (zc * w[:, None]).T @ zc
        out["Sigma_pool"] = Sp; out["Sigma_total"] = Sig + Sp
        out["se_pool"] = np.sqrt(np.maximum(np.diag(Sp), 0)); out["se_total"] = np.sqrt(np.maximum(np.diag(Sig + Sp), 0))
    return out


def correlation(Sigma):
    """R_hat = D^{-1} Sigma D^{-1}; rows/columns of contrasts with se = 0 are returned as identity rows."""
    se = np.sqrt(np.maximum(np.diag(Sigma), 0)); ok = se > SE_TOL
    R = np.eye(len(se))
    R[np.ix_(ok, ok)] = Sigma[np.ix_(ok, ok)] / np.outer(se[ok], se[ok])
    return np.clip(R, -1, 1)


# =============================================================================================== 2. single tests
def wald_z(theta, se, theta0=0.0):
    theta = np.asarray(theta, float); se = np.asarray(se, float); d = theta - theta0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d / se
    exact = ~(se > SE_TOL)
    with np.errstate(invalid="ignore"):
        big = np.where(np.abs(d) <= 1e-9 * (1 + np.abs(theta0)), 0.0, np.sign(d) * np.inf)
    return np.where(exact, big, z)


def wald_pvalue(z, alternative="two-sided"):
    """p-value of the standardised statistic z = (theta_hat - theta0) / se. alternative: 'two-sided',
    'greater' (H1: theta > theta0), 'less' (H1: theta < theta0)."""
    z = np.asarray(z, float)
    if alternative == "two-sided": return 2 * norm.sf(np.abs(z))
    if alternative == "greater": return norm.sf(z)
    if alternative == "less": return norm.cdf(z)
    raise ValueError(alternative)


def wald_test(theta, se, theta0=0.0, alternative="two-sided"):
    z = wald_z(theta, se, theta0); return dict(z=z, p=wald_pvalue(z, alternative))


def noninferiority_test(theta, se, margin, sense="upper"):
    """Non-inferiority for a harm contrast theta (e.g. a difference of fairness gaps, smaller is better):
    sense='upper': H0: theta >= margin vs H1: theta < margin; sense='lower': H0: theta <= -margin vs H1: theta > -margin.
    margin may be a vector (one per contrast)."""
    if sense == "upper": return wald_test(theta, se, np.asarray(margin, float), "less")
    return wald_test(theta, se, -np.asarray(margin, float), "greater")


def equivalence_test(theta, se, margin):
    """TOST: H0: |theta| >= margin vs H1: |theta| < margin; p = max of the two one-sided p-values."""
    up = noninferiority_test(theta, se, margin, "upper"); lo = noninferiority_test(theta, se, margin, "lower")
    return dict(p=np.maximum(up["p"], lo["p"]), p_upper=up["p"], p_lower=lo["p"])


# =============================================================================================== 3. multiple testing
def adjust_pvalues(p, method):
    """Adjusted p-values: 'none', 'bonferroni', 'sidak', 'holm' (step-down; monotone)."""
    p = np.asarray(p, float); m = len(p)
    if method == "none": return p.copy()
    if method == "bonferroni": return np.minimum(1.0, m * p)
    if method == "sidak": return 1 - (1 - p) ** m
    if method == "holm":
        o = np.argsort(p, kind="stable"); adj = np.minimum(1.0, (m - np.arange(m)) * p[o])
        adj = np.maximum.accumulate(adj); out = np.empty(m); out[o] = adj; return out
    raise ValueError(method)


@lru_cache(maxsize=4)
def _normal_draws(K, ndraw, seed):
    return np.random.default_rng(seed).standard_normal((ndraw, K))


def _psd_factor(R):
    w, Q = np.linalg.eigh((R + R.T) / 2); B = Q * np.sqrt(np.maximum(w, 0))
    s = np.sqrt((B ** 2).sum(1)); return B / np.where(s > 0, s, 1)[:, None]        # unit-variance margins


def maxt_distribution(R, alternative="two-sided", ndraw=20000, seed=0):
    """Sorted Monte-Carlo sample of max_k |X_k| (or max_k X_k if one-sided), X ~ N(0, R)."""
    K = R.shape[0]; X = _normal_draws(K, ndraw, seed) @ _psd_factor(R).T
    m = np.abs(X).max(1) if alternative == "two-sided" else X.max(1)
    return np.sort(m)


def maxt_critical(R, alpha=0.05, alternative="two-sided", ndraw=20000, seed=0, exact=True):
    """Critical value of the single-step max-t test: (1 - alpha)-quantile of max |X|, X ~ N(0, R). K = 1 gives the
    normal quantile and R = I the Sidak value exactly (if exact=True); otherwise Monte Carlo (fixed seed)."""
    R = np.atleast_2d(R); K = R.shape[0]
    if exact and (K == 1 or np.allclose(R, np.eye(K), atol=1e-12)):
        q = (1 - alpha) ** (1.0 / K)
        return float(norm.ppf((1 + q) / 2) if alternative == "two-sided" else norm.ppf(q))
    s = maxt_distribution(R, alternative, ndraw, seed)
    return float(s[min(len(s) - 1, int(np.ceil((1 - alpha) * len(s))) - 1)])


def maxt_adjusted_pvalues(z, R, alternative="two-sided", ndraw=20000, seed=0):
    """P(max_k |X_k| >= |z_k|) under N(0, R) (one-sided: P(max X >= z_k)), Monte Carlo."""
    z = np.asarray(z, float); R = np.atleast_2d(R)
    if R.shape[0] == 1:
        return wald_pvalue(z, "two-sided" if alternative == "two-sided" else "greater")
    s = maxt_distribution(R, alternative, ndraw, seed); a = np.abs(z) if alternative == "two-sided" else z
    return (len(s) - np.searchsorted(s, a, side="left")) / len(s)


METHODS = ("none", "bonferroni", "holm", "maxt")


def simultaneous_test(theta, Sigma, theta0=0.0, alpha=0.05, alternative="two-sided", methods=METHODS, ndraw=20000, seed=0):
    """Test the family H_k: theta_k = theta0_k with familywise level alpha; alternative 'two-sided', 'greater' or
    'less'. Contrasts with se = 0 enter with p in {0, 1} and do not count towards the multiplicity.
    Returns {method: dict(reject (bool K), p_adj (K))} plus 'z', 'p', 'crit' (max-t critical value) and 'R'."""
    theta = np.asarray(theta, float); se = np.sqrt(np.maximum(np.diag(Sigma), 0)); K = len(theta)
    th0 = np.broadcast_to(np.asarray(theta0, float), theta.shape)
    z = wald_z(theta, se, th0); p = wald_pvalue(z, alternative)
    ok = np.isfinite(z) & (se > SE_TOL); R = correlation(Sigma); out = dict(z=z, p=p, R=R, crit=float("nan"))
    a = np.abs(z) if alternative == "two-sided" else (z if alternative == "greater" else -z)     # oriented statistic
    mt = "two-sided" if alternative == "two-sided" else "greater"
    for mth in methods:
        padj = p.copy()
        if ok.any():
            if mth == "maxt":
                Rok = R[np.ix_(ok, ok)]
                if Rok.shape[0] == 1: padj[ok] = p[ok]
                else:
                    s = maxt_distribution(Rok, mt, ndraw, seed)
                    padj[ok] = (len(s) - np.searchsorted(s, a[ok], side="left")) / len(s)
                    out["crit"] = float(s[min(len(s) - 1, int(np.ceil((1 - alpha) * len(s))) - 1)])
                    out["maxt"] = dict(reject=np.where(ok, a > out["crit"], p <= alpha), p_adj=padj); continue
                out["crit"] = maxt_critical(Rok, alpha, mt)
            else:
                padj[ok] = adjust_pvalues(p[ok], mth)
        out[mth] = dict(reject=padj <= alpha, p_adj=padj)
    return out


def simultaneous_bands(theta, Sigma, alpha=0.05, method="maxt", ndraw=20000, seed=0):
    """Simultaneous (1 - alpha) bands theta_hat +/- c se with c = max-t, Bonferroni or unadjusted critical value."""
    theta = np.asarray(theta, float); se = np.sqrt(np.maximum(np.diag(Sigma), 0)); ok = se > SE_TOL; m = max(int(ok.sum()), 1)
    if method == "none": c = norm.ppf(1 - alpha / 2)
    elif method == "bonferroni": c = norm.ppf(1 - alpha / (2 * m))
    elif method == "maxt": c = maxt_critical(correlation(Sigma)[np.ix_(ok, ok)] if ok.any() else np.eye(1), alpha, "two-sided", ndraw, seed)
    else: raise ValueError(method)
    return theta - c * se, theta + c * se, float(c)


# =============================================================================================== 4. power
def power_two_sided(delta, var, alpha=0.05, crit=None):
    """Asymptotic power of the two-sided Wald test: Phi(-c + |d|/sqrt V) + Phi(-c - |d|/sqrt V), c = z_{1-alpha/2}
    unless a critical value (Bonferroni, max-t) is given."""
    c = norm.ppf(1 - alpha / 2) if crit is None else crit
    s = np.abs(np.asarray(delta, float)) / np.sqrt(np.asarray(var, float))
    return norm.cdf(-c + s) + norm.cdf(-c - s)


def power_one_sided(delta, var, alpha=0.05, crit=None):
    c = norm.ppf(1 - alpha) if crit is None else crit
    return norm.cdf(-c + np.asarray(delta, float) / np.sqrt(np.asarray(var, float)))


def required_variance(delta, alpha=0.05, power=0.8, crit=None, two_sided=True):
    """Largest variance V with power >= `power` at effect |delta| (closed form for the one-sided and the main
    two-sided term, refined by root finding for both tails)."""
    c = (norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)) if crit is None else crit
    v0 = (abs(delta) / (c + norm.ppf(power))) ** 2
    if not two_sided: return v0
    f = lambda v: power_two_sided(delta, v, alpha, crit) - power
    return brentq(f, v0 * 0.5, v0 * 1.5 + 1e-300, xtol=1e-14 * v0) if f(v0 * 0.5) > 0 > f(v0 * 1.5) else v0


def budget_for_power(delta, c, alpha=0.05, power=0.8, N=None, crit=None, two_sided=True):
    """Label budget L with V(L) = c (1/L - 1/N) (finite-population form; N=None: V = c/L) and the target power."""
    v = required_variance(delta, alpha, power, crit, two_sided)
    return 1.0 / (v / c + (0.0 if N is None else 1.0 / N))


def design_effect(var_ref, var_new):
    """Pitman ARE of design `new` against `ref`: ratio of variances (= ratio of label budgets at equal power)."""
    return np.asarray(var_ref, float) / np.asarray(var_new, float)


def budget_from_curve(Ls, pw, target=0.8):
    """Smallest budget at which a simulated power curve reaches `target`: linear interpolation of power in log L
    between the first pair of grid points that straddle the target. NaN if the curve is above target at the first
    grid point or never reaches it (censored)."""
    Ls = np.asarray(Ls, float); pw = np.asarray(pw, float)
    if pw[0] >= target or pw.max() < target: return float("nan")
    i = int(np.argmax(pw >= target)); x0, x1 = np.log(Ls[i - 1]), np.log(Ls[i]); y0, y1 = pw[i - 1], pw[i]
    return float(np.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))


# =============================================================================================== 5. power-optimal design
def design_variance_matrix(strata, ids, U, V, p, est, kind="resid"):
    """(N_h, C) with C_hk = N_h^2 S_hk^2: proxy-model variance coefficient of target k in stratum h, so that
    Var_k(n) = sum_h C_hk (1/n_h - 1/N_h). kind='resid' is the difference estimator (residual influence values),
    kind='plain' the plug-in estimator (influence values with y ~ Bernoulli(pi))."""
    ids = np.asarray(ids); Nh = np.array([(strata == h).sum() for h in ids])
    if kind == "resid": S2, _ = proxy_residual_variance(strata, ids, U, V, p, est)
    elif kind == "plain":
        A, B, Ez, Ez2, _ = _proxy_influence(U, V, p, _vec(est))
        S2 = np.array([np.maximum(Ez2[strata == h].mean(0) - Ez[strata == h].mean(0) ** 2, 0) for h in ids])
    else: raise ValueError(kind)
    return Nh, Nh[:, None] ** 2 * S2


def predicted_variance(n, Nh, C):
    n = np.asarray(n, float); return ((1.0 / n - 1.0 / Nh)[:, None] * C).sum(0)


def _inner_alloc(a, Nh, lo, L):
    """argmin sum_h a_h / n_h  s.t.  sum n_h = L,  lo_h <= n_h <= N_h  (continuous): n_h = clip(sqrt(a_h)/lam, lo_h, N_h)."""
    if L >= Nh.sum(): return Nh.astype(float)
    s = np.sqrt(np.maximum(a, 0)); lo = lo.astype(float); hi = Nh.astype(float)
    f = lambda lam: np.clip(s / lam, lo, hi).sum() - L
    if lo.sum() >= L: return lo
    a_, b_ = 1e-12, max(s.max(), 1e-12) / (L / len(s)) * 1e3 + 1.0
    while f(b_) > 0: b_ *= 10
    for _ in range(200):
        mid = np.sqrt(a_ * b_); (a_, b_) = (mid, b_) if f(mid) > 0 else (a_, mid)
        if b_ / a_ < 1 + 1e-13: break
    return np.clip(s / np.sqrt(a_ * b_), lo, hi)


def round_to_budget(nc, Nh, lo, L):
    n = np.clip(np.floor(nc + 1e-9).astype(int), lo, Nh); short = int(L - n.sum())
    order = np.argsort(-(nc - np.floor(nc)))
    while short > 0:                                                  # largest remainders first, respecting caps
        moved = False
        for h in order:
            if short <= 0: break
            if n[h] < Nh[h]: n[h] += 1; short -= 1; moved = True
        if not moved: break
    while short < 0:                                                  # only possible if floors were mis-rounded
        h = int(np.argmax(np.where(n > lo, n, -1))); n[h] -= 1; short += 1
    return n


def power_optimal_allocation(Nh, C, delta, L, floor=10, method="auto", iters=3000, tol=1e-3, eta=2.0):
    """Maximin-power allocation: minimise max_k V_k(n) / delta_k^2, V_k(n) = sum_h C_hk (1/n_h - 1/N_h), subject to
    sum n_h = L and min(N_h, floor) <= n_h <= N_h. delta_k = inf (or nan) drops target k.
    method: 'mirror' (dual weights, certified gap), 'slsqp' (epigraph form), 'auto' (mirror, then SLSQP if the gap
    exceeds tol). Returns dict(n (int), n_cont, w (simplex weights over targets), value_primal, value_dual, gap,
    value_int (max_k V_k at the integer allocation / delta_k^2))."""
    Nh = np.asarray(Nh); delta = np.asarray(delta, float); act = np.isfinite(delta) & (delta > 0)
    Cs = C[:, act] / delta[act] ** 2; lo = np.minimum(Nh, floor); H, Ka = Cs.shape
    if Ka == 0: raise ValueError("no active targets")
    val = lambda n: ((1.0 / n - 1.0 / Nh)[:, None] * Cs).sum(0)
    w = np.full(Ka, 1.0 / Ka); best_p, best_n, best_d, best_w = np.inf, None, -np.inf, w.copy(); nbar = np.zeros(H); wsum = 0.0
    if method in ("mirror", "auto"):
        for t in range(1, iters + 1):
            n = _inner_alloc(Cs @ w, Nh, lo, L); v = val(n)
            d = float(w @ v)
            if d > best_d: best_d, best_w = d, w.copy()
            pv = float(v.max())
            if pv < best_p: best_p, best_n = pv, n.copy()
            nbar = (nbar * wsum + n * t) / (wsum + t); wsum += t                 # linearly weighted ergodic average
            if t % 50 == 0:
                pb = float(val(np.clip(nbar, lo, Nh)).max())
                if pb < best_p: best_p, best_n = pb, nbar.copy()
            if best_p <= best_d * (1 + tol * 0.3) + 1e-300: break
            g = v / max(v.max(), 1e-300); w = w * np.exp(eta / np.sqrt(t) * (g - 1)); w = np.maximum(w, 1e-300); w /= w.sum()
    n_cont = best_n
    if method == "slsqp" or (method == "auto" and best_p > best_d * (1 + tol) + 1e-300):
        x0 = np.append(best_n if best_n is not None else _inner_alloc(Cs.mean(1), Nh, lo, L), 0.0)
        sc = float(val(x0[:-1]).max()); x0[-1] = 1.0
        cons = [dict(type="eq", fun=lambda x: x[:-1].sum() - L, jac=lambda x: np.append(np.ones(H), 0.0)),
                dict(type="ineq", fun=lambda x: x[-1] - val(x[:-1]) / sc,
                     jac=lambda x: np.hstack([(Cs / x[:-1, None] ** 2).T / sc, np.ones((Ka, 1))]))]
        r = minimize(lambda x: x[-1], x0, jac=lambda x: np.append(np.zeros(H), 1.0), bounds=[(l, h) for l, h in zip(lo, Nh)] + [(0, None)],
                     constraints=cons, method="SLSQP", options=dict(maxiter=500, ftol=1e-12))
        nn = np.clip(r.x[:-1], lo, Nh)
        if abs(nn.sum() - L) > 1e-6: nn = _inner_alloc(nn ** 2, Nh, lo, L)                  # repair feasibility
        if val(nn).max() < best_p or n_cont is None: n_cont, best_p = nn, float(val(nn).max())
    n_int = round_to_budget(n_cont, Nh, lo, L)
    return dict(n=n_int, n_cont=n_cont, w=best_w, value_primal=float(best_p), value_dual=float(best_d),
                gap=float(best_p / best_d - 1) if best_d > 0 else float("nan"), value_int=float(val(n_int).max()), active=act)
