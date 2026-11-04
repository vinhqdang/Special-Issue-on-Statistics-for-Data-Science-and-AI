"""Helpers for the ablation study of PATTERN-DIF (experiments/label_ablations_dif.py).

Builds on tcs.labeling_dif (difference estimator, allocations, variance floors) and
tcs.labeling_variants (stratifiers, proxies, EO gaps); nothing there is changed.

  * allocate_dif        allocation with the budget-aware floor of tcs.labeling_variants.effective_floor
  * ht_totals           stratified expansion (Horvitz-Thompson) totals from a labelled sample
  * dif_inference       difference-estimator estimate with Wald standard errors (plain, floor 'zero',
                        floor 'max') and Satterthwaite degrees of freedom for each of them
  * dif_bootstrap       stratified bootstrap percentile interval of the difference estimator: units are
                        resampled within strata together with their residuals t_i - f_i, and
                        T* = sum_U f_i + sum_h N_h mean*_h (t_i - f_i), so that the proxy part is held fixed
"""
import numpy as np
from tcs.labeling import proportional_allocation
from tcs.labeling_dif import dif_totals, dif_allocation
from tcs.labeling_variants import effective_floor


def allocate_dif(strata, U, V, p, est, L, min_n=10, kind="plain"):
    """kind in ('plain', 'resid', 'prop'). Returns (ids, N_h, n_h, floor used)."""
    H = len(np.unique(strata)); mn = effective_floor(H, L, min_n)
    if kind == "prop":
        return (*proportional_allocation(strata, L, mn), mn)
    return (*dif_allocation(strata, U, V, p, est, L, kind, mn), mn)


def ht_totals(ctx, Nh, idx_lab, y_lab, U, V):
    """T_hat = sum_h N_h mean_{s_h} t_i; None if a stratum has no labelled unit. ctx: DifContext."""
    t = y_lab[:, None] * U[idx_lab] + (1 - y_lab[:, None]) * V[idx_lab]
    pos = ctx.pos[idx_lab]; H = len(ctx.ids); cnt = np.bincount(pos, minlength=H)
    if (cnt == 0).any(): return None
    tbar = np.zeros((H, U.shape[1])); np.add.at(tbar, pos, t); tbar /= cnt[:, None]
    return (np.asarray(Nh)[:, None] * tbar).sum(0)


def dif_inference(ctx, Nh, nh, U, V, y_lab, idx_lab, est, p):
    """theta, totals T, and for each variance version v in ('none', 'zero', 'max') the standard error and
    the Satterthwaite degrees of freedom. 'zero' / 'max' are the proxy variance floors of tcs.labeling_dif.
    Returns None if a stratum is empty or a denominator total is not positive."""
    ve = ctx.ve; nt = ve.nt; Nh = np.asarray(Nh)
    T, t, r, pos, cnt = dif_totals(None, ctx.ids, Nh, idx_lab, y_lab, U, V, p, est, ctx)
    if T is None or not ve.den_ok(T): return None
    theta = ve.value(T); G = ve.grad(T)
    e = r @ G.T
    H = len(ctx.ids)
    vh = {k: np.zeros((H, nt)) for k in ("none", "zero", "max")}; dfh = np.zeros(H)
    for k in range(H):
        n_h, N_h = int(nh[k]), Nh[k]
        if n_h < 2 or n_h >= N_h: continue
        s2 = e[pos == k].var(0, ddof=1)
        s2p = np.maximum(np.einsum("tc,cd,td->t", G, ctx.Sigma[k], G), 0)
        c = N_h ** 2 * (1 - n_h / N_h) / n_h
        vh["none"][k] = c * s2
        vh["zero"][k] = c * np.where(s2 > 1e-24, s2, s2p)
        vh["max"][k] = c * np.maximum(s2, s2p)
        dfh[k] = n_h - 1
    out = {"theta": theta, "T": T}
    for k, v in vh.items():
        var = v.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = np.where(dfh[:, None] > 0, v ** 2 / np.maximum(dfh[:, None], 1), 0).sum(0)
            df = np.where(den > 0, var ** 2 / den, np.nan)
        out["se_" + k] = np.sqrt(var); out["df_" + k] = df
    return out


def dif_bootstrap(ctx, Nh, U, V, y_lab, idx_lab, est, rng, B=999, level=0.95):
    """Stratified bootstrap percentile interval for the plug-in difference estimator."""
    f = ctx.f
    t = y_lab[:, None] * U[idx_lab] + (1 - y_lab[:, None]) * V[idx_lab]
    r = t - f[idx_lab]; pos = ctx.pos[idx_lab]
    Tb = np.tile(ctx.F.sum(0), (B, 1))
    for k, N_h in enumerate(np.asarray(Nh)):
        rh = r[pos == k]; n = len(rh)
        if n == 0: continue
        w = rng.multinomial(n, np.full(n, 1 / n), size=B) / n
        Tb += N_h * (w @ rh)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = est.value(Tb.T).T
    vals = np.where(np.isfinite(vals), vals, np.nan)
    a = (1 - level) / 2
    return np.nanquantile(vals, a, axis=0), np.nanquantile(vals, 1 - a, axis=0)
