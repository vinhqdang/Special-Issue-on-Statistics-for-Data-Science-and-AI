"""PATTERN-DIF: stratified labelling with a model-assisted difference estimator.

Setting and notation as in tcs.labeling (totals T_c = sum_i t_ic(Y_i), t_ic(y) = y U_ic + (1-y) V_ic,
estimands theta = f(T)). The proxy mean of t_i is
    f_i = pi_i U_i + (1 - pi_i) V_i,        pi_i = mean predicted probability (label free).

Difference estimator of a total under stratified sampling without replacement (strata h, sizes N_h,
labelled samples s_h of size n_h):
    T_dif = sum_{i in U} f_i + sum_h N_h mean_{i in s_h} ( t_i(Y_i) - f_i ).
It is design-unbiased for every label-free proxy f (the second term is the stratified expansion
estimator of sum_U (t_i - f_i), and sum_U f_i is a constant); a poor proxy only costs efficiency.
Estimands: theta_hat = f(T_dif) (plug-in). Linearisation with G = grad f(T_dif): the residual influence
value is e_i = G (t_i(Y_i) - f_i), and
    V_hat = sum_h N_h^2 (1 - n_h/N_h) s_h^2(e) / n_h,
with s_h^2 the sample variance of e within stratum h. With one stratum this is PPI (lambda = 1)
under simple random sampling, with strata = group x #models predicting 1 it is PATTERN-DIF.

Allocations (all label free; same API as tcs.labeling.neyman_allocation):
  'plain'  the PATTERN allocation: n_h proportional to N_h sqrt( sum_k w_k S_hk^2 ), S_hk^2 the proxy
           variance of the influence values z_ik = G_k t_i, w_k = 1 / S_k^2 the inverse overall proxy
           variance (so each target is weighted by its variance under SRS). It is optimal for the
           plug-in (expansion) estimator.
  'resid'  the same rule applied to the variance that the difference estimator actually has. The
           residual of target k is e_ik = z_ik - E_pi z_ik = (Y_i - pi_i)(A_ik - B_ik), where A_ik, B_ik
           are the influence values of unit i if Y_i = 1 and if Y_i = 0. Under the proxy model
           Y_i ~ Bernoulli(pi_i) it has mean 0 and variance pi_i (1 - pi_i)(A_ik - B_ik)^2, so the
           within-stratum variance of the residuals that enters V_hat is estimated by
           S_hk^2(e) = mean_{i in h} pi_i (1 - pi_i)(A_ik - B_ik)^2 (closed form, no simulation), and
           n_h proportional to N_h sqrt( sum_k w_k S_hk^2(e) ) minimises sum_k w_k Var_k(T_dif)
           for fixed sum n_h (generalised Neyman). The weights w_k stay those of the plain influence
           values, 1 / S_k^2, so that 'plain' and 'resid' weigh targets identically and differ only in
           the stratum scores. This coincides with tcs.labeling_baselines.stratppi_allocation applied
           to the given strata.

Zero-variance guard ('floor'). A stratum whose sample variance of e_i is exactly 0 (typical for
small strata in which all proxy-predicted outcomes are matched) contributes 0 to V_hat and makes the
interval too short. floor='zero' replaces such a stratum's s_h^2 by its proxy residual variance
S_hk^2(e) above (label free, cf. the 'proxy variance floor' in tcs.labeling_variants.estimate_full);
floor='max' uses max(s_h^2, S_hk^2(e)) in every stratum (conservative).
"""
import numpy as np
from tcs.labeling import neyman_allocation, _round_alloc, draw          # noqa: F401  (draw re-exported)
from tcs.labeling_baselines import VecEstimands, _proxy_influence

Z95 = 1.959964
ALLOCATIONS = ("plain", "resid")


def _vec(est):
    if isinstance(est, VecEstimands): return est
    ve = getattr(est, "_vec_cache", None)
    if ve is None:
        ve = VecEstimands(est); est._vec_cache = ve
    return ve


# ---------------------------------------------------------------- allocation
def proxy_residual_variance(strata, ids, U, V, p, est):
    """Per stratum and target: S_hk^2(e) = mean_{i in h} pi_i (1-pi_i) (A_ik - B_ik)^2 (shape (H, n_t)),
    and the target weights w_k = 1 / S_k^2 of the plain influence values."""
    A, B, Ez, Ez2, w = _proxy_influence(U, V, p, _vec(est))
    r2 = (p * (1 - p))[:, None] * (A - B) ** 2
    S2 = np.array([r2[strata == h].mean(0) for h in ids])
    return S2, w


def dif_allocation(strata, U, V, p, est, L, kind="plain", min_n=10):
    """Allocation of L labels over strata. kind in ('plain', 'resid'), see module docstring.
    Returns (ids, N_h, n_h) as tcs.labeling.neyman_allocation."""
    if kind == "plain":
        return neyman_allocation(strata, U, V, p, est, L, min_n)
    if kind == "resid":
        ids = np.unique(strata); Nh = np.array([(strata == h).sum() for h in ids])
        S2, w = proxy_residual_variance(strata, ids, U, V, p, est)
        score = np.sqrt((S2 * w).sum(1))
        return ids, Nh, _round_alloc(Nh * score, Nh, L, min_n)
    raise ValueError(f"unknown allocation kind {kind!r}")


# ---------------------------------------------------------------- estimation
class DifContext:
    """Label-free quantities reused across replications for one stratification: proxy means f_i, their
    stratum sums F_h, the position of each unit's stratum, and per-stratum proxy covariance matrices
    Sigma_h = mean_{i in h} pi_i (1-pi_i) (u_i - v_i)(u_i - v_i)^T of the residual totals."""

    def __init__(self, strata, ids, U, V, p, est):
        self.ve = _vec(est); self.ids = np.asarray(ids)
        self.f = p[:, None] * U + (1 - p[:, None]) * V
        self.pos = np.searchsorted(self.ids, strata)
        H = len(self.ids)
        self.F = np.array([self.f[self.pos == k].sum(0) for k in range(H)])
        d = U - V; w = (p * (1 - p))[:, None] * d
        self.Sigma = [(w[self.pos == k].T @ d[self.pos == k]) / max((self.pos == k).sum(), 1) for k in range(H)]


def dif_totals(strata, ids, Nh, idx_lab, y_lab, U, V, p, est=None, ctx=None):
    """T_dif = sum_U f_i + sum_h N_h mean_{s_h}(t_i - f_i). Returns (T or None if a stratum has no label,
    t, residuals t - f, stratum position, stratum sample sizes)."""
    ctx = ctx or DifContext(strata, ids, U, V, p, est)
    t = y_lab[:, None] * U[idx_lab] + (1 - y_lab[:, None]) * V[idx_lab]
    r = t - ctx.f[idx_lab]; pos = ctx.pos[idx_lab]; H = len(ctx.ids)
    cnt = np.bincount(pos, minlength=H)
    if (cnt == 0).any(): return None, t, r, pos, cnt
    rbar = np.zeros((H, U.shape[1])); np.add.at(rbar, pos, r); rbar /= cnt[:, None]
    return ctx.F.sum(0) + (np.asarray(Nh)[:, None] * rbar).sum(0), t, r, pos, cnt


def _nan(nt):
    x = np.full(nt, np.nan); return x, x, x, x


def estimate_dif_all(strata, ids, Nh, nh, U, V, y_lab, idx_lab, est, p, ctx=None):
    """Difference-estimator estimates and standard errors for one labelled sample.
    Returns (theta, {'none': se, 'zero': se_floor_zero, 'max': se_floor_max}); NaN arrays if a stratum is
    empty or a denominator total is not positive."""
    ctx = ctx or DifContext(strata, ids, U, V, p, est)
    ve = ctx.ve; nt = ve.nt; Nh = np.asarray(Nh)
    T, t, r, pos, cnt = dif_totals(strata, ids, Nh, idx_lab, y_lab, U, V, p, est, ctx)
    if T is None or not ve.den_ok(T):
        x = np.full(nt, np.nan); return x, {"none": x, "zero": x, "max": x}
    theta = ve.value(T); G = ve.grad(T)
    e = r @ G.T                                                          # residual influence values (n, n_t)
    v = {k: np.zeros(nt) for k in ("none", "zero", "max")}
    for k in range(len(ctx.ids)):
        n_h, N_h = int(nh[k]), Nh[k]
        if n_h < 2 or n_h >= N_h: continue
        s2 = e[pos == k].var(0, ddof=1)
        s2p = np.maximum(np.einsum("tc,cd,td->t", G, ctx.Sigma[k], G), 0)
        c = N_h ** 2 * (1 - n_h / N_h) / n_h
        v["none"] += c * s2
        v["zero"] += c * np.where(s2 > 1e-24, s2, s2p)
        v["max"] += c * np.maximum(s2, s2p)
    return theta, {k: np.sqrt(x) for k, x in v.items()}


def estimate_dif(strata, ids, Nh, nh, U, V, y_lab, idx_lab, est, p, floor=None, ctx=None, alpha_z=Z95):
    """Same return shape as tcs.labeling.estimate: (theta, se, lo, hi). floor in (None, 'zero', 'max')."""
    theta, ses = estimate_dif_all(strata, ids, Nh, nh, U, V, y_lab, idx_lab, est, p, ctx)
    se = ses[floor or "none"]
    return theta, se, theta - alpha_z * se, theta + alpha_z * se
