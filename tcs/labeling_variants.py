"""Variants of the stratified labelling design used in the ablation and robustness studies.

Everything here builds on tcs.labeling (which is left unchanged):
  * alternative stratifiers (binary agree/disagree, joint pattern of three label-free chosen models,
    K-count crossed with proxy bins);
  * alternative proxies for the allocation (single model, constant, corrupted, true labels);
  * allocation with a budget-aware floor (the default floor of 10 labels per stratum can exceed L
    when strata are many);
  * inference variants: Satterthwaite degrees of freedom for t intervals, stratified bootstrap
    percentile intervals;
  * per-model group gaps and the equalised-odds gap EO_m = max(|dTPR_m|, |dFPR_m|) from totals.
"""
import numpy as np
from tcs.labeling import collapse_small, neyman_allocation, proportional_allocation, _round_alloc


# ---------------------------------------------------------------- stratifiers
def strata_kcount(pred, g, min_size=50):
    """Default: group x K (number of models predicting 1), small strata merged within group."""
    k = pred.sum(1).astype(int)
    return collapse_small(g.astype(int) * (pred.shape[1] + 1) + k, g.astype(int), min_size)


def strata_binary(pred, g, min_size=50):
    """Group x {all models agree, at least one disagrees}."""
    k = pred.sum(1).astype(int); M = pred.shape[1]
    dis = ((k > 0) & (k < M)).astype(int)
    return collapse_small(g.astype(int) * 2 + dis, g.astype(int), min_size)


def top_agreement_models(pred, p, k=3):
    """The k models whose predictions agree most often with the thresholded ensemble mean (no labels)."""
    ens = (p.mean(1) >= 0.5).astype(int) if p.ndim == 2 else (p >= 0.5).astype(int)
    agree = (pred == ens[:, None]).mean(0)
    return [int(m) for m in np.argsort(-agree, kind="stable")[:k]]


def strata_joint(pred, g, models, min_size=50):
    """Group x joint prediction pattern of the given models. Patterns are ordered by
    (number of 1s, binary code) so that merging of small strata goes to a 'neighbouring' pattern."""
    sub = pred[:, models].astype(int); k = len(models)
    code = sub @ (2 ** np.arange(k))
    order = sorted(range(2 ** k), key=lambda c: (bin(c).count("1"), c))
    rank = np.empty(2 ** k, int); rank[order] = np.arange(2 ** k)
    return collapse_small(g.astype(int) * 2 ** k + rank[code], g.astype(int), min_size)


def strata_kproxy(pred, g, pi, nbins=2, min_size=50):
    """Group x K x (quantile bin of the proxy within the (group, K) cell)."""
    k = pred.sum(1).astype(int); M = pred.shape[1]; g = g.astype(int)
    cell = g * (M + 1) + k
    b = np.zeros(len(k), int)
    for c in np.unique(cell):
        sel = np.flatnonzero(cell == c)
        if len(sel) < nbins: continue
        r = np.argsort(np.argsort(pi[sel], kind="stable"), kind="stable")
        b[sel] = (r * nbins) // len(sel)
    return collapse_small(cell * nbins + b, g, min_size)


# ---------------------------------------------------------------- proxies (all label-free except 'oracle')
def proxies(p_models, g, y, rng, single=(0, 7), noise_sd=2.0):
    """Dictionary of allocation proxies pi_i. p_models: (N, M) predicted probabilities."""
    ens = p_models.mean(1)
    out = {"ens": ens, "const": np.full(len(ens), 0.5)}
    for m in single:
        out[f"single{m}"] = p_models[:, m]
    sh = ens.copy()
    for gg in np.unique(g):
        sel = np.flatnonzero(g == gg); sh[sel] = ens[rng.permutation(sel)]
    out["shuffle"] = sh
    lo = np.log(np.clip(ens, 1e-4, 1 - 1e-4) / (1 - np.clip(ens, 1e-4, 1 - 1e-4)))
    out["noise"] = 1 / (1 + np.exp(-(lo + noise_sd * rng.standard_normal(len(ens)))))
    out["flip"] = 1 - ens
    out["oracle"] = y.astype(float)
    return out


# ---------------------------------------------------------------- allocation with budget-aware floor
def effective_floor(H, L, min_n, share=0.8):
    """Floor per stratum: min_n unless H*min_n would use more than `share` of the budget."""
    return int(min_n) if H * min_n <= share * L else max(2, int(share * L // H))


def allocate(strata, U, V, p, est, L, min_n=10, kind="neyman"):
    H = len(np.unique(strata)); mn = effective_floor(H, L, min_n)
    if kind == "prop":
        ids, Nh, nh = proportional_allocation(strata, L, mn)
    else:
        ids, Nh, nh = neyman_allocation(strata, U, V, p, est, L, mn)
    return ids, Nh, nh, mn


# ---------------------------------------------------------------- estimation with t / bootstrap
def stratum_means(strata, ids, t, h_of):
    return np.array([t[h_of == h].mean(0) if (h_of == h).any() else np.zeros(t.shape[1]) for h in ids])


def estimate_full(strata, ids, Nh, nh, U, V, y_lab, idx_lab, est):
    """theta, linearisation SE, Satterthwaite df, estimated totals T (NaNs if a denominator is 0)."""
    t = y_lab[:, None] * U[idx_lab] + (1 - y_lab[:, None]) * V[idx_lab]
    h_of = strata[idx_lab]
    T = (Nh[:, None] * stratum_means(strata, ids, t, h_of)).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = est.value(T); G = est.gradient(T)
    nt = len(theta)
    if not np.all(np.isfinite(theta)):
        nan = np.full(nt, np.nan); return nan, nan, nan, T
    z = t @ G.T
    vh = np.zeros((len(ids), nt)); dfh = np.zeros(len(ids))
    for k, (h, N_h, n_h) in enumerate(zip(ids, Nh, nh)):
        sel = h_of == h
        if n_h >= 2 and n_h < N_h:
            vh[k] = N_h ** 2 * (1 - n_h / N_h) * z[sel].var(0, ddof=1) / n_h; dfh[k] = n_h - 1
    var = vh.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(dfh[:, None] > 0, vh ** 2 / np.maximum(dfh[:, None], 1), 0).sum(0)
        df = np.where(den > 0, var ** 2 / den, np.nan)
    return theta, np.sqrt(var), df, T


def bootstrap_percentile(strata, ids, Nh, U, V, y_lab, idx_lab, est, rng, B=999, level=0.95):
    """Stratified (within-stratum, with replacement) bootstrap percentile interval for est.value."""
    t = y_lab[:, None] * U[idx_lab] + (1 - y_lab[:, None]) * V[idx_lab]
    h_of = strata[idx_lab]
    Tb = np.zeros((B, U.shape[1]))
    for h, N_h in zip(ids, Nh):
        th = t[h_of == h]; n = len(th)
        if n == 0: continue
        w = rng.multinomial(n, np.full(n, 1 / n), size=B) / n          # (B, n)
        Tb += N_h * (w @ th)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = est.value(Tb.T).T                                          # (B, n_t)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    a = (1 - level) / 2
    return np.nanquantile(vals, a, axis=0), np.nanquantile(vals, 1 - a, axis=0)


# ---------------------------------------------------------------- per-model gaps and EO
def model_gaps(T, ix, M):
    """Return (dTPR_m, dFPR_m) arrays of length M from totals (ix: name -> column)."""
    tpr = lambda m, gg: T[ix[("TP", m, gg)]] / T[ix[("P", gg)]]
    fpr = lambda m, gg: T[ix[("FP", m, gg)]] / T[ix[("Q", gg)]]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.array([tpr(m, 1) - tpr(m, 0) for m in range(M)])
        b = np.array([fpr(m, 1) - fpr(m, 0) for m in range(M)])
    return a, b


def eo_gap(T, ix, M):
    a, b = model_gaps(T, ix, M)
    return np.maximum(np.abs(a), np.abs(b))


class GapEstimands:
    """Per-model group gaps dTPR_m, dFPR_m as allocation targets (same interface as Estimands for
    neyman_allocation: value, gradient)."""
    def __init__(self, names, M):
        self.ix = {n: k for k, n in enumerate(names)}; self.M = M

    def value(self, T):
        a, b = model_gaps(T, self.ix, self.M); return np.concatenate([a, b])

    def gradient(self, T):
        ix = self.ix; G = np.zeros((2 * self.M, len(T)))
        for r, (num, den) in enumerate((("TP", "P"), ("FP", "Q"))):
            for m in range(self.M):
                for gg, sgn in ((1, 1.0), (0, -1.0)):
                    D = T[ix[(den, gg)]]
                    G[r * self.M + m, ix[(num, m, gg)]] += sgn / D
                    G[r * self.M + m, ix[(den, gg)]] += -sgn * T[ix[(num, m, gg)]] / D ** 2
        return G


def kendall_tau(a, b):
    """Kendall tau-b between two score vectors (ties handled by scipy)."""
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).statistic)
