"""Label-efficient comparison of classifiers on error and group-fairness metrics.

Setting. A pool of N unlabeled instances; for every instance the predictions of M fixed models and
the group g in {0, 1} are known; the label y is unknown and costs one unit to obtain. Budget: L labels.

Estimands are smooth functions of population totals T_c = sum_i t_ic(y_i), each linear in y_i:
  P_g  = sum_{i in g} y_i,            Q_g  = sum_{i in g} (1 - y_i),
  TP_mg = sum_{i in g} y_i yhat_mi,   FP_mg = sum_{i in g} (1 - y_i) yhat_mi,   E_m = sum_i |yhat_mi - y_i|.
For a pair (m, j) the comparison estimands are
  dTPR(m,j) = (TP_m1/P_1 - TP_m0/P_0) - (TP_j1/P_1 - TP_j0/P_0)   (difference of TPR gaps),
  dFPR(m,j) = same with FP and Q                                    (difference of FPR gaps),
  dERR(m,j) = (E_m - E_j) / N                                        (difference of error rates).
Key structural fact: the numerators of these differences only involve instances on which m and j
disagree; where all models agree only the denominators P_g, Q_g carry information.

Design. Strata h = (group, number of models predicting 1). Stratified sampling without replacement,
n_h labels in stratum h, Horvitz-Thompson totals T^_c = sum_h N_h * mean_h(t_c), plug-in estimands,
linearisation variance  Var = sum_h N_h^2 (1 - n_h/N_h) s_h^2(z) / n_h  with z the influence values.
Allocation (Neyman-type, no labels needed): n_h proportional to N_h * sqrt( sum_t S_ht^2 / S_t^2 ),
where S_ht^2 is the within-stratum variance of target t's influence values with y replaced by a
Bernoulli(p_i) proxy (p_i: average predicted probability of the models), and S_t^2 its overall
variance (so each target is weighted by its variance under simple random sampling).
Unbiasedness of the totals does not depend on the proxy; a poor proxy only costs efficiency.
"""
import numpy as np
from itertools import combinations


# ---------------------------------------------------------------- totals and estimands
def total_columns(pred, g):
    """Return (U, V): t_ic(y) = y*U_ic + (1-y)*V_ic, plus column names."""
    N, M = pred.shape
    cols_u, cols_v, names = [], [], []
    for gg in (0, 1):
        ind = (g == gg).astype(float)
        cols_u.append(ind); cols_v.append(np.zeros(N)); names.append(("P", gg))
        cols_u.append(np.zeros(N)); cols_v.append(ind); names.append(("Q", gg))
        for m in range(M):
            cols_u.append(ind * pred[:, m]); cols_v.append(np.zeros(N)); names.append(("TP", m, gg))
            cols_u.append(np.zeros(N)); cols_v.append(ind * pred[:, m]); names.append(("FP", m, gg))
    for m in range(M):
        cols_u.append(1 - pred[:, m]); cols_v.append(pred[:, m]); names.append(("E", m))
    return np.column_stack(cols_u), np.column_stack(cols_v), names


class Estimands:
    def __init__(self, names, M, N, pairs=None):
        self.ix = {n: k for k, n in enumerate(names)}; self.M = M; self.N = N
        self.pairs = pairs if pairs is not None else list(combinations(range(M), 2))
        self.labels = [(kind, m, j) for (m, j) in self.pairs for kind in ("dTPR", "dFPR", "dERR")]

    def value(self, T):
        ix = self.ix; out = []
        for (m, j) in self.pairs:
            tpr = lambda a, gg: T[ix[("TP", a, gg)]] / T[ix[("P", gg)]]
            fpr = lambda a, gg: T[ix[("FP", a, gg)]] / T[ix[("Q", gg)]]
            out.append((tpr(m, 1) - tpr(m, 0)) - (tpr(j, 1) - tpr(j, 0)))
            out.append((fpr(m, 1) - fpr(m, 0)) - (fpr(j, 1) - fpr(j, 0)))
            out.append((T[ix[("E", m)]] - T[ix[("E", j)]]) / self.N)
        return np.array(out)

    def gradient(self, T):
        """d estimand / d total, shape (n_targets, n_totals), analytic."""
        ix = self.ix; G = np.zeros((3 * len(self.pairs), len(T)))
        r = 0
        for (m, j) in self.pairs:
            for kind, num, den in (("dTPR", "TP", "P"), ("dFPR", "FP", "Q")):
                for gg, sgn in ((1, 1.0), (0, -1.0)):
                    D = T[ix[(den, gg)]]
                    G[r, ix[(num, m, gg)]] += sgn / D
                    G[r, ix[(num, j, gg)]] -= sgn / D
                    G[r, ix[(den, gg)]] += -sgn * (T[ix[(num, m, gg)]] - T[ix[(num, j, gg)]]) / D ** 2
                r += 1
            G[r, ix[("E", m)]] = 1 / self.N; G[r, ix[("E", j)]] = -1 / self.N; r += 1
        return G


# ---------------------------------------------------------------- strata and allocation
def make_strata(pred, g, kind="pattern"):
    """Integer stratum id per instance. kind: 'pattern' (group x #models predicting 1) or 'group'."""
    if kind == "group":
        return g.astype(int)
    k = pred.sum(1).astype(int)
    return g.astype(int) * (pred.shape[1] + 1) + k


def neyman_allocation(strata, U, V, p, est, L, min_n=2):
    """Proxy-based allocation. U, V: total columns; p: proxy P(y=1); est: Estimands."""
    T_proxy = (p[:, None] * U + (1 - p[:, None]) * V).sum(0)
    G = est.gradient(T_proxy)                                  # (n_t, n_c)
    A, Bv = U @ G.T, V @ G.T                                   # influence if y=1 / y=0, (N, n_t)
    Ez = p[:, None] * A + (1 - p[:, None]) * Bv
    Ez2 = p[:, None] * A ** 2 + (1 - p[:, None]) * Bv ** 2
    S2_all = Ez2.mean(0) - Ez.mean(0) ** 2                     # overall variance per target
    w = 1 / np.maximum(S2_all, 1e-18)
    ids = np.unique(strata); Nh = np.array([(strata == h).sum() for h in ids])
    score = np.array([np.sqrt(np.sum(w * np.maximum(Ez2[strata == h].mean(0) - Ez[strata == h].mean(0) ** 2, 0))) for h in ids])
    return ids, Nh, _round_alloc(Nh * score, Nh, L, min_n)


def proportional_allocation(strata, L, min_n=2):
    ids = np.unique(strata); Nh = np.array([(strata == h).sum() for h in ids])
    return ids, Nh, _round_alloc(Nh.astype(float), Nh, L, min_n)


def _round_alloc(weight, Nh, L, min_n):
    n = np.minimum(Nh, min_n).astype(float)                   # floor so variances exist
    weight = np.where(weight > 0, weight, 1e-12)
    for _ in range(50):                                        # iterative capping at N_h
        free = n < Nh
        rem = L - n.sum()
        if rem <= 0 or not free.any(): break
        add = rem * weight * free / (weight * free).sum()
        n = np.minimum(Nh, n + add)
        if (L - n.sum()) < 1e-9: break
    n = np.floor(n).astype(int); n = np.maximum(n, np.minimum(Nh, min_n))
    short = L - n.sum()
    order = np.argsort(-(weight / np.maximum(n, 1)))
    for h in order:                                            # hand out the rounding remainder
        if short <= 0: break
        if n[h] < Nh[h]: n[h] += 1; short -= 1
    return n


# ---------------------------------------------------------------- estimation
def estimate(strata, ids, Nh, nh, U, V, y_lab, idx_lab, est, alpha_z=1.959964):
    """Estimates, standard errors and 95% CIs from the labelled sample idx_lab (labels y_lab)."""
    t = y_lab[:, None] * U[idx_lab] + (1 - y_lab[:, None]) * V[idx_lab]     # (n, n_c)
    h_of = strata[idx_lab]
    T = np.zeros(U.shape[1]); means = {}
    for h, N_h in zip(ids, Nh):
        sel = h_of == h
        if sel.sum() == 0: continue
        mu = t[sel].mean(0); means[h] = mu; T += N_h * mu
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = est.value(T); G = est.gradient(T)
    if not np.all(np.isfinite(theta)):                                         # a denominator total is zero
        nan = np.full(len(theta), np.nan); return nan, nan, nan, nan
    z = t @ G.T                                                               # influence values (n, n_t)
    var = np.zeros(len(theta))
    for h, N_h, n_h in zip(ids, Nh, nh):
        sel = h_of == h
        if n_h >= 2 and n_h < N_h:
            var += N_h ** 2 * (1 - n_h / N_h) * z[sel].var(0, ddof=1) / n_h
    se = np.sqrt(var)
    return theta, se, theta - alpha_z * se, theta + alpha_z * se


def draw(strata, ids, nh, rng):
    out = []
    for h, n_h in zip(ids, nh):
        pool = np.flatnonzero(strata == h)
        out.append(rng.choice(pool, int(n_h), replace=False))
    return np.concatenate(out)
