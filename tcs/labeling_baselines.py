"""Baseline labelling designs and estimators for comparing fixed classifiers on fairness contrasts.

All methods estimate the same totals T_c = sum_i t_ic(Y_i), t_ic(y) = y U_ic + (1 - y) V_ic
(tcs.labeling.total_columns), plug them into the same smooth map theta = f(T) (tcs.labeling.Estimands)
and form 95% Wald intervals from a linearisation variance. They differ in how the L labelled
instances are chosen and in how the totals are estimated.

  B1 'is'        Importance sampling in the spirit of Sawade, Landwehr and Scheffer (2012): draws with
                 replacement from an instrumental distribution q (mixture over model pairs of the
                 proxy-optimal distributions), self-normalised (Hajek) totals, linearisation variance.
  B2 'score'     Proxy-score stratification (Bennett and Carvalho 2010; Kumar and Raj 2018): strata =
                 group x within-group quantile bins of pi_i, same weighted Neyman rule as PATTERN.
  B3 'dis'       Disagreement-focused two-phase design: SRS of n0 from the pool, then all remaining
                 labels by SRS from the any-disagreement set D; combined stratified estimator over
                 {A, D} with the realised sample sizes.
  B4 'ppi'       Prediction-powered (difference) estimator with SRS labels (Angelopoulos et al. 2023),
     'ppipp'     power-tuned version (PPI++, Angelopoulos, Duchi and Zrnic), and
     'stratppi'  stratified PPI (Fisch et al. 2024): strata = group x pi-bins, per-stratum tuning,
                 allocation optimal for the PPI residual variance.
  B5 'pattern_ppi'  PATTERN strata and allocation with the stratified, power-tuned difference
                 (model-assisted, GREG-type) estimator.

The proxy is pi_i = mean_m phat_mi throughout, and the 'prediction' of t_ic is its proxy mean
f_ic = pi_i U_ic + (1 - pi_i) V_ic.
"""
import numpy as np
from tcs.labeling import make_strata, collapse_small, neyman_allocation, _round_alloc

Z95 = 1.959964


# ---------------------------------------------------------------- vectorised estimands
class VecEstimands:
    """Vectorised version of tcs.labeling.Estimands (same target order: per pair dTPR, dFPR, dERR)."""

    def __init__(self, est):
        ix = est.ix; self.N = est.N; nt = len(est.labels); self.nt = nt
        self.num_m = np.zeros((nt, 2), int); self.num_j = np.zeros((nt, 2), int); self.den = np.zeros((nt, 2), int)
        self.err = np.zeros(nt, bool); self.e_m = np.zeros(nt, int); self.e_j = np.zeros(nt, int)
        r = 0
        for (m, j) in est.pairs:
            for num, den in (("TP", "P"), ("FP", "Q")):
                for gg in (0, 1):
                    self.num_m[r, gg] = ix[(num, m, gg)]; self.num_j[r, gg] = ix[(num, j, gg)]; self.den[r, gg] = ix[(den, gg)]
                r += 1
            self.err[r] = True; self.e_m[r] = ix[("E", m)]; self.e_j[r] = ix[("E", j)]; r += 1
        self.ratio = ~self.err
        self.n_c = len(est.ix)

    def value_rows(self, Tm):
        """Tm: (n_t, n_c), row t is the total vector used for target t. Returns theta (n_t,)."""
        r = np.arange(self.nt); out = np.empty(self.nt)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (Tm[r[:, None], self.num_m] - Tm[r[:, None], self.num_j]) / Tm[r[:, None], self.den]
        out[:] = d[:, 1] - d[:, 0]
        out[self.err] = (Tm[r[self.err], self.e_m[self.err]] - Tm[r[self.err], self.e_j[self.err]]) / self.N
        return out

    def grad_rows(self, Tm):
        """Row t: gradient of target t evaluated at Tm[t]. Shape (n_t, n_c)."""
        G = np.zeros((self.nt, self.n_c)); r = np.arange(self.nt)
        with np.errstate(divide="ignore", invalid="ignore"):
            for gg, s in ((1, 1.0), (0, -1.0)):
                rr = r[self.ratio]
                D = Tm[rr, self.den[rr, gg]]
                num = Tm[rr, self.num_m[rr, gg]] - Tm[rr, self.num_j[rr, gg]]
                np.add.at(G, (rr, self.num_m[rr, gg]), s / D)
                np.add.at(G, (rr, self.num_j[rr, gg]), -s / D)
                np.add.at(G, (rr, self.den[rr, gg]), -s * num / D ** 2)
        re = r[self.err]
        G[re, self.e_m[re]] = 1 / self.N; G[re, self.e_j[re]] = -1 / self.N
        return G

    def value(self, T):
        return self.value_rows(np.broadcast_to(T, (self.nt, len(T))))

    def grad(self, T):
        return self.grad_rows(np.broadcast_to(T, (self.nt, len(T))))

    def den_ok(self, Tm):
        """All denominator totals strictly positive and finite (Tm: one total vector or one row per target)."""
        Tm = np.asarray(Tm)
        if not np.all(np.isfinite(Tm)): return False
        rr = np.flatnonzero(self.ratio)
        D = Tm[self.den[rr]] if Tm.ndim == 1 else Tm[rr[:, None], self.den[rr]]
        return bool(np.all(D > 0))


def _nan(nt):
    x = np.full(nt, np.nan); return x, x, x, x


def _proxy_influence(U, V, p, ve, centred=False):
    """Influence values under y=1 (A) and y=0 (B) at the proxy totals; target weights w_t = 1 / S_t^2."""
    T_pi = (p[:, None] * U + (1 - p[:, None]) * V).sum(0)
    G = ve.grad(T_pi)
    A, B = U @ G.T, V @ G.T
    if centred:                          # subtract theta/N per unit (zero for the ratio targets: f homogeneous of degree 0)
        c = (G @ T_pi) / ve.N; A = A - c; B = B - c
    Ez = p[:, None] * A + (1 - p[:, None]) * B
    Ez2 = p[:, None] * A ** 2 + (1 - p[:, None]) * B ** 2
    S2 = Ez2.mean(0) - Ez.mean(0) ** 2
    return A, B, Ez, Ez2, 1 / np.maximum(S2, 1e-18)


# ---------------------------------------------------------------- B1: importance sampling (Sawade et al. 2012)
def is_distribution(U, V, p, ve, eps=0.05):
    """Instrumental distribution q over the pool.

    For a single target t, the variance of the linearised self-normalised estimator under
    with-replacement draws from q is (1/n) sum_i E[a_it^2] / q_i (a_it: centred influence,
    sum_i a_it = 0), minimised by q_i proportional to sqrt(E[a_it^2]) (Sawade et al. 2012, Thm 1,
    with p(x) uniform on the pool and the loss difference replaced by the influence value).
    E[.] is taken under y_i ~ Bernoulli(pi_i). For a pair (m, j) the three targets are combined
    with the weights 1/S_t^2 used by PATTERN; the pairwise distributions are then averaged with
    equal weights (Sawade et al.'s mixture over pairs), and mixed with the uniform distribution
    with weight eps (defensive floor, keeps every q_i > 0)."""
    A, B, Ez, Ez2, w = _proxy_influence(U, V, p, ve, centred=True)
    N, nt = Ez2.shape; npair = nt // 3
    q = np.zeros(N)
    for k in range(npair):
        cols = slice(3 * k, 3 * k + 3)
        s = np.sqrt((Ez2[:, cols] * w[cols]).sum(1))
        if s.sum() > 0: q += s / s.sum()
    q = q / q.sum()
    return (1 - eps) * q + eps / N


def is_draw(q, L, rng):
    """Draw with replacement from q until L distinct instances have been drawn (L labels spent)."""
    N = len(q); draws = np.empty(0, int)
    while len(np.unique(draws)) < L:
        draws = np.concatenate([draws, rng.choice(N, size=max(L, 64), p=q)])
    _, first = np.unique(draws, return_index=True)
    cut = np.sort(first)[L - 1]
    return draws[:cut + 1]


def is_estimate(q, draws, y, U, V, ve):
    """Self-normalised (Hajek) totals T = N sum_k t_k/q_k / sum_k 1/q_k and linearisation SE."""
    N = ve.N; n = len(draws); yk = y[draws]
    t = yk[:, None] * U[draws] + (1 - yk[:, None]) * V[draws]
    w = 1 / q[draws]
    T = N * (w[:, None] * t).sum(0) / w.sum()
    if not ve.den_ok(T): return _nan(ve.nt)
    th = ve.value(T); G = ve.grad(T)
    z = ((t - T / N) * w[:, None]) @ G.T                        # T_hat - T ~ mean_k (t_k - T/N)/q_k (Hajek linearisation)
    se = np.sqrt(z.var(0, ddof=1) / n) if n > 1 else np.full(ve.nt, np.nan)
    return th, se, th - Z95 * se, th + Z95 * se


# ---------------------------------------------------------------- B2: proxy-score strata
def score_strata(p, g, bins=10, min_size=50):
    """Strata = group x within-group quantile bins of the proxy pi_i (ties broken by rank)."""
    st = np.zeros(len(p), int)
    for gg in (0, 1):
        idx = np.flatnonzero(g == gg)
        r = np.argsort(np.argsort(p[idx], kind="stable"), kind="stable")
        st[idx] = gg * bins + (r * bins // len(idx))
    return collapse_small(st, g.astype(int), min_size)


# ---------------------------------------------------------------- B3: disagreement-focused two-phase design
def dis_draw(pred, L, frac0, rng):
    """Phase 1: SRS of n0 = round(frac0 L) from the pool. Phase 2: SRS of L - n0 from D minus phase 1,
    D = {i : the M models do not all agree}. Returns labelled indices, strata (0 = A, 1 = D), sizes."""
    N, M = pred.shape; K = pred.sum(1); inD = (K > 0) & (K < M)
    n0 = int(round(frac0 * L))
    s0 = rng.choice(N, n0, replace=False)
    rest = np.setdiff1d(np.flatnonzero(inD), s0)
    s1 = rng.choice(rest, min(L - n0, len(rest)), replace=False)
    idx = np.concatenate([s0, s1])
    if len(idx) < L:                     # D exhausted: spend the remainder on A by SRS
        restA = np.setdiff1d(np.arange(N), idx)
        idx = np.concatenate([idx, rng.choice(restA, L - len(idx), replace=False)])
    st = inD.astype(int); ids = np.array([0, 1])
    Nh = np.array([(st == 0).sum(), (st == 1).sum()])
    nh = np.array([(st[idx] == 0).sum(), (st[idx] == 1).sum()])
    return idx, st, ids, Nh, nh


# ---------------------------------------------------------------- B4/B5: (stratified) prediction-powered estimators
def ppi_estimate(strata, ids, Nh, nh, idx, y, U, V, p, ve, tune=True, lam_clip=(0.0, 1.0)):
    """Stratified difference estimator, per target t and stratum h:
        T^(t) = sum_h [ lam_ht F_h + N_h mean_{S_h}(t_i - lam_ht f_i) ],   F_h = sum_{i in h} f_i,
    f_i = pi_i U_i + (1 - pi_i) V_i. lam = 1 is PPI (difference estimator); tune=True chooses
    lam_ht = Cov_h(a, b) / Var_h(b) on the labelled units (a = grad_t . t_i, b = grad_t . f_i;
    no (1 + n/N) factor because f is known on the whole pool), clipped to lam_clip.
    One stratum gives PPI / PPI++ under SRS. Variance: sum_h N_h^2 (1 - n_h/N_h) s^2_h(a - lam b) / n_h."""
    yk = y[idx]
    t = yk[:, None] * U[idx] + (1 - yk[:, None]) * V[idx]
    f_all = p[:, None] * U + (1 - p[:, None]) * V
    f = f_all[idx]; h_of = strata[idx]
    H = len(ids); nc = U.shape[1]
    F = np.array([f_all[strata == h].sum(0) for h in ids])                  # (H, n_c)
    tbar = np.zeros((H, nc)); fbar = np.zeros((H, nc)); sel = []
    for k, h in enumerate(ids):
        s = h_of == h; sel.append(s)
        if s.sum() == 0: return _nan(ve.nt)
        tbar[k] = t[s].mean(0); fbar[k] = f[s].mean(0)
    T_stsi = (Nh[:, None] * tbar).sum(0)
    D = F - Nh[:, None] * fbar                                              # (H, n_c) correction per stratum
    lam = np.ones((H, ve.nt))
    if tune:
        T0 = T_stsi + D.sum(0)                                              # lam = 1 start for the gradient
        if not ve.den_ok(T0): T0 = T_stsi
        if not ve.den_ok(T0): return _nan(ve.nt)
        G0 = ve.grad(T0)
        a, b = t @ G0.T, f @ G0.T
        for k in range(H):
            s = sel[k]
            if s.sum() < 3: continue
            bc = b[s] - b[s].mean(0); ac = a[s] - a[s].mean(0)
            vb = (bc ** 2).sum(0)
            lam[k] = np.where(vb > 1e-30, (ac * bc).sum(0) / np.maximum(vb, 1e-300), 1.0)
        lam = np.clip(lam, *lam_clip)
    Tm = T_stsi[None, :] + lam.T @ D                                        # (n_t, n_c) row t for target t
    if not ve.den_ok(Tm): return _nan(ve.nt)
    th = ve.value_rows(Tm); G = ve.grad_rows(Tm)
    a, b = t @ G.T, f @ G.T                                                 # (n, n_t)
    var = np.zeros(ve.nt)
    for k in range(H):
        s = sel[k]; n_h, N_h = nh[k], Nh[k]
        if n_h >= 2 and n_h < N_h:
            r = a[s] - lam[k] * b[s]
            var += N_h ** 2 * (1 - n_h / N_h) * r.var(0, ddof=1) / n_h
    se = np.sqrt(var)
    return th, se, th - Z95 * se, th + Z95 * se


def stratppi_allocation(strata, U, V, p, ve, L, min_n=10):
    """Neyman allocation for the PPI residual: under y ~ Bernoulli(pi) the residual a - b of target t
    has mean zero and variance pi (1 - pi) (A - B)^2, so n_h proportional to
    N_h sqrt( sum_t w_t mean_{i in h} pi_i (1 - pi_i) (A_it - B_it)^2 ), with the PATTERN weights w_t."""
    A, B, Ez, Ez2, w = _proxy_influence(U, V, p, ve)
    r2 = (p * (1 - p))[:, None] * (A - B) ** 2
    ids = np.unique(strata); Nh = np.array([(strata == h).sum() for h in ids])
    score = np.array([np.sqrt(np.sum(w * r2[strata == h].mean(0))) for h in ids])
    return ids, Nh, _round_alloc(Nh * score, Nh, L, min_n)
