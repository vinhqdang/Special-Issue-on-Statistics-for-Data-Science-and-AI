"""Certified model selection: choose one model and bound its worst-case regret.

Setting. M trained models, K criteria (smaller is better, common units), a polytope W of acceptable
weight vectors with vertex matrix V (E x K). Vertex scores s_m(v) = v'mu_m. The worst-case regret of
model m over W is
    R_m = max_{v in V} [ s_m(v) - min_j s_j(v) ] = max_{v, j} d_mj(v),   d_mj(v) = s_m(v) - s_j(v),
because a linear function of w attains its maximum over a polytope at a vertex.

Estimates use one test sample and a paired bootstrap (the same resample for every model).
z_mjv = (d*_mjv - d^_mjv) / se_mjv is the centred studentised bootstrap difference. The family of all
ordered pairs is symmetric (d_mj = -d_jm), so its maximum controls deviations in both directions.

Choice: m^ = argmin_m R^_m (plug-in minimax regret). Three upper bounds for R_m^:

one-stage   U_m = max(0, max_{j != m, v} d^_mjv + q se_mjv),  q = (1-a) quantile of max_{m,j,v} z.
            Simultaneous over m, hence valid for the data-selected m^ (asymptotically).
two-stage   Stage 1 (level a/2): S_v = {j : max_i d^_jiv/se_jiv <= q1} contains the true best model at
            every vertex. Stage 2 (level a/2): U_m = max(0, max_v max_{j in S_v, j != m} d^_mjv + q2 se).
            Valid on the intersection of the two events, so with probability >= 1-a asymptotically.
selective   q_sel = (1-a) quantile over b of max_{j,v} (-z_{m*_b j v}), where m*_b is the plug-in
            choice in bootstrap sample b. Uses only the comparisons of the chosen model; it
            mimics the selection step but has no proof of validity here (heuristic).
"""
import numpy as np


def _diffs(point, boot, V, eps=1e-12):
    V = np.atleast_2d(V)
    s0, sb = point @ V.T, boot @ V.T                         # (M, E), (B, M, E)
    d0 = s0[:, None, :] - s0[None, :, :]                     # (M, M, E)
    db = sb[:, :, None, :] - sb[:, None, :, :]               # (B, M, M, E)
    se = db.std(0, ddof=1)
    ok = se > eps
    z = np.where(ok[None], (db - d0[None]) / np.where(ok, se, 1.0)[None], 0.0)
    return d0, db, np.where(ok, se, 0.0), z


def certify(point, boot, V, alpha=0.05):
    point = np.asarray(point, float); boot = np.asarray(boot, float)
    d0, db, se, z = _diffs(point, boot, V)
    B, M = db.shape[0], d0.shape[0]
    off = ~np.eye(M, dtype=bool)
    zmax = z[:, off, :].reshape(B, -1).max(1)               # max over all ordered pairs and vertices
    R_hat = d0.max(axis=(1, 2))                              # plug-in worst-case regret (>= 0, j = m gives 0)
    choice = int(R_hat.argmin())

    # one-stage
    q = float(np.quantile(zmax, 1 - alpha))
    U1 = np.maximum(0.0, np.where(off[:, :, None], d0 + q * se, -np.inf).max(axis=(1, 2)))

    # two-stage
    q1 = float(np.quantile(zmax, 1 - alpha / 2)); q2 = q1
    t = np.where(se > 0, d0 / np.where(se > 0, se, 1.0), 0.0)
    t_off = np.where(off[:, :, None], t, -np.inf)
    S = t_off.max(axis=1) <= q1                              # (M, E): j plausibly best at vertex v
    cand = off[:, :, None] & S[None, :, :]                   # rival j allowed for (m, ., v)
    U2 = np.maximum(0.0, np.where(cand, d0 + q2 * se, -np.inf).max(axis=(1, 2)))

    # selective (heuristic)
    Rb = db.max(axis=(2, 3))                                 # (B, M) bootstrap plug-in regrets
    mb = Rb.argmin(1)
    negz = -z[np.arange(B), mb]                              # (B, M, E) comparisons of the bootstrap choice
    negz[np.arange(B), mb, :] = -np.inf                      # drop j = m*
    qs = float(np.quantile(negz.reshape(B, -1).max(1), 1 - alpha))
    U3 = max(0.0, float(np.where(off[choice][:, None], d0[choice] + qs * se[choice], -np.inf).max()))

    return dict(choice=choice, R_hat=R_hat, U_one=U1, U_two=U2, U_sel=U3, q=q, q1=q1, q_sel=qs,
                n_plausible=S.sum(0))
