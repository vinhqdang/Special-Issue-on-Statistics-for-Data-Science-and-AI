"""Preference-robust trust confidence set.

The user states a set W of acceptable weight vectors over the K criteria (all criteria smaller is
better, in common units). Model m is W-dominated by j if w'(mu_m - mu_j) > 0 for every w in W.
For a polytope W this holds iff it holds at every vertex of W, so the procedure is the trust
confidence set applied to the vertex scores c @ V.T (one "criterion" per vertex).

W = full simplex  -> vertices are the unit vectors -> ordinary (Pareto) TCS.
W = {w0}          -> one scalar score -> max-t confidence set for the best model under w0.
"""
import itertools
import numpy as np
from .core import trust_confidence_set


def box_simplex_vertices(w0, r):
    """Vertices of {w : sum w = 1, max(0, w0-r) <= w <= min(1, w0+r)}."""
    w0 = np.asarray(w0, float); K = len(w0)
    lo, hi = np.maximum(0, w0 - r), np.minimum(1, w0 + r)
    V = []
    for i in range(K):
        others = [k for k in range(K) if k != i]
        for bits in itertools.product((0, 1), repeat=K - 1):
            w = np.empty(K)
            for k, b in zip(others, bits): w[k] = hi[k] if b else lo[k]
            w[i] = 1 - w[others].sum()
            if lo[i] - 1e-12 <= w[i] <= hi[i] + 1e-12: V.append(w)
    V = np.unique(np.round(np.array(V), 12), axis=0)
    return V


def preference_confidence_set(point, boot, V, alpha=0.05):
    V = np.atleast_2d(V)
    return trust_confidence_set(point @ V.T, boot @ V.T, alpha)


def w_dominated(mu, V):
    """Population W-dominated mask from criteria means mu (M, K) and vertices V."""
    s = mu @ np.atleast_2d(V).T
    return np.array([np.any(np.all(s < s[m], axis=1)) for m in range(len(s))])


def worst_regret(mu, V):
    """Population worst-case regret over W of every model: max_{v in V, j} v'(mu_m - mu_j) >= 0."""
    s = mu @ np.atleast_2d(V).T                                   # (M, E)
    return (s[:, None, :] - s[None, :, :]).max(axis=(1, 2))


def certified_selection(point, boot, V, alpha=0.05, eps=1e-12):
    """Choose a model with a simultaneous upper confidence bound on its worst-case regret over W.

    U_m = max_{j, v} [ v'd_mj + q * se_mjv ], q = (1-alpha) quantile of the bootstrap maximum over
    (m, j != m, v) of the centred studentised v'(d* - d). Then, asymptotically,
    P(R_m(W) <= U_m for all m) >= 1 - alpha, so the chosen model's regret is certified by U_chosen.
    """
    V = np.atleast_2d(V)
    s0, sb = point @ V.T, boot @ V.T                              # (M, E), (B, M, E)
    d0 = s0[:, None, :] - s0[None, :, :]
    db = sb[:, :, None, :] - sb[:, None, :, :]
    se = db.std(0, ddof=1); se = np.where(se < eps, 0.0, se)
    z = np.where(se[None] > 0, (db - d0[None]) / np.where(se > 0, se, 1)[None], 0.0)
    M = len(s0); off = ~np.eye(M, dtype=bool)
    q = float(np.quantile(z[:, off, :].reshape(len(z), -1).max(1), 1 - alpha))
    U = np.where(off[:, :, None], d0 + q * se, 0.0).max(axis=(1, 2))
    U = np.maximum(U, 0.0)
    return dict(choice=int(U.argmin()), U=U, q=q, plugin=d0.max(axis=(1, 2)))
