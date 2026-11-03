"""Trust confidence sets (TCS).

Given M candidate models and K "lower is better" trust criteria evaluated on a
common test sample, return the set of models that are *not* significantly
dominated on every criterion, with asymptotic family-wise control over the
retention of all population-nondominated models.

Population definition. Let mu[m, k] be the population value of criterion k for
model m (smaller is better).  Model m is *strictly dominated* by model j if
mu[m, k] > mu[j, k] for every k.  P0 = {m : no j strictly dominates m}.

Procedure (one-shot, studentised, intersection-union within pair, max over pairs):
  d[m, j, k] = c[m, k] - c[j, k]            (positive = m worse than j)
  t[m, j, k] = d / se,  se from the bootstrap
  S[m]       = max_j min_k t[m, j, k]
  c_alpha    = (1-alpha) quantile of  max_{m != j, k} t*[m, j, k],
               t* = (d* - d) / se   (bootstrap, centred)
The critical value uses the maximum over criteria (not the minimum): for a model in P0 and
any rival j, at least one criterion k(j) has mu[m,k]-mu[j,k] <= 0, so S[m] is bounded by a
max over pairs of single-criterion t statistics, whose centred bootstrap law is c_alpha's
reference.  This is the least-favourable configuration of the intersection-union test.
  TCS        = {m : S[m] <= c_alpha}
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class TCSResult:
    keep: np.ndarray          # bool (M,)
    stat: np.ndarray          # S[m]
    crit: float               # c_alpha
    point: np.ndarray         # (M, K) point estimates
    se: np.ndarray            # (M, M, K) bootstrap s.e. of pairwise differences
    pareto: np.ndarray        # bool (M,) naive Pareto front of point estimates
    dominated_by: np.ndarray  # (M,) index of the pair attaining S[m] (-1 if none)


def pareto_front(c):
    """Boolean mask of non-dominated rows (weak Pareto: dominated if some j is <= in all
    criteria and < in at least one)."""
    M = c.shape[0]
    keep = np.ones(M, bool)
    for m in range(M):
        le = np.all(c <= c[m], axis=1)
        lt = np.any(c < c[m], axis=1)
        if np.any(le & lt):
            keep[m] = False
    return keep


def trust_confidence_set(point, boot, alpha=0.05, eps=1e-12):
    """point: (M, K) estimates; boot: (B, M, K) bootstrap replicates (same statistic)."""
    point = np.asarray(point, float)
    boot = np.asarray(boot, float)
    B, M, K = boot.shape
    d0 = point[:, None, :] - point[None, :, :]                     # (M, M, K)
    db = boot[:, :, None, :] - boot[:, None, :, :]                 # (B, M, M, K)
    se = db.std(axis=0, ddof=1)
    se = np.where(se < eps, np.inf, se)                            # degenerate => t = 0
    t0 = d0 / se
    ts = (db - d0[None]) / se[None]
    pair0 = t0.min(axis=2)                                         # (M, M)
    np.fill_diagonal(pair0, -np.inf)
    pairb = ts.max(axis=3)                                         # (B, M, M)
    idx = np.arange(M)
    pairb[:, idx, idx] = -np.inf
    S = pair0.max(axis=1)
    c_alpha = float(np.quantile(pairb.reshape(B, -1).max(axis=1), 1 - alpha))
    return TCSResult(
        keep=S <= c_alpha, stat=S, crit=c_alpha, point=point, se=se,
        pareto=pareto_front(point), dominated_by=pair0.argmax(axis=1),
    )
