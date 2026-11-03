"""Step-down trust confidence sets with an optional practical-equivalence margin.

Target sets (mu[m, k] = population value, smaller is better; eps_k >= 0 a margin)
  * eps-domination: j eps-dominates m if mu[m, k] - mu[j, k] > eps_k for every k.
  * P0(eps) = {m : no j eps-dominates m}.   eps = 0 gives the strict-dominance set P0 of core.py,
    and P0 is a subset of P0(eps).
Guarantee (asymptotic, same conditions as Proposition 1): Pr(P0(eps) subset of TCS) >= 1 - alpha.
Every excluded model is certified as being worse than some rival by more than eps_k on every
criterion.  A margin changes the target (a larger set), it does not add power for the target P0.

Statistics: t[m,j,k] = (d[m,j,k] - eps_k) / se,  S[m] = max_{j != m} min_k t[m,j,k].
Bootstrap: ts = (d* - d)/se (centred, margin-free).

Variants
  stepdown=True   Step-down.  Step s uses the candidate family A_s (initially all models), critical value
                  q_s = (1-alpha) quantile of max over {m in A_s, j != m, k} of ts, and removes
                  {m in A_s : S[m] > q_s}.  Repeat until nothing is removed.  Rivals j always range over ALL
                  models (a model that has been excluded is still a valid dominator in the population).
                  Validity: for m in P0(eps), S[m] <= Z(P0(eps)) := max over (m' in P0(eps), j, k with
                  delta <= eps) of z.  A false exclusion at step s happens first while P0(eps) is a
                  subset of A_s, hence q_s >= q(P0(eps)) and requires Z > q(P0(eps)): probability -> alpha.
  preselect=beta  (two-stage, "sharper" critical value).  Triples (m,j,k) with t0 > c_pre are dropped from the
                  critical-value family, where c_pre is the (1-beta) quantile of the max over all triples; the
                  main critical value uses level alpha - beta.  On the event {Z <= c_pre} (prob >= 1-beta
                  asymptotically) every triple with delta <= eps survives the preselection, and a
                  Bonferroni split gives total error <= alpha.  Asymptotic; it costs the beta in level
                  and gains only when many triples are clearly positive.  Argued, not proved in
                  detail (needs the same joint bootstrap consistency).
"""
from dataclasses import dataclass, field
import numpy as np
from .core import pareto_front


@dataclass
class StepdownResult:
    keep: np.ndarray
    stat: np.ndarray
    crit: float                       # final-step critical value
    crits: list = field(default_factory=list)   # critical value at each step
    n_steps: int = 1
    keep_step1: np.ndarray = None     # retained set after the first step (= single-step TCS)
    eps: np.ndarray = None
    pareto: np.ndarray = None


def _prep(point, boot, eps_k, tiny=1e-12):
    point = np.asarray(point, float); boot = np.asarray(boot, float)
    B, M, K = boot.shape
    d0 = point[:, None, :] - point[None, :, :]
    db = boot[:, :, None, :] - boot[:, None, :, :]
    se = db.std(axis=0, ddof=1)
    degenerate = se < tiny
    se = np.where(degenerate, np.inf, se)
    num = d0 - eps_k[None, None, :]
    t0 = num / se
    # zero-variance pair: t = 0 if the difference is not below the margin (as in core.py), else -inf
    t0 = np.where(degenerate, np.where(num >= 0, 0.0, -np.inf), t0)
    ts = (db - d0[None]) / se[None]
    ts = np.where(degenerate[None], 0.0, ts)
    return t0, ts


def trust_confidence_set_sd(point, boot, alpha=0.05, eps=0.0, stepdown=True, preselect=None):
    point = np.asarray(point, float)
    B, M, K = np.asarray(boot).shape
    eps_k = np.broadcast_to(np.asarray(eps, float), (K,)).copy()
    t0, ts = _prep(point, boot, eps_k)
    offdiag = ~np.eye(M, dtype=bool)                                   # (M, M)
    pair0 = t0.min(axis=2)
    pair0 = np.where(offdiag, pair0, -np.inf)
    S = pair0.max(axis=1)

    fam = np.broadcast_to(offdiag[:, :, None], (M, M, K)).copy()      # triples (m, j, k)
    level = 1 - alpha
    if preselect:
        beta = float(preselect)
        c_pre = np.quantile(np.where(fam[None], ts, -np.inf).reshape(B, -1).max(1), 1 - beta)
        fam = fam & (t0 <= c_pre)
        level = 1 - (alpha - beta)

    A = np.ones(M, bool)
    crits, keep1 = [], None
    while True:
        F = fam & A[:, None, None]
        vals = np.where(F[None], ts, -np.inf).reshape(B, -1).max(1)
        q = float(np.quantile(vals, level))
        crits.append(q)
        newA = A & (S <= q)
        if keep1 is None:
            keep1 = newA.copy()
        if not stepdown or newA.sum() == A.sum():
            A = newA
            break
        A = newA
    return StepdownResult(keep=A, stat=S, crit=crits[-1], crits=crits, n_steps=len(crits),
                          keep_step1=keep1, eps=eps_k, pareto=pareto_front(point))


def eps_nondominated(mu, eps=0.0):
    """Population set P0(eps) for a (M, K) matrix of population values."""
    mu = np.asarray(mu, float)
    M, K = mu.shape
    eps = np.broadcast_to(np.asarray(eps, float), (K,))
    out = np.ones(M, bool)
    for m in range(M):
        for j in range(M):
            if j != m and np.all(mu[m] - mu[j] > eps):
                out[m] = False
    return out
