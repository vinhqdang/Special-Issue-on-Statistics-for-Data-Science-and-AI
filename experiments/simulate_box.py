"""Head-to-head: paired-bootstrap TCS versus a marginal confidence-box procedure.

Box procedure (set-wise, Bonferroni over models and criteria): keep model m unless some rival j
has upper_j,k < lower_m,k for every criterion k, with two-sided normal intervals at level
alpha/(M*K) per (model, criterion).
"""
import numpy as np, json, sys
from multiprocessing import Pool
from scipy.stats import norm
sys.path.insert(0, ".")
from tcs import trust_confidence_set
from experiments.simulate import pop, K


def box(x, alpha):
    n, M, _ = x.shape
    est = x.mean(0)
    se = x.std(0, ddof=1) / np.sqrt(n)
    z = norm.ppf(1 - alpha / (2 * M * K))
    lo, hi = est - z * se, est + z * se
    keep = np.ones(M, bool)
    for m in range(M):
        for j in range(M):
            if j != m and np.all(hi[j] < lo[m]):
                keep[m] = False
                break
    return keep


def one(args):
    n, delta, rho, seed, B, alpha = args
    rng = np.random.default_rng(seed)
    mu = pop(delta); M = mu.shape[0]
    Sig_k = rho * np.ones((K, K)) + (1 - rho) * np.eye(K)
    Sig_m = 0.5 * np.ones((M, M)) + 0.5 * np.eye(M)
    L = np.linalg.cholesky(np.kron(Sig_m, Sig_k))
    x = mu.reshape(1, M, K) + 0.15 * (rng.standard_normal((n, M * K)) @ L.T).reshape(n, M, K)
    idx = rng.integers(0, n, size=(B, n))
    r = trust_confidence_set(x.mean(0), np.stack([x[i].mean(0) for i in idx]), alpha)
    kb = box(x, alpha)
    P0 = np.array([True] * 4 + [False] * 2 + [True] * 2)
    return dict(tcs_cover=r.keep[P0].all(), box_cover=kb[P0].all(),
                tcs_excl=(~r.keep[~P0]).mean(), box_excl=(~kb[~P0]).mean(),
                tcs_size=r.keep.sum(), box_size=kb.sum())


if __name__ == "__main__":
    R, B, alpha = 300, 300, 0.05
    grid = [(n, d, rho) for n in (200, 500, 2000) for d in (0.02, 0.05) for rho in (0.0, 0.5)]
    out = []
    with Pool(4) as p:
        for gi, (n, d, rho) in enumerate(grid):
            res = p.map(one, [(n, d, rho, 7000 + 1000 * gi + r, B, alpha) for r in range(R)])
            row = dict(n=n, delta=d, rho=rho, **{k: float(np.mean([x[k] for x in res])) for k in res[0]})
            out.append(row); print(row, flush=True)
    json.dump(out, open("results/simulation_box.json", "w"), indent=1)
