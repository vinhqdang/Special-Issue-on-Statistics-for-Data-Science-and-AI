"""Monte-Carlo check of TCS: retention of population-Pareto models and exclusion power."""
import numpy as np, json, sys
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs import trust_confidence_set, pareto_front

K = 3
# population means (lower is better); rows = models
BASE = np.array([
    [0.10, 0.10, 0.30],   # 0 Pareto
    [0.30, 0.10, 0.10],   # 1 Pareto
    [0.10, 0.30, 0.10],   # 2 Pareto
    [0.20, 0.20, 0.20],   # 3 Pareto (balanced)
    [0.10, 0.10, 0.10],   # placeholder, replaced below
    [0.10, 0.10, 0.10],
])


def pop(delta):
    mu = BASE[:4].copy()
    one_ = np.ones(3)
    return np.vstack([
        mu,
        mu[3] + delta * one_,                      # 4 strictly dominated by model 3
        mu[3] + 2 * delta * one_,                  # 5 strictly dominated by model 3
        mu[3] + delta * np.array([1, 1, 0.0]),     # 6 tied with model 3 on criterion 3 (in P0)
        mu[3] + delta * np.array([1, 1, -1.0]),    # 7 trade-off (in P0)
    ])


def one(args):
    n, delta, rho, seed, B, alpha = args
    rng = np.random.default_rng(seed)
    mu = pop(delta)
    M = mu.shape[0]
    # per-instance criteria: mean + noise, noise correlated across criteria (rho) and models (0.5)
    s = 0.15
    Sig_k = rho * np.ones((K, K)) + (1 - rho) * np.eye(K)
    Sig_m = 0.5 * np.ones((M, M)) + 0.5 * np.eye(M)
    L = np.linalg.cholesky(np.kron(Sig_m, Sig_k))
    z = rng.standard_normal((n, M * K)) @ L.T
    x = mu.reshape(1, M, K) + s * z.reshape(n, M, K)
    est = x.mean(0)
    idx = rng.integers(0, n, size=(B, n))
    boot = np.stack([x[i].mean(0) for i in idx])
    r = trust_confidence_set(est, boot, alpha)
    P0 = np.array([True] * 4 + [False] * 2 + [True] * 2)
    return dict(
        tcs_cover=bool(r.keep[P0].all()),
        naive_cover=bool(r.pareto[P0].all()),
        tcs_size=int(r.keep.sum()),
        naive_size=int(r.pareto.sum()),
        tcs_excl=float((~r.keep[~P0]).mean()),
        naive_excl=float((~r.pareto[~P0]).mean()),
    )


if __name__ == "__main__":
    R, B, alpha = 400, 300, 0.05
    out = []
    jobs = []
    grid = [(n, d, rho) for n in (200, 500, 2000) for d in (0.02, 0.05) for rho in (0.0, 0.5)]
    with Pool() as p:
        for gi, (n, d, rho) in enumerate(grid):
            res = p.map(one, [(n, d, rho, 1000 * gi + r, B, alpha) for r in range(R)])
            row = dict(n=n, delta=d, rho=rho)
            for k in res[0]:
                row[k] = float(np.mean([x[k] for x in res]))
            out.append(row)
            print(row, flush=True)
    json.dump(out, open("results/simulation.json", "w"), indent=1)
