"""Monte-Carlo check of TCS with non-smooth criteria: error, 10-bin ECE, equalised-odds gap.

Population: x ~ N(0, I_5), group g ~ Bernoulli(0.3), y ~ Bernoulli(sigmoid(beta'x + b0 + b1 g)).
Model m is a fixed scoring rule p_m = sigmoid(s_m * (beta_m'x + o_m g)), prediction 1{p_m>0.5}.
Population criteria are the plug-in functionals (binned ECE with fixed bins, EO gap with max/abs)
of the population law, approximated by a very large independent sample.
Estimates use bootstrap weights (multinomial counts), so the same resample serves all models.
"""
import sys, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs import trust_confidence_set, pareto_front
from scipy.stats import norm

D, M, NB = 5, 12, 10
BETA = np.array([1.0, -0.8, 0.6, 0.0, 0.0])


def sig(z): return 1 / (1 + np.exp(-z))


import os
DUP = os.environ.get("DUP") == "1"   # last model is an exact copy of model 0 (ties on every criterion)


def make_design(seed):
    r = np.random.default_rng(seed)
    des = dict(beta=BETA + 0.45 * r.standard_normal((M, D)), s=np.exp(r.uniform(-0.7, 0.7, M)),
               o=r.uniform(-0.7, 0.7, M))
    if DUP:
        for k in des: des[k][-1] = des[k][0]
    return des


def draw(n, rng):
    x = rng.standard_normal((n, D)); g = (rng.random(n) < 0.3).astype(int)
    y = (rng.random(n) < sig(x @ BETA - 0.2 + 0.3 * g)).astype(int)
    return x, g, y


def models(des, x, g):
    p = sig(des["s"] * (x @ des["beta"].T + des["o"] * g[:, None]))     # (n, M)
    return p, (p > 0.5).astype(float)


def crit_w(W, y, g, p, pred):
    """W: (B, n) nonnegative weights. returns (B, M, 3): error, ECE, EO gap."""
    sw = W.sum(1, keepdims=True)                                         # (B,1)
    err = (W @ (pred != y[:, None])) / sw
    bins = np.minimum((p * NB).astype(int), NB - 1)
    ece = np.zeros_like(err)
    for b in range(NB):
        mk = (bins == b).astype(float)
        cnt = W @ mk; conf = W @ (mk * p); acc = W @ (mk * y[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            term = np.where(cnt > 0, np.abs(conf - acc) / np.maximum(cnt, 1e-12), 0.0) * cnt
        ece += term / sw
    eo = np.zeros_like(err)
    for lab in (1, 0):
        m0 = ((y == lab) & (g == 0)).astype(float)[:, None]; m1 = ((y == lab) & (g == 1)).astype(float)[:, None]
        r0 = (W @ (pred * m0)) / np.maximum(W @ m0, 1e-12); r1 = (W @ (pred * m1)) / np.maximum(W @ m1, 1e-12)
        eo = np.maximum(eo, np.abs(r0 - r1))
    return np.stack([err, ece, eo], axis=2)


def population(des, N=2_000_000, seed=123):
    rng = np.random.default_rng(seed); x, g, y = draw(N, rng); p, pred = models(des, x, g)
    return crit_w(np.ones((1, N)), y, g, p, pred)[0]


def dominated_mask(mu):
    return np.array([np.any(np.all(mu < mu[m], axis=1)) for m in range(len(mu))])


def box_keep(point, boot, alpha):
    B, Mm, K = boot.shape; se = boot.std(0, ddof=1); z = norm.ppf(1 - alpha / (2 * Mm * K))
    lo, hi = point - z * se, point + z * se
    keep = np.ones(Mm, bool)
    for m in range(Mm):
        for j in range(Mm):
            if j != m and np.all(hi[j] < lo[m]): keep[m] = False; break
    return keep


def one(args):
    dseed, n, seed, B, alpha = args
    des = make_design(dseed); rng = np.random.default_rng(seed)
    x, g, y = draw(n, rng); p, pred = models(des, x, g)
    point = crit_w(np.ones((1, n)), y, g, p, pred)[0]
    W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    boot = crit_w(W, y, g, p, pred)
    r = trust_confidence_set(point, boot, alpha)
    kb = box_keep(point, boot, alpha)
    return dict(tcs=r.keep, box=kb, naive=r.pareto, point=point)


if __name__ == "__main__":
    R, B, alpha = int(sys.argv[1]), 300, 0.05
    designs = [int(a) for a in sys.argv[2].split(",")]
    out = []
    with Pool(4) as pool:
        for ds in designs:
            mu = population(make_design(ds)); dom = dominated_mask(mu); P0 = ~dom
            # smallest margin by which dominated models are dominated, relative to the criteria scale
            marg = [float(np.min([np.max(mu[m] - mu[j]) * -1 + 0 for j in range(M) if np.all(mu[j] < mu[m])] or [np.nan])) if dom[m] else None for m in range(M)]
            print(f"design {ds}: |P0|={P0.sum()} of {M}", flush=True)
            for n in (500, 2000, 10000):
                res = pool.map(one, [(ds, n, 100000 * ds + 17 * n + r, B, alpha) for r in range(R)])
                row = dict(design=ds, n=n, P0=int(P0.sum()), R=R)
                for k in ("tcs", "box", "naive"):
                    keep = np.array([x[k] for x in res])
                    row[k + "_retain"] = float(keep[:, P0].all(1).mean())
                    row[k + "_excl"] = float((~keep[:, dom]).mean()) if dom.any() else None
                    row[k + "_size"] = float(keep.sum(1).mean())
                out.append(row); print(row, flush=True)
    json.dump(out, open("results/simulation_nonsmooth_dup.json" if DUP else "results/simulation_nonsmooth.json", "w"), indent=1)
