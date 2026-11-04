"""Checks for tcs.labeling_dif. Run:  python tests/test_labeling_dif.py   (or pytest tests/test_labeling_dif.py)

1. exact: on a tiny pool, enumerate every stratified sample; the difference estimator of every total is
   exactly unbiased (even for a deliberately bad proxy), its variance equals the stratified formula and
   the variance estimator is exactly unbiased for it.
2. Monte Carlo: on a synthetic pool, the linearised variance of the plug-in difference estimator of the
   comparison estimands matches the Monte Carlo variance; allocations are valid; PPI (one stratum) agrees
   with tcs.labeling_baselines.ppi_estimate(tune=False).
"""
import sys, itertools
import numpy as np
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, make_strata, proportional_allocation, draw
from tcs.labeling_baselines import VecEstimands, ppi_estimate
from tcs.labeling_dif import DifContext, dif_totals, dif_allocation, estimate_dif, estimate_dif_all


def tiny_pool(rng, N=13, M=3):
    pred = rng.integers(0, 2, (N, M)); g = rng.integers(0, 2, N); y = rng.integers(0, 2, N).astype(float)
    U, V, names = total_columns(pred, g)
    return pred, g, y, U, V, names


def test_exact():
    rng = np.random.default_rng(1)
    pred, g, y, U, V, names = tiny_pool(rng)
    strata = np.array([0] * 5 + [1] * 4 + [2] * 4); ids = np.array([0, 1, 2]); Nh = np.array([5, 4, 4]); nh = np.array([2, 3, 2])
    p = np.clip(rng.random(len(y)), 0.05, 0.95)                         # unrelated to y: a bad proxy
    est = Estimands(names, pred.shape[1], len(y)); ctx = DifContext(strata, ids, U, V, p, est)
    t_all = y[:, None] * U + (1 - y[:, None]) * V; Ttrue = t_all.sum(0); f = ctx.f
    groups = [list(itertools.combinations(np.flatnonzero(strata == h), n)) for h, n in zip(ids, nh)]
    Ts, Vs = [], []
    for combo in itertools.product(*groups):
        idx = np.array([i for c in combo for i in c])
        T, t, r, pos, cnt = dif_totals(strata, ids, Nh, idx, y[idx], U, V, p, ctx=ctx)
        v = np.zeros(U.shape[1])
        for k in range(3):
            v += Nh[k] ** 2 * (1 - nh[k] / Nh[k]) * r[pos == k].var(0, ddof=1) / nh[k]
        Ts.append(T); Vs.append(v)
    Ts, Vs = np.array(Ts), np.array(Vs)
    assert np.allclose(Ts.mean(0), Ttrue, atol=1e-10), "T_dif not unbiased"
    truevar = sum(Nh[k] ** 2 * (1 - nh[k] / Nh[k]) * (t_all - f)[strata == ids[k]].var(0, ddof=1) / nh[k] for k in range(3))
    assert np.allclose(Ts.var(0), truevar, atol=1e-10), "variance of T_dif differs from the stratified formula"
    assert np.allclose(Vs.mean(0), truevar, atol=1e-10), "variance estimator not unbiased"
    print(f"exact enumeration over {len(Ts)} samples: unbiased, variance formula and variance estimator exact")


def synthetic(rng, N=3000, M=3):
    x = rng.standard_normal((N, 2)); g = (x[:, 0] + 0.5 * rng.standard_normal(N) > 0.3).astype(int)
    py = 1 / (1 + np.exp(-(1.2 * x[:, 1] - 0.4)))
    y = (rng.random(N) < py).astype(float)
    pm = np.clip(py[:, None] + 0.25 * rng.standard_normal((N, M)), 0.02, 0.98)
    pred = (pm > 0.5).astype(int); p = pm.mean(1)
    U, V, names = total_columns(pred, g)
    return pred, g, y, p, U, V, Estimands(names, M, N)


def test_mc():
    rng = np.random.default_rng(7)
    pred, g, y, p, U, V, est = synthetic(rng)
    ve = VecEstimands(est); truth = est.value((y[:, None] * U + (1 - y[:, None]) * V).sum(0))
    strata = make_strata(pred, g, "pattern"); L = 400
    for kind in ("plain", "resid"):
        ids, Nh, nh = dif_allocation(strata, U, V, p, est, L, kind)
        assert nh.sum() == L and (nh <= Nh).all() and (nh >= np.minimum(Nh, 10)).all(), (kind, nh)
        ctx = DifContext(strata, ids, U, V, p, est); R = 3000; th = np.empty((R, len(truth))); se = np.empty_like(th)
        for r in range(R):
            idx = draw(strata, ids, nh, rng)
            th[r], ses = estimate_dif_all(strata, ids, Nh, nh, U, V, y[idx], idx, est, p, ctx); se[r] = ses["none"]
        mc = th.var(0, ddof=1); ratio = se.__pow__(2).mean(0) / mc
        cover = ((th - 1.959964 * se <= truth) & (truth <= th + 1.959964 * se)).mean(0)
        bias = (th.mean(0) - truth) / np.sqrt(mc / R)
        print(f"[{kind}] n_h={nh.tolist()}  E[V_hat]/MCvar: median {np.median(ratio):.3f} range [{ratio.min():.3f},{ratio.max():.3f}];"
              f" coverage mean {cover.mean():.3f} min {cover.min():.3f}; |bias| in MC-se units max {np.abs(bias).max():.2f}")
        assert 0.9 < np.median(ratio) < 1.1 and ratio.min() > 0.8 and ratio.max() < 1.25
        assert np.abs(bias).max() < 4 and cover.mean() > 0.93
    # PPI check: one stratum, same sample
    one = np.zeros(len(y), int); i1, N1, n1 = proportional_allocation(one, L)
    idx = draw(one, i1, n1, rng)
    a = estimate_dif(one, i1, N1, n1, U, V, y[idx], idx, est, p)
    b = ppi_estimate(one, i1, N1, n1, idx, y, U, V, p, ve, tune=False)
    assert all(np.allclose(u, w, equal_nan=True) for u, w in zip(a, b)), "differs from ppi_estimate(tune=False)"
    idx = draw(strata, ids, nh, rng)
    a = estimate_dif(strata, ids, Nh, nh, U, V, y[idx], idx, est, p)
    b = ppi_estimate(strata, ids, Nh, nh, idx, y, U, V, p, ve, tune=False)
    assert all(np.allclose(u, w, equal_nan=True) for u, w in zip(a, b)), "stratified: differs from ppi_estimate(tune=False)"
    print("matches tcs.labeling_baselines.ppi_estimate(tune=False) (one stratum and PATTERN strata)")


if __name__ == "__main__":
    test_exact(); test_mc(); print("all tests passed")
