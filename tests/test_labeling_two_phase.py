"""Check tcs.labeling_two_phase: with L = N/4 labels in a large pool, V_pool_hat approximates N * Var(z_i) computed on the
full pool, and the total variance matches the Monte Carlo variance of theta_hat around the population value.
Run: python tests/test_labeling_two_phase.py"""
import sys
import numpy as np
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, make_strata, neyman_allocation, draw
from tcs.labeling_dif import DifContext
from tcs.labeling_two_phase import estimate_two_phase


def test_vpool():
    rng = np.random.default_rng(3); N, M = 6000, 3
    x = rng.standard_normal((N, 2)); g = (rng.random(N) < 0.35).astype(int)
    py = 1 / (1 + np.exp(-(1.2 * x[:, 1] - 0.4))); y = (rng.random(N) < py).astype(float)
    pm = np.clip(py[:, None] + 0.25 * rng.standard_normal((N, M)), 0.02, 0.98); pred = (pm > 0.5).astype(int); p = pm.mean(1)
    U, V, names = total_columns(pred, g); est = Estimands(names, M, N); ctx0 = None
    st = make_strata(pred, g, "pattern"); ids, Nh, nh = neyman_allocation(st, U, V, p, est, 1500)
    ctx = DifContext(st, ids, U, V, p, est); ve = ctx.ve
    t = y[:, None] * U + (1 - y[:, None]) * V; T = t.sum(0); G = ve.grad(T); z = t @ G.T
    vtrue = N * z.var(0)
    vh = []
    for r in range(200):
        idx = draw(st, ids, nh, rng); o = estimate_two_phase(st, ids, Nh, nh, U, V, y[idx], idx, est, p, ctx=ctx)
        vh.append(o["se_pool"] ** 2)
    ratio = np.nanmedian(np.mean(vh, 0) / vtrue); print("median E[Vpool_hat]/N Var(z) =", round(float(ratio), 3))
    assert 0.9 < ratio < 1.1


if __name__ == "__main__":
    test_vpool(); print("ok")
