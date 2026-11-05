"""Checks for tcs.labeling_tests. Run:  python tests/test_labeling_tests.py   (or pytest tests/test_labeling_tests.py)

Exact: Sigma-hat diagonal equals the existing standard errors (difference estimator with and without floors, plug-in
estimator, two-phase total); the max-t critical value is the normal quantile for K = 1, the Sidak value for
independent statistics, and the normal quantile for perfectly correlated ones; Holm rejects whenever Bonferroni does;
Wald/NI/TOST p-values; the power calculators invert each other; the power-optimal allocation is feasible, its duality
gap is small, mirror descent and SLSQP agree, K = 1 recovers Neyman allocation, and it is never worse than the
default allocation on its own criterion.
"""
import sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, make_strata, neyman_allocation, estimate, draw
from tcs.labeling_dif import DifContext, estimate_dif_all, dif_allocation
from tcs.labeling_two_phase import estimate_two_phase
import tcs.labeling_tests as lt


def make_pool(seed=3, N=3000, M=5):
    rng = np.random.default_rng(seed)
    g = (rng.random(N) < 0.35).astype(int); x = rng.standard_normal((N, 3))
    y = (rng.random(N) < 1 / (1 + np.exp(-(x[:, 0] - 0.2 + 0.4 * g)))).astype(float)
    P = 1 / (1 + np.exp(-(x @ rng.standard_normal((3, M)) * 0.8 + 0.3 * g[:, None] * rng.standard_normal(M))))
    pred = (P > 0.5).astype(int); p = P.mean(1)
    U, V, names = total_columns(pred, g); est = Estimands(names, M, N)
    return dict(pred=pred, g=g, y=y, p=p, U=U, V=V, est=est, N=N)


def sample(d, L, seed=1, kind="pattern"):
    st = make_strata(d["pred"], d["g"], "pattern") if kind == "pattern" else np.zeros(d["N"], int)
    ids, Nh, nh = neyman_allocation(st, d["U"], d["V"], d["p"], d["est"], L)
    idx = draw(st, ids, nh, np.random.default_rng(seed)); return st, ids, Nh, nh, idx


def test_sigma_diagonal():
    d = make_pool(); U, V, y, p, est = d["U"], d["V"], d["y"], d["p"], d["est"]
    st, ids, Nh, nh, idx = sample(d, 400); ctx = DifContext(st, ids, U, V, p, est)
    th, ses = estimate_dif_all(st, ids, Nh, nh, U, V, y[idx], idx, est, p, ctx)
    for fl, key in ((None, "none"), ("zero", "zero"), ("max", "max")):
        J = lt.joint_estimate(st, ids, Nh, nh, U, V, y[idx], idx, est, p, "dif", fl, ctx)
        assert np.allclose(J["theta"], th) and np.allclose(J["se_design"], ses[key], rtol=1e-9, atol=1e-12), fl
        assert np.allclose(J["Sigma"], J["Sigma"].T)
        ev = np.linalg.eigvalsh(J["Sigma"]); assert ev.min() > -(1e-8 if fl is None else 1e-3) * ev.max()   # floors may break PSD slightly; max-t projects
    # plug-in
    thp, sep, _, _ = estimate(st, ids, Nh, nh, U, V, y[idx], idx, est)
    Jp = lt.joint_estimate(st, ids, Nh, nh, U, V, y[idx], idx, est, p, "plugin", None, ctx)
    assert np.allclose(Jp["theta"], thp) and np.allclose(Jp["se_design"], sep, rtol=1e-9, atol=1e-12)
    # two-phase
    tp = estimate_two_phase(st, ids, Nh, nh, U, V, y[idx], idx, est, p, ctx=ctx)
    J = lt.joint_estimate(st, ids, Nh, nh, U, V, y[idx], idx, est, p, "dif", None, ctx, two_phase=True)
    assert np.allclose(J["se_total"], tp["se_total"], rtol=1e-9) and np.allclose(J["se_pool"], tp["se_pool"], rtol=1e-9)


def test_sigma_floor_offdiag():
    d = make_pool(5, N=1500); U, V, y, p, est = d["U"], d["V"], d["y"], d["p"], d["est"]
    st, ids, Nh, nh, idx = sample(d, 150, 2); ctx = DifContext(st, ids, U, V, p, est)
    J = lt.joint_estimate(st, ids, Nh, nh, U, V, y[idx], idx, est, p, "dif", "zero", ctx)
    R = lt.correlation(J["Sigma"]); assert np.all(np.abs(R) <= 1 + 1e-12) and np.allclose(np.diag(R), 1)


def test_sigma_unbiased_exact():
    """Sigma-hat (all entries) is exactly unbiased for the covariance of the difference estimator, by enumeration."""
    import itertools
    rng = np.random.default_rng(2); N, M = 11, 3
    pred = rng.integers(0, 2, (N, M)); g = rng.integers(0, 2, N); y = rng.integers(0, 2, N).astype(float)
    U, V, names = total_columns(pred, g); est = Estimands(names, M, N)
    strata = np.array([0] * 4 + [1] * 4 + [2] * 3); ids = np.array([0, 1, 2]); Nh = np.array([4, 4, 3]); nh = np.array([2, 2, 2])
    p = np.clip(rng.random(N), 0.1, 0.9); ctx = DifContext(strata, ids, U, V, p, est)
    t_all = y[:, None] * U + (1 - y[:, None]) * V; Ttrue = t_all.sum(0); G0 = ctx.ve.grad(Ttrue)
    # linear targets: use the totals themselves (G = I) through a hand-made check of the covariance of the stratified mean
    groups = [list(itertools.combinations(np.flatnonzero(strata == h), n)) for h, n in zip(ids, nh)]
    S_hat, T_hat = [], []
    for combo in itertools.product(*groups):
        idx = np.array([i for c in combo for i in c])
        T, t, r, pos, cnt = lt.dif_totals(strata, ids, Nh, idx, y[idx], U, V, p, ctx=ctx)
        # covariance estimate of T itself (the linear map G = I): sum_h c_h cov(r_h)
        S = sum(Nh[k] ** 2 * (1 - nh[k] / Nh[k]) / nh[k] * np.cov(r[pos == k], rowvar=False, ddof=1) for k in range(3))
        S_hat.append(S); T_hat.append(T)
    S_hat, T_hat = np.array(S_hat), np.array(T_hat)
    assert np.allclose(S_hat.mean(0), np.cov(T_hat, rowvar=False, ddof=0), atol=1e-9)


def test_maxt_critical():
    for a in (0.01, 0.05, 0.10):
        assert abs(lt.maxt_critical(np.eye(1), a) - norm.ppf(1 - a / 2)) < 1e-12
        assert abs(lt.maxt_critical(np.eye(1), a, "greater") - norm.ppf(1 - a)) < 1e-12
        for K in (2, 5, 20):
            sid = norm.ppf((1 + (1 - a) ** (1 / K)) / 2)
            assert abs(lt.maxt_critical(np.eye(K), a) - sid) < 1e-12                      # exact branch
            assert abs(lt.maxt_critical(np.eye(K), a, exact=False) - sid) < 0.08         # Monte Carlo agrees
    K = 6; R1 = np.ones((K, K))                                                           # perfectly correlated
    assert abs(lt.maxt_critical(R1, 0.05) - norm.ppf(0.975)) < 0.06
    rng = np.random.default_rng(0)
    for _ in range(5):                                                                     # never above Bonferroni, never below unadjusted
        A = rng.standard_normal((10, 10)); R = A @ A.T; s = np.sqrt(np.diag(R)); R = R / np.outer(s, s)
        c = lt.maxt_critical(R, 0.05)
        assert norm.ppf(0.975) - 0.06 <= c <= norm.ppf(1 - 0.05 / 20) + 0.06
    assert lt.maxt_critical(R, 0.05) == lt.maxt_critical(R, 0.05)                         # fixed seed: deterministic
    # bands contain theta and have width 2 c se
    S = np.diag([1.0, 4.0]); lo, hi, c = lt.simultaneous_bands(np.zeros(2), S, 0.05, "bonferroni")
    assert np.allclose(hi, [c, 2 * c]) and abs(c - norm.ppf(1 - 0.05 / 4)) < 1e-12


def test_holm_vs_bonferroni():
    rng = np.random.default_rng(4)
    for _ in range(200):
        m = int(rng.integers(2, 30)); p = rng.random(m) ** rng.uniform(1, 6)
        pb, ph, ps = (lt.adjust_pvalues(p, k) for k in ("bonferroni", "holm", "sidak"))
        assert np.all(ph <= pb + 1e-12) and np.all(ps <= pb + 1e-12) and np.all(pb >= p)
        for a in (0.01, 0.05, 0.1):
            assert np.all((pb <= a) <= (ph <= a))                                          # Bonferroni rejections subset of Holm
    # Holm adjusted p-values are monotone in p
    p = np.array([0.001, 0.02, 0.04, 0.3]); ph = lt.adjust_pvalues(p, "holm")
    assert np.allclose(ph, [0.004, 0.06, 0.08, 0.3])
    # power ordering on a simulated family: Holm >= Bonferroni, max-t >= Bonferroni (up to MC noise), all <= unadjusted
    rng = np.random.default_rng(1); K = 30; th = np.r_[np.zeros(20), np.full(10, 3.2)]; S = np.eye(K)
    cnt = {m: 0 for m in lt.METHODS}
    for _ in range(300):
        J = lt.simultaneous_test(th + rng.standard_normal(K), S, 0.0, 0.05, ndraw=4000)
        for m in lt.METHODS: cnt[m] += J[m]["reject"][20:].mean()
    assert cnt["holm"] >= cnt["bonferroni"] and cnt["none"] >= cnt["holm"]
    assert cnt["maxt"] >= cnt["bonferroni"] - 8.0                          # Sidak-type crit. is almost Bonferroni here; MC noise


def test_single_tests():
    th, se = np.array([0.3, -1.2, 2.5]), np.array([1.0, 1.0, 1.0])
    assert np.allclose(lt.wald_test(th, se)["p"], 2 * norm.sf(np.abs(th)))
    assert np.allclose(lt.wald_test(th, se, 0, "greater")["p"], norm.sf(th))
    ni = lt.noninferiority_test(0.5, 0.25, 1.0); assert abs(ni["p"] - norm.cdf((0.5 - 1.0) / 0.25)) < 1e-12
    eq = lt.equivalence_test(0.1, 0.1, 0.5); assert abs(eq["p"] - max(norm.cdf((0.1 - 0.5) / 0.1), norm.sf((0.1 + 0.5) / 0.1))) < 1e-12
    assert lt.equivalence_test(0.0, 0.1, 0.5)["p"] < lt.equivalence_test(0.4, 0.1, 0.5)["p"]
    z = lt.wald_z(np.array([0.0, 0.2]), np.array([0.0, 0.0])); assert z[0] == 0 and np.isinf(z[1])
    assert lt.wald_test(np.array([0.0, 0.2]), np.array([0.0, 0.0]))["p"].tolist() == [1.0, 0.0]


def test_power():
    for a, pw in ((0.05, 0.8), (0.01, 0.9)):
        v = lt.required_variance(0.1, a, pw); assert abs(lt.power_two_sided(0.1, v, a) - pw) < 1e-9
        v1 = lt.required_variance(0.1, a, pw, two_sided=False); assert abs(lt.power_one_sided(0.1, v1, a) - pw) < 1e-9
    assert abs(lt.power_two_sided(0.0, 1.0, 0.05) - 0.05) < 1e-12                         # size
    c = 5.0; N = 8000; L = lt.budget_for_power(0.1, c, 0.05, 0.8, N=N)
    assert abs(lt.power_two_sided(0.1, c * (1 / L - 1 / N), 0.05) - 0.8) < 1e-9
    assert abs(lt.budget_for_power(0.1, 4 * c, 0.05, 0.8) / lt.budget_for_power(0.1, c, 0.05, 0.8) - 4) < 1e-9   # ARE = variance ratio
    Ls = np.array([100, 200, 400, 800]); pw = np.array([0.4, 0.6, 0.9, 0.99])
    assert 200 < lt.budget_from_curve(Ls, pw) < 400 and np.isnan(lt.budget_from_curve(Ls, pw * 0.5))


def _alloc_problem():
    d = make_pool(7, N=4000, M=5); U, V, p, est = d["U"], d["V"], d["p"], d["est"]
    st = make_strata(d["pred"], d["g"], "pattern"); ids = np.unique(st)
    Nh, C = lt.design_variance_matrix(st, ids, U, V, p, est, "resid")
    th = est.value((d["y"][:, None] * U + (1 - d["y"][:, None]) * V).sum(0)); delta = np.maximum(np.abs(th), 0.02)
    return d, st, ids, Nh, C, delta


def test_power_optimal():
    d, st, ids, Nh, C, delta = _alloc_problem(); L = 300
    r = lt.power_optimal_allocation(Nh, C, delta, L, floor=10, method="mirror")
    assert r["n"].sum() == L and np.all(r["n"] >= np.minimum(Nh, 10)) and np.all(r["n"] <= Nh)
    assert r["value_dual"] <= r["value_primal"] * (1 + 1e-9) and r["gap"] < 0.01, r["gap"]
    assert abs(r["n_cont"].sum() - L) < 1e-6
    r2 = lt.power_optimal_allocation(Nh, C, delta, L, floor=10, method="slsqp")
    assert abs(r2["value_primal"] / r["value_primal"] - 1) < 0.01 and r2["value_primal"] >= r["value_dual"] * (1 - 1e-6)
    # never worse than the default (weights 1/S^2) and than proportional allocation on the maximin criterion
    ids_, Nh_, nh_ = neyman_allocation(st, d["U"], d["V"], d["p"], d["est"], L)
    crit = lambda n: (lt.predicted_variance(n, Nh, C) / delta ** 2).max()
    assert r["value_int"] <= crit(nh_) * (1 + 1e-6) and r["value_int"] <= crit(np.minimum(Nh, np.maximum(10, np.round(L * Nh / Nh.sum()))).astype(int)) * 1.0001
    # dual characterisation: n_h proportional to N_h sqrt(sum_k w_k S_hk^2 / delta_k^2) where unconstrained
    w = r["w"]; S2 = C / Nh[:, None] ** 2; sc = Nh * np.sqrt((S2[:, r["active"]] * (w / delta[r["active"]] ** 2)).sum(1))
    free = (r["n_cont"] > np.minimum(Nh, 10) + 1e-6) & (r["n_cont"] < Nh - 1e-6)
    ratio = r["n_cont"][free] / sc[free]; assert np.ptp(ratio) / ratio.mean() < 1e-6
    # K = 1 recovers Neyman allocation (continuous)
    r1 = lt.power_optimal_allocation(Nh, C[:, :1], np.array([0.1]), L, floor=10, method="mirror")
    nn = lt._inner_alloc(C[:, 0], Nh, np.minimum(Nh, 10), L); assert np.allclose(r1["n_cont"], nn, atol=1e-6)
    assert abs(nn.sum() - L) < 1e-8


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"): f(); print("ok", name)
