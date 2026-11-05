"""Hypothesis tests, familywise error control, power and power-optimal design for PATTERN-DIF contrasts.

Synthetic population (the generative model of experiments/simulate_nonsmooth.py, nine fixed scoring rules):
x ~ N(0, I_5), g ~ Bernoulli(0.3), y ~ Bernoulli(sigmoid(BETA'x - 0.2 + 0.3 g)); the label model does not depend on x4, x5.
Models 0-3 use x4, their TWINS 4-7 use x5 with identical weights elsewhere (same scale and group offset), model 8 uses neither.
(x4, x5) are iid N(0,1) and independent of (x1..x3, g, y), so swapping them is measure preserving for (y, g): the joint law
of (twin prediction, y, g) equals that of (base prediction, y, g). Hence TPR gap, FPR gap and error rate of base and twin
coincide in the POPULATION and the three contrasts of each of the four pairs (i, i+4) are exactly zero (12 true nulls of
108), while the predictions differ on a sizeable part of every pool (non-trivial variance). Population contrasts: totals
over 2e7 draws; the 12 nulls are set to 0 (the Monte-Carlo values are stored as a check).

Studies (python experiments/testing_sim.py run <study> R | report):
  syn   B1 size (pool-conditional target theta_U and population target theta_pop with two-phase variance, two-sided,
        one-sided, TOST at the boundary) and B2 FWER/power (synthetic) for SRS, PATTERN (plug-in), PATTERN-DIF;
        pools N in {2000, 8000}, L in {250, 500, 1000, 2000} with L <= N/4, new pool in every replication.
  real  B2 simultaneous coverage of the family of pool contrasts (live contrasts) on the 11 cached pools.
  b3    power curves, label budget for 80% power, comparison with the asymptotic (oracle-variance) prediction.
  b4    power-optimal (maximin) allocation versus the default allocation.
Results: results/testing_sim.json (one key per study); tables and figures by `report`.
"""
import sys, os, json, zlib, time
import numpy as np
from itertools import combinations
from multiprocessing import Pool
sys.path.insert(0, ".")
from scipy.stats import norm
from experiments import simulate_nonsmooth as sn
from experiments.label_efficiency_dif import setup as real_setup, KEYS as REAL_KEYS
from tcs.labeling import total_columns, Estimands, make_strata, neyman_allocation, proportional_allocation, draw, estimate
from tcs.labeling_baselines import VecEstimands
from tcs.labeling_dif import DifContext, dif_allocation, estimate_dif_all
import tcs.labeling_tests as lt

M, NPOP = 9, 20_000_000
OUT = "results/testing_sim.json"
METHODS = lt.METHODS                               # none, bonferroni, holm, maxt
MLAB = {"none": "Unadjusted", "bonferroni": "Bonferroni", "holm": "Holm", "maxt": "Max-$t$"}
DLAB = {"srs": "SRS", "pat": "PATTERN", "dif": "PATTERN-DIF"}
ALPHAS = (0.01, 0.05, 0.10)
NS = (2000, 8000); LS = (250, 500, 1000, 2000)
PAIRS = list(combinations(range(M), 2))
NULL_PAIRS = [PAIRS.index((i, i + 4)) for i in range(4)]
NULL_IDX = np.array([3 * k + r for k in NULL_PAIRS for r in range(3)])
KINDS = np.array([k for _ in PAIRS for k in ("dTPR", "dFPR", "dERR")])
S = {}                                             # shared (fork) state
TRUTH = None


# ------------------------------------------------------------------------------------------- synthetic population
def syn_design(seed=11):
    r = np.random.default_rng(seed); D = 5
    beta = np.zeros((M, D)); s = np.zeros(M); o = np.zeros(M)
    for i in range(4):
        b = sn.BETA + 0.45 * r.standard_normal(D); b[3] = b[4] = 0
        c = r.choice([-1, 1]) * r.uniform(0.6, 1.0); s[i] = np.exp(r.uniform(-0.7, 0.7)); o[i] = r.uniform(-0.7, 0.7)
        beta[i] = b; beta[i, 3] = c; beta[i + 4] = b; beta[i + 4, 4] = c; s[i + 4] = s[i]; o[i + 4] = o[i]
    b = sn.BETA + 0.45 * r.standard_normal(D); b[3] = b[4] = 0; beta[8] = b; s[8] = np.exp(r.uniform(-0.7, 0.7)); o[8] = r.uniform(-0.7, 0.7)
    return dict(beta=beta, s=s, o=o)


DES = syn_design()


def syn_models(x, g):
    p = sn.sig(DES["s"] * (x @ DES["beta"].T + DES["o"] * g[:, None])); return p, (p > 0.5).astype(int)


def syn_truth():
    f = "results/testing_syn_truth.npz"
    if os.path.exists(f):
        z = np.load(f); return z["theta"], z["mc"]
    rng = np.random.default_rng(2025); T = 0; left = NPOP; names = None
    while left > 0:
        c = min(500_000, left); left -= c
        x, g, y = sn.draw(c, rng); p, pred = syn_models(x, g); U, V, names = total_columns(pred, g)
        T = T + (y[:, None] * U + (1 - y[:, None]) * V).sum(0)
    mc = Estimands(names, M, NPOP).value(T); th = mc.copy(); th[NULL_IDX] = 0.0
    np.savez(f, theta=th, mc=mc); return th, mc


def make_pool(N, rng):
    x, g, y = sn.draw(N, rng); p, pred = syn_models(x, g); U, V, names = total_columns(pred, g)
    est = Estimands(names, M, N)
    return dict(pred=pred, g=g, y=y.astype(float), p=p.mean(1), U=U, V=V, est=est, ve=VecEstimands(est), N=N)


def syn_fixed(N=20000, seed=7):
    """One fixed synthetic pool, in the format of experiments.label_efficiency_dif.setup (pool-conditional target)."""
    d = make_pool(N, np.random.default_rng(seed)); U, V, y = d["U"], d["V"], d["y"]
    d["truth"] = d["est"].value((y[:, None] * U + (1 - y[:, None]) * V).sum(0)); d["live"] = np.ones(len(d["truth"]), bool)
    d["M"] = M; d["kinds"] = KINDS; return d


# ------------------------------------------------------------------------------------------- B1 / B2 synthetic
def rep_syn(args):
    N, L, seed = args
    rng = np.random.default_rng(seed); d = make_pool(N, rng); U, V, y, p, est = d["U"], d["V"], d["y"], d["p"], d["est"]
    th_pool = est.value((y[:, None] * U + (1 - y[:, None]) * V).sum(0))
    one = np.zeros(N, int); ids0, Nh0, nh0 = proportional_allocation(one, L)
    stp = make_strata(d["pred"], d["g"], "pattern"); idsp, Nhp, nhp = neyman_allocation(stp, U, V, p, est, L)
    ctx0 = DifContext(one, ids0, U, V, p, est); ctxp = DifContext(stp, idsp, U, V, p, est)
    idx0 = draw(one, ids0, nh0, rng); idxp = draw(stp, idsp, nhp, rng)
    runs = {"srs": (one, ids0, Nh0, nh0, idx0, "plugin", ctx0), "pat": (stp, idsp, Nhp, nhp, idxp, "plugin", ctxp),
            "dif": (stp, idsp, Nhp, nhp, idxp, "dif", ctxp)}
    out = {"th_pool": th_pool}
    for nm, (st, ids, Nh, nh, idx, kind, ctx) in runs.items():
        J = lt.joint_estimate(st, ids, Nh, nh, U, V, y[idx], idx, est, p, kind, None, ctx, two_phase=True)
        if J is None: out[nm] = None; continue
        th, Sd, St = J["theta"], J["Sigma"], J["Sigma_total"]
        A = lt.simultaneous_test(th, Sd, th_pool, 0.05)                    # pool target, all nulls true
        B = lt.simultaneous_test(th, St, TRUTH, 0.05)                      # population target, all nulls true
        C = lt.simultaneous_test(th, St, 0.0, 0.05)                        # H0: theta_k = 0 for all k (12 true)
        out[nm] = dict(theta=th.astype(np.float32), se_d=J["se_design"].astype(np.float32), se_t=J["se_total"].astype(np.float32),
                       fam_pool=np.array([A[m]["reject"].any() for m in METHODS]), fam_pop=np.array([B[m]["reject"].any() for m in METHODS]),
                       null_rej=np.array([C[m]["reject"] for m in METHODS]), crit=C["crit"])
    return out


def _rates(theta, se, th0, alphas=ALPHAS, tost_mask=None):
    """Rejection rates (per contrast, over replications) at the boundary of the null. theta, se: (R, K); th0 (R, K) or (K,)."""
    z = lt.wald_z(theta, se, th0); res = {}
    for a in alphas:
        za = norm.ppf(1 - a / 2); zo = norm.ppf(1 - a)
        two = (np.abs(z) > za).mean(0); less = (z < -zo).mean(0); gre = (z > zo).mean(0)
        marg = np.abs(np.broadcast_to(th0, theta.shape))
        # TOST at the boundary: margin |theta0|, true value = theta0 = +/- margin; H0: |theta| >= margin
        p_t = lt.equivalence_test(theta, se, marg)["p"]
        tost = (p_t <= a).mean(0)
        if tost_mask is not None: tost = tost[tost_mask]
        res[str(a)] = dict(two=[float(two.mean()), float(two.max()), float(two.min())], less=[float(less.mean()), float(less.max())],
                           greater=[float(gre.mean()), float(gre.max())], tost=[float(tost.mean()), float(tost.max())])
    return res


def agg_syn(N, L, R, res):
    row = dict(N=N, L=L, R=R); thp = np.array([r["th_pool"] for r in res])
    mat = (np.abs(TRUTH) >= 0.02); false_idx = np.setdiff1d(np.arange(len(TRUTH)), NULL_IDX)
    mat_idx = np.flatnonzero(mat & ~np.isin(np.arange(len(TRUTH)), NULL_IDX))
    for nm in ("srs", "pat", "dif"):
        ok = np.array([r[nm] is not None for r in res]); rr = [r[nm] for r in res if r[nm] is not None]
        e = dict(fail=float(1 - ok.mean()))
        th = np.array([r["theta"] for r in rr], float); sd = np.array([r["se_d"] for r in rr], float); st = np.array([r["se_t"] for r in rr], float)
        tp = thp[ok]
        e["b1_pool"] = _rates(th, sd, tp, tost_mask=mat)               # pool-conditional test of the pool value
        e["b1_pop_total"] = _rates(th, st, TRUTH[None, :], tost_mask=mat)    # two-phase test of the population value
        e["b1_pop_design"] = _rates(th, sd, TRUTH[None, :], tost_mask=mat)   # pool-conditional test of the population value
        e["varpi_median"] = float(np.median(((st ** 2 - sd ** 2) / np.maximum(sd ** 2, 1e-30)).mean(0)))
        e["fam_pool_cov"] = {m: float(1 - np.mean([x["fam_pool"][i] for x in rr])) for i, m in enumerate(METHODS)}
        e["fam_pop_cov"] = {m: float(1 - np.mean([x["fam_pop"][i] for x in rr])) for i, m in enumerate(METHODS)}
        nr = np.array([x["null_rej"] for x in rr])                       # (R, 4, K)
        e["fwer"] = {m: float(nr[:, i, :][:, NULL_IDX].any(1).mean()) for i, m in enumerate(METHODS)}
        e["power_false"] = {m: float(nr[:, i, :][:, false_idx].mean()) for i, m in enumerate(METHODS)}
        e["power_material"] = {m: float(nr[:, i, :][:, mat_idx].mean()) for i, m in enumerate(METHODS)}
        e["n_material"] = int(len(mat_idx)); e["crit_maxt_mean"] = float(np.mean([x["crit"] for x in rr]))
        e["null_reject_each_unadj"] = float(nr[:, 0, :][:, NULL_IDX].mean())
        row[nm] = e
    return row


def load_out():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save_out(key, val):
    J = load_out(); J[key] = val; json.dump(J, open(OUT, "w"), indent=1)


def run_syn(R):
    global TRUTH
    TRUTH, mc = syn_truth(); rows = []
    print("null contrasts MC values (should be ~0): max |.| =", float(np.abs(mc[NULL_IDX]).max()), "; min |non-null| =", float(np.abs(np.delete(TRUTH, NULL_IDX)).min()), flush=True)
    for N in NS:
        for L in LS:
            if L > N / 4: continue
            t0 = time.time(); Rr = R
            with Pool(2) as pool:
                res = pool.map(rep_syn, [(N, L, zlib.crc32(f"tsyn{N}_{L}".encode()) % 100003 * 1000 + r) for r in range(Rr)], chunksize=5)
            row = agg_syn(N, L, Rr, res); rows.append(row); e = row["dif"]
            print(N, L, f"{time.time()-t0:.0f}s fail={e['fail']:.3f} size5: pool {e['b1_pool']['0.05']['two'][:2]} pop {e['b1_pop_total']['0.05']['two'][:2]} varpi={e['varpi_median']:.2f} "
                  f"FWER {e['fwer']} powmat {e['power_material']}", flush=True)
            save_out("syn", dict(R=R, null_idx=NULL_IDX.tolist(), truth=TRUTH.tolist(), mc_null_max=float(np.abs(mc[NULL_IDX]).max()), rows=rows))


# ------------------------------------------------------------------------------------------- B2 real pools
def rep_real(args):
    key, L, seed = args
    d = S[key]; D = S[(key, L)]; rng = np.random.default_rng(seed); U, V, y, p, est = d["U"], d["V"], d["y"], d["p"], d["est"]
    live = d["live"]; truth = d["truth"][live]; out = []
    idx0 = draw(D["srs"][0], D["srs"][1], D["srs"][3], rng); idxp = draw(D["pat"][0], D["pat"][1], D["pat"][3], rng)
    jobs = [("srs", idx0, "plugin", None), ("pat", idxp, "plugin", None), ("dif", idxp, "dif", None), ("dif_fz", idxp, "dif", "zero")]
    for nm, idx, kind, fl in jobs:
        st, ids, Nh, nh, ctx = D["srs" if nm == "srs" else "pat"]
        J = lt.joint_estimate(st, ids, Nh, nh, U, V, y[idx], idx, est, p, kind, fl, ctx)
        if J is None: out.append(None); continue
        th = J["theta"][live]; Sg = J["Sigma"][np.ix_(live, live)]
        A = lt.simultaneous_test(th, Sg, truth, 0.05)
        out.append(dict(rej=np.array([A[m]["reject"].any() for m in METHODS]), crit=A["crit"], K=int(live.sum()),
                        zmax=float(np.nanmax(np.abs(A["z"][np.isfinite(A["z"])])) if np.isfinite(A["z"]).any() else np.nan)))
    return out


def run_real(R, keys=REAL_KEYS):
    rows = [r for r in load_out().get("real", {}).get("rows", []) if r["dataset"] not in keys]
    for key in keys:
        d = real_setup(key)
        if min(d["Pg"] + d["Qg"]) < 30: print(key, "skipped", flush=True); continue
        S[key] = d; U, V, p, est = d["U"], d["V"], d["p"], d["est"]
        for L in (250, 500, 1000, 2000):
            if L > d["N"] / 4: continue
            one = np.zeros(d["N"], int); i0, N0, n0 = proportional_allocation(one, L)
            sp = make_strata(d["pred"], d["g"], "pattern"); ip, Np, npat = neyman_allocation(sp, U, V, p, est, L)
            S[(key, L)] = dict(srs=(one, i0, N0, n0, DifContext(one, i0, U, V, p, est)), pat=(sp, ip, Np, npat, DifContext(sp, ip, U, V, p, est)))
            t0 = time.time()
            with Pool(2) as pool:
                res = pool.map(rep_real, [(key, L, zlib.crc32(key.encode()) % 9973 + 17 * L + r + 500000) for r in range(R)], chunksize=5)
            row = dict(dataset=key, N=d["N"], L=L, R=R, K=int(d["live"].sum()))
            for j, nm in enumerate(("srs", "pat", "dif", "dif_fz")):
                ok = [r[j] for r in res if r[j] is not None]
                e = dict(fail=float(1 - len(ok) / R), cover={m: float(1 - np.mean([x["rej"][i] for x in ok])) for i, m in enumerate(METHODS)},
                         crit_maxt=float(np.mean([x["crit"] for x in ok])))
                row[nm] = e
            rows.append(row)
            print(key, L, f"{time.time()-t0:.0f}s K={row['K']}", {nm: {m: round(v, 3) for m, v in row[nm]["cover"].items()} for nm in ("dif",)}, flush=True)
            save_out("real", dict(R=R, rows=rows))
        del S[key]


# ------------------------------------------------------------------------------------------- oracle variances
def oracle_S2(d, strata, ids, kind):
    """(H, K) finite-population variances (ddof 1) of the influence values (kind 'plugin': G t_i; 'dif': G (t_i - f_i)),
    G = gradient at the true totals of the pool. Var_k(n) = sum_h N_h^2 (1/n_h - 1/N_h) S2_hk."""
    U, V, y, p, ve = d["U"], d["V"], d["y"], d["p"], d["ve"]
    t = y[:, None] * U + (1 - y[:, None]) * V; G = ve.grad(t.sum(0))
    r = t if kind == "plugin" else t - (p[:, None] * U + (1 - p[:, None]) * V)
    e = r @ G.T
    return np.array([e[strata == h].var(0, ddof=1) for h in ids])


def oracle_var(n, Nh, S2):
    n = np.asarray(n, float); Nh = np.asarray(Nh, float); return ((Nh ** 2 * (1 / n - 1 / Nh))[:, None] * S2).sum(0)


# ------------------------------------------------------------------------------------------- B3 power curves
def b3_prepare(key, d):
    N = d["N"]; Lmax = int(min(4000, N // 4))
    if Lmax < 400: return None
    U, V, p, est = d["U"], d["V"], d["p"], d["est"]; th = d["truth"]; live = d["live"] & (np.abs(th) > 0)
    one = np.zeros(N, int); sp = make_strata(d["pred"], d["g"], "pattern"); ids_p = np.unique(sp)
    S2 = dict(srs=oracle_S2(d, one, [0], "plugin"), pat=oracle_S2(d, sp, ids_p, "plugin"), dif=oracle_S2(d, sp, ids_p, "dif"))
    Ld = np.unique(np.round(np.geomspace(200, Lmax, 40)).astype(int)); K = len(th)
    Vd = {m: np.zeros((len(Ld), K)) for m in S2}
    for i, L in enumerate(Ld):
        ip, Np, npat = neyman_allocation(sp, U, V, p, est, int(L))
        Vd["srs"][i] = oracle_var([L], [N], S2["srs"]); Vd["pat"][i] = oracle_var(npat, Np, S2["pat"]); Vd["dif"][i] = oracle_var(npat, Np, S2["dif"])
    pw = {m: lt.power_two_sided(th[None, :], Vd[m]) for m in Vd}
    L80 = {m: np.array([lt.budget_from_curve(Ld, pw[m][:, k]) for k in range(K)]) for m in Vd}
    good = np.flatnonzero(live & np.isfinite(L80["srs"]) & np.isfinite(L80["dif"]) & np.isfinite(L80["pat"]))
    if len(good) == 0: return None
    order = good[np.argsort(L80["srs"][good])]; pick = order[np.unique(np.round(np.linspace(0, len(order) - 1, min(6, len(order)))).astype(int))]
    Lg = np.unique(np.round(np.geomspace(200, Lmax, 7)).astype(int))
    pred_grid = {}; vg = {m: np.zeros((len(Lg), K)) for m in S2}
    for i, L in enumerate(Lg):
        ip, Np, npat = neyman_allocation(sp, U, V, p, est, int(L))
        vg["srs"][i] = oracle_var([L], [N], S2["srs"]); vg["pat"][i] = oracle_var(npat, Np, S2["pat"]); vg["dif"][i] = oracle_var(npat, Np, S2["dif"])
    return dict(Ld=Ld, Lg=Lg, pick=pick, L80_pred={m: L80[m][pick] for m in L80}, V_grid={m: vg[m][:, pick] for m in vg},
                pw_grid={m: lt.power_two_sided(th[None, pick], vg[m][:, pick]) for m in vg}, theta=th[pick])


def rep_b3(args):
    key, L, seed = args
    d = S[key]; D = S[(key, L)]; sel = S[(key, "sel")]; rng = np.random.default_rng(seed); U, V, y, p, est = d["U"], d["V"], d["y"], d["p"], d["est"]
    st0, i0, N0, n0 = D["srs"]; stp, ip, Np, npat, ctx = D["pat"]
    idx0 = draw(st0, i0, n0, rng); idxp = draw(stp, ip, npat, rng)
    th0, se0, _, _ = estimate(st0, i0, N0, n0, U, V, y[idx0], idx0, est)
    thp, sep, _, _ = estimate(stp, ip, Np, npat, U, V, y[idxp], idxp, est)
    thd, sed = estimate_dif_all(stp, ip, Np, npat, U, V, y[idxp], idxp, est, p, ctx)
    return np.array([[th0[sel], se0[sel]], [thp[sel], sep[sel]], [thd[sel], sed["none"][sel]]], np.float32)


def run_b3(R, include=None):
    pools = [("synthetic", None)] + [(k, None) for k in REAL_KEYS]
    out = []
    for key, _ in pools:
        d = syn_fixed() if key == "synthetic" else real_setup(key)
        if key != "synthetic" and min(d["Pg"] + d["Qg"]) < 30: continue
        prep = b3_prepare(key, d)
        if prep is None: print(key, "no contrasts / budget too small", flush=True); continue
        S[key] = d; sel = prep["pick"]; S[(key, "sel")] = sel; N = d["N"]; Lg = prep["Lg"]
        U, V, p, est = d["U"], d["V"], d["p"], d["est"]; one = np.zeros(N, int); sp = make_strata(d["pred"], d["g"], "pattern")
        pw = np.zeros((3, len(Lg), len(sel))); mse = np.zeros((3, len(Lg), len(sel))); sev = np.zeros((3, len(Lg), len(sel))); fail = np.zeros(len(Lg))
        t0 = time.time()
        for i, L in enumerate(Lg):
            i0, N0, n0 = proportional_allocation(one, int(L)); ip, Np, npat = neyman_allocation(sp, U, V, p, est, int(L))
            S[(key, int(L))] = dict(srs=(one, i0, N0, n0), pat=(sp, ip, Np, npat, DifContext(sp, ip, U, V, p, est)))
            with Pool(2) as pool:
                res = np.array(pool.map(rep_b3, [(key, int(L), zlib.crc32(key.encode()) % 9973 + 31 * int(L) + r + 900000) for r in range(R)], chunksize=10), float)
            ok = np.isfinite(res).all(axis=(1, 2, 3)); fail[i] = 1 - ok.mean(); res = res[ok]
            for m in range(3):
                th, se = res[:, m, 0, :], res[:, m, 1, :]; tr = d["truth"][sel]
                pw[m, i] = (np.abs(lt.wald_z(th, se, 0.0)) > norm.ppf(0.975)).mean(0)
                mse[m, i] = ((th - tr) ** 2).mean(0); sev[m, i] = (se ** 2).mean(0)
        L80 = np.array([[lt.budget_from_curve(Lg, pw[m, :, k]) for k in range(len(sel))] for m in range(3)])
        L80g = {m: np.array([lt.budget_from_curve(Lg, prep["pw_grid"][nm][:, k]) for k in range(len(sel))]) for m, nm in enumerate(("srs", "pat", "dif"))}
        rec = dict(dataset=key, N=N, R=R, Lg=Lg.tolist(), contrasts=sel.tolist(), kinds=d["kinds"][sel].tolist() if "kinds" in d else None,
                   theta=prep["theta"].tolist(), power_emp=pw.tolist(), power_pred=[prep["pw_grid"][m].tolist() for m in ("srs", "pat", "dif")],
                   mse=mse.tolist(), mean_se2=sev.tolist(), var_oracle=[prep["V_grid"][m].tolist() for m in ("srs", "pat", "dif")],
                   L80_emp=L80.tolist(), L80_pred_grid=[L80g[m].tolist() for m in range(3)], L80_pred_dense=[prep["L80_pred"][m].tolist() for m in ("srs", "pat", "dif")],
                   fail_max=float(fail.max()))
        out.append(rec); save_out("b3", dict(R=R, rows=out))
        pr = np.array(rec["L80_pred_dense"][0]) / np.array(rec["L80_pred_dense"][2]); er = L80[0] / L80[2]
        print(key, f"{time.time()-t0:.0f}s", "contrasts", len(sel), "pred ratio", np.round(pr, 2), "emp ratio", np.round(er, 2), flush=True)
        for k in [key] + [kk for kk in list(S) if isinstance(kk, tuple) and kk[0] == key]: S.pop(k, None)


# ------------------------------------------------------------------------------------------- B4 power-optimal design
B4_POOLS = ("synthetic", "adult-sex", "bank", "electricity")
DELTA_COMMON = 0.05; DELTA_MIN = 0.02


def rep_b4(args):
    key, L, name, seed = args
    d = S[key]; st, ids, Nh, nh, ctx = S[(key, L, name)]; rng = np.random.default_rng(seed)
    idx = draw(st, ids, nh, rng); th, ses = estimate_dif_all(st, ids, Nh, nh, d["U"], d["V"], d["y"][idx], idx, d["est"], d["p"], ctx)
    return np.array([th, ses["none"]], np.float32)


def run_b4(R):
    rows = []
    for key in B4_POOLS:
        d = syn_fixed() if key == "synthetic" else real_setup(key)
        S[key] = d; U, V, p, est = d["U"], d["V"], d["p"], d["est"]; th = d["truth"]; live = d["live"]
        sp = make_strata(d["pred"], d["g"], "pattern"); ids = np.unique(sp)
        Nh, C = lt.design_variance_matrix(sp, ids, U, V, p, est, "resid"); S2o = oracle_S2(d, sp, ids, "dif")
        m1 = live & (np.abs(th) >= DELTA_MIN)                      # E1: contrasts with |theta| >= 0.02, effect = |theta| (H0: theta = 0)
        d1 = np.where(m1, np.abs(th), np.nan); d2 = np.where(live, DELTA_COMMON, np.nan)   # E2: all live contrasts, common effect 0.05
        sgn = np.where(th >= 0, 1.0, -1.0); th0 = {"E1": np.zeros(len(th)), "E2": th - sgn * DELTA_COMMON}
        dl = {"E1": np.maximum(np.abs(th), 1e-12), "E2": np.full(len(th), DELTA_COMMON)}; sets = {"E1": m1, "E2": live}
        for L in (500, 1000):
            allocs = {}
            ids_, Nh_, n_def = neyman_allocation(sp, U, V, p, est, L); allocs["default"] = n_def
            allocs["resid"] = dif_allocation(sp, U, V, p, est, L, "resid")[2]
            r1 = lt.power_optimal_allocation(Nh, C, d1, L); r2 = lt.power_optimal_allocation(Nh, C, d2, L)
            allocs["maximin_E1"] = r1["n"]; allocs["maximin_E2"] = r2["n"]
            info = dict(gap_E1=r1["gap"], gap_E2=r2["gap"])
            for nm, n in allocs.items(): S[(key, L, nm)] = (sp, ids, Nh, n, DifContext(sp, ids, U, V, p, est))
            row = dict(dataset=key, N=d["N"], L=L, R=R, K=int(live.sum()), K1=int(m1.sum()), n_strata=int(len(ids)), alloc={}, **info)
            for nm, n in allocs.items():
                t0 = time.time()
                with Pool(2) as pool:
                    res = np.array(pool.map(rep_b4, [(key, L, nm, zlib.crc32(key.encode()) % 9973 + 31 * L + r + 700000) for r in range(R)], chunksize=10), float)
                ok = np.isfinite(res).all(axis=(1, 2)); res = res[ok]; thh, se = res[:, 0, :], res[:, 1, :]
                e = dict(n=[int(x) for x in n], fail=float(1 - ok.mean()), crit={})
                Vo = oracle_var(n, Nh, S2o)
                for ev in ("E1", "E2"):
                    pw = (np.abs(lt.wald_z(thh, se, th0[ev])) > norm.ppf(0.975)).mean(0)[sets[ev]]
                    pp = lt.power_two_sided(dl[ev], Vo)[sets[ev]]
                    e[ev] = dict(min=float(pw.min()), mean=float(pw.mean()), q10=float(np.quantile(pw, 0.10)), pred_min=float(pp.min()), pred_mean=float(pp.mean()),
                                 pred_q10=float(np.quantile(pp, 0.10)), n_below80=int((pw < 0.8).sum()), pred_n_below80=int((pp < 0.8).sum()))
                    e["crit"][ev] = float((Vo / dl[ev] ** 2)[sets[ev]].max()); e[ev]["n_targets"] = int(sets[ev].sum())
                row["alloc"][nm] = e
                print(key, L, nm, f"{time.time()-t0:.0f}s", {ev: (round(e[ev]["min"], 3), round(e[ev]["mean"], 3), round(e[ev]["pred_min"], 3)) for ev in ("E1", "E2")}, flush=True)
            rows.append(row); save_out("b4", dict(R=R, delta_min=DELTA_MIN, delta_common=DELTA_COMMON, rows=rows))
        S.clear()


# ------------------------------------------------------------------------------------------- reporting
def _f(x, d=3):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _ideal_max(R, a, K=108, nrep=2000, seed=1):
    """Mean of the maximum over K contrasts of a Binomial(R, a)/R rate (what a perfectly calibrated test shows as 'worst')."""
    return float(np.mean(np.random.default_rng(seed).binomial(R, a, size=(nrep, K)).max(1) / R))


NAME = {"synthetic": "Synthetic pool", "adult-sex": "Adult (sex)", "adult-race": "Adult (race)", "bank": "Bank marketing", "credit-card": "Credit card",
        "law-school": "Law school", "magic": "MAGIC", "eeg": "EEG eye state", "phoneme": "Phoneme", "spambase": "Spambase", "churn": "Churn",
        "electricity": "Electricity"}


def _sz(e, key, a="0.05", k="two"):
    v = e[key][a][k]; return v[0], v[1]


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    J = load_out(); os.makedirs("manuscript_v2/tables", exist_ok=True); os.makedirs("manuscript_v2/figures", exist_ok=True)
    T = "manuscript_v2/tables/"; summ = {}
    syn, real, b3, b4 = J.get("syn"), J.get("real"), J.get("b3"), J.get("b4")

    # ---------------- synthetic: size and FWER
    if syn:
        rows = syn["rows"]; R = syn["R"]
        out = [r"\begin{tabular}{rr r rr rr rrrr}\toprule",
               r" & & & \multicolumn{2}{c}{Size, pool value} & \multicolumn{2}{c}{Size, population value} & \multicolumn{4}{c}{FWER, population nulls} \\",
               r"$N$ & $L$ & $\varpi$ & mean & max & mean & max & Unadj. & Bonf. & Holm & Max-$t$ \\\midrule"]
        last = None
        for r in rows:
            e = r["dif"]; a = _sz(e, "b1_pool"); b = _sz(e, "b1_pop_total")
            if last is not None and r["N"] != last: out.append(r"\addlinespace")
            out.append((f"{r['N']:,}".replace(",", "{,}") if r["N"] != last else "") + f" & {r['L']} & {e['varpi_median']:.2f} & {_f(a[0])} & {_f(a[1])} & {_f(b[0])} & {_f(b[1])} & "
                       + " & ".join(_f(e["fwer"][m]) for m in METHODS) + r" \\")
            last = r["N"]
        out.append(r"\bottomrule\end{tabular}")
        open(T + "test_size_main.tex", "w").write("\n".join(out) + "\n")
        summ["ideal_max_0.05"] = _ideal_max(R, 0.05); summ["ideal_max_0.01"] = _ideal_max(R, 0.01); summ["ideal_max_0.10"] = _ideal_max(R, 0.10)
        # full size table: two-sided, all designs and alphas
        out = [r"\resizebox{\textwidth}{!}{\begin{tabular}{rrl " + "rr rr " * 3 + r"}\toprule",
               r" & & & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$\alpha={a}$}}" for a in ALPHAS) + r" \\",
               r" & & & " + " & ".join(r"\multicolumn{2}{c}{Pool, cond.} & \multicolumn{2}{c}{Pop., two-phase}" for a in ALPHAS) + r" \\",
               r"$N$ & $L$ & Design & " + " & ".join("mean & max & mean & max" for a in ALPHAS) + r" \\\midrule"]
        for r in rows:
            for nm in ("srs", "pat", "dif"):
                e = r[nm]; c = []
                for a in ALPHAS:
                    c += [_f(x) for x in (*_sz(e, "b1_pool", str(a)), *_sz(e, "b1_pop_total", str(a)))]
                out.append((f"{r['N']:,}".replace(",", "{,}") + f" & {r['L']}" if nm == "srs" else " & ") + f" & {DLAB[nm]} & " + " & ".join(c) + r" \\")
            out.append(r"\addlinespace")
        out[-1] = r"\bottomrule\end{tabular}}"
        open(T + "test_size_full.tex", "w").write("\n".join(out) + "\n")
        # population value tested with the pool-conditional variance (shows the pool term)
        out = [r"\begin{tabular}{rr r rr rr rr}\toprule", r" & & & \multicolumn{2}{c}{$\alpha=0.01$} & \multicolumn{2}{c}{$\alpha=0.05$} & \multicolumn{2}{c}{$\alpha=0.10$} \\",
               r"$N$ & $L$ & $\varpi$ & mean & max & mean & max & mean & max \\\midrule"]
        for r in rows:
            e = r["dif"]; out.append(f"{r['N']} & {r['L']} & {e['varpi_median']:.2f} & " + " & ".join(_f(x) for a in ALPHAS for x in _sz(e, "b1_pop_design", str(a))) + r" \\")
        out.append(r"\bottomrule\end{tabular}")
        open(T + "test_size_popdesign.tex", "w").write("\n".join(out) + "\n")
        # one-sided / TOST at the boundary (alpha = 0.05 and 0.10), PATTERN-DIF and others
        out = [r"\resizebox{\textwidth}{!}{\begin{tabular}{rrl " + "rr rr rr " * 2 + r"}\toprule",
               r" & & & " + " & ".join(rf"\multicolumn{{6}}{{c}}{{$\alpha={a}$}}" for a in (0.05, 0.10)) + r" \\",
               r" & & & " + " & ".join(r"\multicolumn{2}{c}{$H_1:\theta<\theta_0$} & \multicolumn{2}{c}{$H_1:\theta>\theta_0$} & \multicolumn{2}{c}{TOST}" for a in (0.05, 0.10)) + r" \\",
               r"$N$ & $L$ & Design & " + " & ".join("mean & max & mean & max & mean & max" for a in (0.05, 0.10)) + r" \\\midrule"]
        for r in rows:
            for nm in ("srs", "pat", "dif"):
                e = r[nm]; c = []
                for a in (0.05, 0.10):
                    for k in ("less", "greater", "tost"): c += [_f(x) for x in e["b1_pool"][str(a)][k][:2]]
                out.append((f"{r['N']:,}".replace(",", "{,}") + f" & {r['L']}" if nm == "srs" else " & ") + f" & {DLAB[nm]} & " + " & ".join(c) + r" \\")
            out.append(r"\addlinespace")
        out[-1] = r"\bottomrule\end{tabular}}"
        open(T + "test_onesided_full.tex", "w").write("\n".join(out) + "\n")
        # FWER, power, coverage of the whole family (synthetic), all designs
        out = [r"\resizebox{\textwidth}{!}{\begin{tabular}{rrl rrrr rrrr rrrr rrrr}\toprule",
               r" & & & \multicolumn{4}{c}{FWER (12 true nulls)} & \multicolumn{4}{c}{Power, 80 material false nulls} & \multicolumn{4}{c}{Pool family covered} & \multicolumn{4}{c}{Population family covered} \\",
               r"$N$ & $L$ & Design & " + " & ".join(" & ".join(["Un.", "Bonf.", "Holm", "Max-$t$"]) for _ in range(4)) + r" \\\midrule"]
        for r in rows:
            for nm in ("srs", "pat", "dif"):
                e = r[nm]
                out.append((f"{r['N']:,}".replace(",", "{,}") + f" & {r['L']}" if nm == "srs" else " & ") + f" & {DLAB[nm]} & " +
                           " & ".join(_f(e[k][m]) for k in ("fwer", "power_material", "fam_pool_cov", "fam_pop_cov") for m in METHODS) + r" \\")
            out.append(r"\addlinespace")
        out[-1] = r"\bottomrule\end{tabular}}"
        open(T + "test_fwer_full.tex", "w").write("\n".join(out) + "\n")
        summ["syn"] = {f"{r['N']}_{r['L']}": {nm: dict(fwer=r[nm]["fwer"], size_pool=_sz(r[nm], "b1_pool"), size_pop=_sz(r[nm], "b1_pop_total"),
                                                       size_popdesign=_sz(r[nm], "b1_pop_design"), power_material=r[nm]["power_material"],
                                                       fam_pool=r[nm]["fam_pool_cov"], fam_pop=r[nm]["fam_pop_cov"], varpi=r[nm]["varpi_median"], fail=r[nm]["fail"])
                                              for nm in ("srs", "pat", "dif")} for r in rows}

    # ---------------- real pools: simultaneous coverage
    if real:
        rows = real["rows"]; Rr = real["R"]
        out = [r"\resizebox{\textwidth}{!}{\begin{tabular}{lrr rrrr r}\toprule", r"Setting & $L$ & $K$ & Unadj. & Bonf./Holm & Max-$t$ & Max-$t$ (floor) & $c_{\max t}/c_{\rm Bonf}$ \\\midrule"]
        last = None
        for r in rows:
            e, ef = r["dif"], r["dif_fz"]
            if last is not None and r["dataset"] != last: out.append(r"\addlinespace")
            cb = norm.ppf(1 - 0.05 / (2 * r["K"]))
            out.append((NAME[r["dataset"]] if r["dataset"] != last else "") + f" & {r['L']} & {r['K']} & {_f(e['cover']['none'])} & {_f(e['cover']['bonferroni'])} & {_f(e['cover']['maxt'])} & {_f(ef['cover']['maxt'])} & {e['crit_maxt']/cb:.3f}" + r" \\")
            last = r["dataset"]
        out.append(r"\bottomrule\end{tabular}}")
        open(T + "test_real_cover_full.tex", "w").write("\n".join(out) + "\n")
        Ls = sorted({r["L"] for r in rows})
        out = [r"\begin{tabular}{rl rrrr rrrr}\toprule", r" & & \multicolumn{4}{c}{Mean over cells} & \multicolumn{4}{c}{Minimum over cells} \\",
               r"$L$ & Design & Un. & Bonf. & Holm & Max-$t$ & Un. & Bonf. & Holm & Max-$t$ \\\midrule"]
        summ["real"] = {}
        for L in Ls:
            sub = [r for r in rows if r["L"] == L]
            for nm, lab in (("srs", "SRS"), ("pat", "PATTERN"), ("dif", "PATTERN-DIF"), ("dif_fz", "PATTERN-DIF + floor")):
                mean = [np.mean([r[nm]["cover"][m] for r in sub]) for m in METHODS]; mn = [min(r[nm]["cover"][m] for r in sub) for m in METHODS]
                out.append((f"{L} ({len(sub)} cells)" if nm == "srs" else "") + f" & {lab} & " + " & ".join(_f(x) for x in mean) + " & " + " & ".join(_f(x, 2) for x in mn) + r" \\")
                summ["real"][f"{L}_{nm}"] = dict(mean=dict(zip(METHODS, mean)), min=dict(zip(METHODS, mn)), cells=len(sub))
            out.append(r"\addlinespace")
        out[-1] = r"\bottomrule\end{tabular}"
        open(T + "test_real_cover_summary.tex", "w").write("\n".join(out) + "\n")
        for nm in ("srs", "pat", "dif", "dif_fz"):
            summ["real"][f"all_{nm}"] = {m: dict(mean=float(np.mean([r[nm]["cover"][m] for r in rows])), min=float(min(r[nm]["cover"][m] for r in rows)),
                                              share_below_90=float(np.mean([r[nm]["cover"][m] < 0.90 for r in rows]))) for m in METHODS}
        summ["real"]["fail_max"] = float(max(r[nm]["fail"] for r in rows for nm in ("srs", "pat", "dif", "dif_fz")))
        summ["real"]["crit_ratio_median"] = float(np.median([r["dif"]["crit_maxt"] / norm.ppf(1 - 0.05 / (2 * r["K"])) for r in rows]))

    # ---------------- B3
    if b3:
        eff_main = {}
        try:
            for r in json.load(open("results/label_efficiency_dif.json")):
                eff_main[(r["dataset"], r["L"])] = r.get("eff_pdif_plain_vs_srs_all")
        except Exception: pass
        per = []; allr = []
        for rec in b3["rows"]:
            Lg = np.array(rec["Lg"]); n = len(rec["contrasts"])
            pr = np.array(rec["L80_pred_dense"][0]) / np.array(rec["L80_pred_dense"][2]); em = np.array(rec["L80_emp"][0]) / np.array(rec["L80_emp"][2])
            pr_pat = np.array(rec["L80_pred_dense"][0]) / np.array(rec["L80_pred_dense"][1]); em_pat = np.array(rec["L80_emp"][0]) / np.array(rec["L80_emp"][1])
            pgr = np.array(rec["L80_pred_grid"][0]) / np.array(rec["L80_pred_grid"][2])
            vs, vd = np.array(rec["var_oracle"][0]), np.array(rec["var_oracle"][2])
            vr = np.array([np.exp(np.interp(np.log(rec["L80_pred_dense"][2][k]), np.log(Lg), np.log(vs[:, k]))) / np.exp(np.interp(np.log(rec["L80_pred_dense"][2][k]), np.log(Lg), np.log(vd[:, k]))) for k in range(n)])   # V_SRS/V_DIF at the DIF 80%-power budget
            ok = np.isfinite(em) & np.isfinite(pr)
            ratio = em[ok] / pr[ok]
            d = dict(dataset=rec["dataset"], n=n, n_pair=int(ok.sum()), var_ratio=float(np.median(vr)), pred=float(np.median(pr)), pred_grid=float(np.median(pgr)),
                     emp=float(np.nanmedian(em)) if ok.any() else float("nan"), ratio_med=float(np.median(ratio)) if ok.any() else float("nan"),
                     within20=float(np.mean(np.abs(np.log(ratio)) <= np.log(1.2))) if ok.any() else float("nan"),
                     eff_main=eff_main.get((rec["dataset"], 500)), pred_pat=float(np.median(pr_pat)), emp_pat=float(np.nanmedian(em_pat)) if np.isfinite(em_pat).any() else float("nan"))
            per.append(d)
            for k in range(n):
                allr.append(dict(dataset=rec["dataset"], k=rec["contrasts"][k], vr=vr[k], pred=pr[k], emp=em[k], pred_pat=pr_pat[k], emp_pat=em_pat[k],
                                 L80_pred_srs=rec["L80_pred_dense"][0][k], L80_pred_dif=rec["L80_pred_dense"][2][k], L80_emp_srs=rec["L80_emp"][0][k], L80_emp_dif=rec["L80_emp"][2][k],
                                 L80_grid_pred_dif=rec["L80_pred_grid"][2][k], L80_grid_pred_srs=rec["L80_pred_grid"][0][k], L80_emp_pat=rec["L80_emp"][1][k], L80_pred_pat=rec["L80_pred_dense"][1][k]))
        A = [r for r in allr if np.isfinite(r["pred"]) and np.isfinite(r["emp"])]
        lp, le = np.log([r["pred"] for r in A]), np.log([r["emp"] for r in A])
        agg = dict(n_contrasts=len(allr), n_pairs=len(A), median_ratio_emp_over_pred=float(np.exp(np.median(le - lp))), gm_ratio=float(np.exp(np.mean(le - lp))),
                   iqr_ratio=[float(np.exp(np.quantile(le - lp, q))) for q in (0.25, 0.75)], within20=float(np.mean(np.abs(le - lp) <= np.log(1.2))),
                   within50=float(np.mean(np.abs(le - lp) <= np.log(1.5))), spearman=float(spearmanr(lp, le)[0]), censored_emp=int(len(allr) - len(A)))
        # absolute budgets: empirical L80 versus like-for-like predicted L80 on the same grid, by design
        for nm, kp, ke in (("srs", "L80_grid_pred_srs", "L80_emp_srs"), ("dif", "L80_grid_pred_dif", "L80_emp_dif")):
            v = np.array([r[ke] / r[kp] for r in allr if np.isfinite(r[ke]) and np.isfinite(r[kp])])
            agg[f"abs_budget_ratio_{nm}"] = dict(median=float(np.median(v)), q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)), n=int(len(v)))
        vv = [(r["emp"], r["vr"], r["pred"]) for r in A]
        agg["emp_over_varratio_median"] = float(np.exp(np.median([np.log(a / b) for a, b, c in vv]))); agg["pred_over_varratio_median"] = float(np.exp(np.median([np.log(c / b) for a, b, c in vv])))
        agg["emp_over_varratio_range"] = [float(min(a / b for a, b, c in vv)), float(max(a / b for a, b, c in vv))]
        agg["pred_over_emp_range"] = [float(min(r["emp"] / r["pred"] for r in A)), float(max(r["emp"] / r["pred"] for r in A))]
        pat = [(r["emp_pat"], r["pred_pat"]) for r in allr if np.isfinite(r["emp_pat"]) and np.isfinite(r["pred_pat"])]
        agg["pat_median_ratio"] = float(np.exp(np.median([np.log(a / b) for a, b in pat]))) if pat else None
        summ["b3"] = dict(per_setting=per, overall=agg, fail_max=max(r["fail_max"] for r in b3["rows"]))
        out = [r"\begin{tabular}{lrrrrr}\toprule", r"Setting & Contrasts & $V_{\rm SRS}/V_{\rm DIF}$ & Predicted $L_{\rm SRS}/L_{\rm DIF}$ & Observed & Obs./pred.\ \\\midrule"]
        for d in per:
            out.append(f"{NAME[d['dataset']]} & {d['n']} & {d['var_ratio']:.2f} & {d['pred']:.2f} & {_f(d['emp'], 2)} & {_f(d['ratio_med'], 2)}" + r" \\")
        out.append(r"\midrule" + f"\nAll & {agg['n_contrasts']} & & {np.exp(np.median([np.log(r['pred']) for r in A])):.2f} & {np.exp(np.median([np.log(r['emp']) for r in A])):.2f} & {agg['median_ratio_emp_over_pred']:.2f}" + r" \\")
        out.append(r"\bottomrule\end{tabular}")
        open(T + "test_power_main.tex", "w").write("\n".join(out) + "\n")
        out = [r"\resizebox{\textwidth}{!}{\begin{tabular}{lr rr rr r rr r}\toprule",
               r"Setting & Contr. & $V_{\rm SRS}/V_{\rm DIF}$ & MSE ratio, $L=500$ & \multicolumn{2}{c}{$L_{\rm SRS}/L_{\rm DIF}$} & Obs./pred. & \multicolumn{2}{c}{$L_{\rm SRS}/L_{\rm PATTERN}$} & Within \\",
               r" & & (oracle, at $L_{\rm DIF}$) & (main study) & pred. & obs. & median & pred. & obs. & $\pm20\%$ \\\midrule"]
        for d in per:
            out.append(f"{NAME[d['dataset']]} & {d['n_pair']}/{d['n']} & {d['var_ratio']:.2f} & {_f(d['eff_main'], 2)} & {d['pred']:.2f} & {_f(d['emp'], 2)} & {_f(d['ratio_med'], 2)} & {d['pred_pat']:.2f} & {_f(d['emp_pat'], 2)} & {_f(d['within20'], 2)}" + r" \\")
        out.append(r"\bottomrule\end{tabular}}")
        open(T + "test_power_full.tex", "w").write("\n".join(out) + "\n")
        # per-contrast L80 table (compact) for the synthetic pool and one real pool
    # ---------------- B4
    if b4:
        out = [r"\resizebox{\textwidth}{!}{\begin{tabular}{lrl rr rrr rr rrr}\toprule",
               r" & & & \multicolumn{5}{c}{E1: $|\theta_k|\ge0.02$, effect $|\theta_k|$} & \multicolumn{5}{c}{E2: all live contrasts, effect $0.05$} \\",
               r"Pool & $L$ & Allocation & min & mean & pred. min & $<80\%$ & $K_1$ & min & mean & pred. min & $<80\%$ & $K$ \\\midrule"]
        S4 = []
        for r in b4["rows"]:
            for nm, lab in (("default", "Default (plain)"), ("resid", "Default (resid)"), ("maximin_E1", "Maximin, E1"), ("maximin_E2", "Maximin, E2")):
                e = r["alloc"][nm]; a, b = e["E1"], e["E2"]
                out.append((NAME[r["dataset"]] + f" & {r['L']}" if nm == "default" else " & ") + f" & {lab} & {_f(a['min'], 2)} & {_f(a['mean'], 2)} & {_f(a['pred_min'], 2)} & {a['n_below80']} & {r['K1'] if nm == 'default' else ''} & "
                           f"{_f(b['min'], 2)} & {_f(b['mean'], 2)} & {_f(b['pred_min'], 2)} & {b['n_below80']} & {r['K'] if nm == 'default' else ''}" + r" \\")
            out.append(r"\addlinespace")
            d, m1, m2 = r["alloc"]["default"], r["alloc"]["maximin_E1"], r["alloc"]["maximin_E2"]
            S4.append(dict(dataset=r["dataset"], L=r["L"], K1=r["K1"], K=r["K"], gap_E1=r["gap_E1"], gap_E2=r["gap_E2"],
                           E1=dict(min_default=d["E1"]["min"], min_opt=m1["E1"]["min"], mean_default=d["E1"]["mean"], mean_opt=m1["E1"]["mean"], pred_min_default=d["E1"]["pred_min"],
                                   pred_min_opt=m1["E1"]["pred_min"], crit_ratio=d["crit"]["E1"] / m1["crit"]["E1"], below_default=d["E1"]["n_below80"], below_opt=m1["E1"]["n_below80"]),
                           E2=dict(min_default=d["E2"]["min"], min_opt=m2["E2"]["min"], mean_default=d["E2"]["mean"], mean_opt=m2["E2"]["mean"], pred_min_default=d["E2"]["pred_min"],
                                   pred_min_opt=m2["E2"]["pred_min"], crit_ratio=d["crit"]["E2"] / m2["crit"]["E2"], below_default=d["E2"]["n_below80"], below_opt=m2["E2"]["n_below80"])))
        out[-1] = r"\bottomrule\end{tabular}}"
        open(T + "test_opt_full.tex", "w").write("\n".join(out) + "\n")
        summ["b4"] = S4
        out = [r"\begin{tabular}{lr rrr rrr}\toprule", r" & & \multicolumn{3}{c}{E1 (effect $|\theta_k|$)} & \multicolumn{3}{c}{E2 (common effect $0.05$)} \\",
               r"Pool & $L$ & default & maximin & $\times V$ & default & maximin & $\times V$ \\\midrule"]
        for s in S4:
            out.append(f"{NAME[s['dataset']]} & {s['L']} & {_f(s['E1']['min_default'], 2)} & {_f(s['E1']['min_opt'], 2)} & {s['E1']['crit_ratio']:.2f} & "
                       f"{_f(s['E2']['min_default'], 2)} & {_f(s['E2']['min_opt'], 2)} & {s['E2']['crit_ratio']:.2f}" + r" \\")
        out.append(r"\bottomrule\end{tabular}")
        open(T + "test_opt_main.tex", "w").write("\n".join(out) + "\n")

    # ---------------- figure
    if syn and b3 and b4:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        cN = {2000: "#d1495b", 8000: "#1b4f9c"}
        ax = axes[0, 0]; rows = syn["rows"]
        for r in rows:
            e = r["dif"]; x = e["varpi_median"]
            ax.scatter(x, _sz(e, "b1_pool")[0], marker="o", s=42, color="#1b4f9c", label="pool-conditional test of pool value" if r is rows[0] else None)
            ax.scatter(x, _sz(e, "b1_pop_design")[0], marker="s", s=42, color="#d1495b", label="pool-conditional test of population value" if r is rows[0] else None)
            ax.scatter(x, _sz(e, "b1_pop_total")[0], marker="^", s=48, facecolor="none", edgecolor="#00798c", lw=1.4, label="two-phase test of population value" if r is rows[0] else None)
        ax.axhline(0.05, color="grey", ls="--", lw=0.8); ax.set_xscale("log"); ax.set_xlabel(r"median $\hat V_{\rm pool}/\hat V_{\rm design}$"); ax.set_ylabel("rejection rate at nominal 5% (mean over 108 contrasts)")
        ax.set_ylim(0, 0.35); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left"); ax.set_title("(a) Size of the two-sided Wald test, PATTERN-DIF", fontsize=9)
        ax = axes[0, 1]; w = 0.2; cols = {"none": "#bbbbbb", "bonferroni": "#edae49", "holm": "#d1495b", "maxt": "#1b4f9c"}
        for j, m in enumerate(("none", "bonferroni", "maxt")):
            ax.bar(np.arange(len(rows)) + (j - 1) * 0.26, [1 - r["dif"]["fam_pop_cov"][m] for r in rows], 0.26, color=cols[m], label=MLAB[m].replace("$", "") + (" (= Holm)" if m == "bonferroni" else ""))
        ax.axhline(0.05, color="black", ls="--", lw=0.8); ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"{r['N']}\n{r['L']}" for r in rows], fontsize=8)
        ax.set_xlabel("pool size $N$ (upper) and budget $L$ (lower)"); ax.set_ylabel("FWER, all 108 population nulls true"); ax.legend(fontsize=7, ncol=3, loc="upper center"); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0); ax.set_title("(b) Familywise error of the 108-contrast family, PATTERN-DIF", fontsize=9)
        ax = axes[1, 0]; keys = [r["dataset"] for r in b3["rows"]]; pal = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, rec in enumerate(b3["rows"]):
            pr = np.array(rec["L80_pred_dense"][0]) / np.array(rec["L80_pred_dense"][2]); em = np.array(rec["L80_emp"][0]) / np.array(rec["L80_emp"][2])
            ax.scatter(pr, em, s=22, color=pal[i % 20], label=NAME[rec["dataset"]], alpha=0.9)
        lim = [0.6, 40]; ax.plot(lim, lim, color="black", lw=0.8); ax.plot(lim, [1.2 * v for v in lim], color="grey", lw=0.6, ls=":"); ax.plot(lim, [v / 1.2 for v in lim], color="grey", lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(r"predicted $L_{\rm SRS}/L_{\rm DIF}$ (oracle variances)"); ax.set_ylabel(r"observed $L_{\rm SRS}/L_{\rm DIF}$ (80% power)")
        ax.legend(fontsize=6, ncol=2, loc="upper left"); ax.grid(alpha=0.3); ax.set_title("(c) Label-budget ratio for 80% power", fontsize=9)
        ax = axes[1, 1]
        for i, s in enumerate(summ["b4"]):
            ax.plot([i - 0.15, i - 0.15], [s["E2"]["min_default"], s["E2"]["min_opt"]], color="#999", lw=1)
            ax.scatter(i - 0.15, s["E2"]["min_default"], color="#1b4f9c", s=26, label="default allocation" if i == 0 else None)
            ax.scatter(i - 0.15, s["E2"]["min_opt"], color="#d1495b", s=26, marker="D", label="maximin allocation" if i == 0 else None)
            ax.plot([i + 0.15, i + 0.15], [s["E1"]["min_default"], s["E1"]["min_opt"]], color="#999", lw=1)
            ax.scatter(i + 0.15, s["E1"]["min_default"], facecolor="none", edgecolor="#1b4f9c", s=26); ax.scatter(i + 0.15, s["E1"]["min_opt"], facecolor="none", edgecolor="#d1495b", s=26, marker="D")
        ax.set_xticks(range(len(summ["b4"]))); ax.set_xticklabels([f"{NAME[s['dataset']].split(' ')[0][:5]}\n{s['L']}" for s in summ["b4"]], fontsize=7)
        ax.set_ylabel("minimum power over targeted contrasts"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
        ax.set_title("(d) Worst-case power: filled = E2 (common effect 0.05), open = E1 (effect $|\\theta_k|$)", fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"): fig.savefig(f"manuscript_v2/figures/testing_sim.{ext}", dpi=200)
    json.dump(summ, open("results/testing_sim_summary.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(summ, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:12000])


if __name__ == "__main__":
    if sys.argv[1] == "run":
        st = sys.argv[2]; R = int(sys.argv[3])
        if st == "syn": run_syn(R)
        elif st == "real": run_real(R, sys.argv[4].split(",") if len(sys.argv) > 4 else REAL_KEYS)
        elif st == "b3": run_b3(R)
        elif st == "b4": run_b4(R)
    else:
        report()
