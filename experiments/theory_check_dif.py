"""Check the theory of the difference estimator (manuscript_v2, Sections theory/simulation, PATTERN-DIF).

Estimator. With the label-free proxy mean psi_i = pi_i U_i + (1 - pi_i) V_i (known for every unit of the pool),
    T_dif = sum_{i in U} psi_i + sum_h N_h mean_{s_h}(t_i(Y_i) - psi_i),    theta_dif = f(T_dif),
the stratified difference estimator with lambda = 1 (tcs.labeling_baselines.ppi_estimate, tune=False). Plain HT is psi = 0.
Residual influence values: e_i = z_i - ztilde_i, ztilde_i = grad f(T)' psi_i, so e_i = (Y_i - pi_i) d_i, d_i = grad f' (U_i - V_i).

Predictions of the efficiency MSE_plain / MSE_dif of the SAME PATTERN sample (same strata, same allocation):
  LIN   exact linearised ratio  sum_h c_h S2_{z,h} / sum_h c_h S2_{e,h},  c_h = N_h^2 (1/n_h - 1/N_h), true labels in the formula;
  ANT   label-free anticipated ratio under the working model Y_i ~ Bernoulli(pi_i):  E S2_z = S2_{ztilde} + sbar2,
        E S2_e = sbar2, sbar2_h = mean_h pi (1 - pi) d^2        (ratio = sum c (S2_zt + sbar2) / sum c sbar2);
  OPT   ratio V_dif / V_opt for the per-stratum optimal lambda_h = Cov_h(z, zt) / S2_{zt,h} (fraction by which tuning could help).

Parts
  0  exact enumeration on a tiny pool: unbiasedness and covariance formula for any proxy and lambda (machine precision);
  A  synthetic pools with Y ~ Bernoulli(pi) (calibrated proxy by construction), sharpness a varied; and one pool with
     proxies of varying calibration (kappa: pi_proxy = sigmoid(kappa * logit pi); kappa < 0 is anti-correlated);
  B  four cached real pools (Monte Carlo, R replications): unbiasedness of T_dif, variance formula, variance estimator,
     coverage, predicted vs observed efficiency; allocation computed for the plain or for the residual variances;
  C  predicted efficiency for all 35 (pool, budget) cells of results/label_baselines.json, compared with the observed
     pattern_dif vs pattern efficiency stored there (R = 200).
Usage: python experiments/theory_check_dif.py [R_syn=2000] [R_real=2000]; then python experiments/theory_check_dif.py floor [R=1000]
(Part D, run second: zero-variance strata and the proxy variance floor for the difference estimator.)
Writes results/theory_check_dif.json and manuscript_v2/figures/fig_sim_dif_*.{pdf,png}. At most two worker processes.
"""
import sys, json, os, itertools, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs.labeling import (total_columns, Estimands, make_strata, neyman_allocation, estimate, draw)
from tcs.labeling_baselines import VecEstimands, ppi_estimate, stratppi_allocation

NPROC = 2
Z95 = 1.959964
REAL = {"adult-sex": 500, "bank": 500, "eeg": 500, "spambase": 250}
ALL_KEYS = ["adult-sex", "adult-race", "bank", "credit-card", "law-school", "magic", "eeg", "phoneme", "spambase",
            "churn", "electricity"]
S = {}   # state shared with forked workers


# ------------------------------------------------------------------ population quantities
def pop_stats(U, V, y, pi, est, strata):
    """Per-stratum population quantities of z (plain influence), zt (proxy influence), e = z - zt, at the true totals,
    and the label-free anticipated quantities (gradient at the proxy totals)."""
    N = len(y)
    t = y[:, None] * U + (1 - y[:, None]) * V
    T = t.sum(0); Gm = est.gradient(T)
    psi = pi[:, None] * U + (1 - pi[:, None]) * V
    Z = t @ Gm.T; Zt = psi @ Gm.T; E = Z - Zt
    Tp = psi.sum(0); Gp = est.gradient(Tp); A = U @ Gp.T; B = V @ Gp.T
    r2 = (pi * (1 - pi))[:, None] * (A - B) ** 2                              # anticipated residual variance of unit i
    Ez = pi[:, None] * A + (1 - pi[:, None]) * B
    ids = np.unique(strata); H = len(ids); nt = Z.shape[1]
    out = dict(ids=ids, Nh=np.zeros(H, int), S2z=np.zeros((H, nt)), S2zt=np.zeros((H, nt)), Cov=np.zeros((H, nt)),
               S2e=np.zeros((H, nt)), sbar2=np.zeros((H, nt)), S2ztA=np.zeros((H, nt)), S2zA=np.zeros((H, nt)))
    for k, h in enumerate(ids):
        s = strata == h; n = int(s.sum()); out["Nh"][k] = n
        if n < 2: continue
        out["S2z"][k] = Z[s].var(0, ddof=1); out["S2zt"][k] = Zt[s].var(0, ddof=1); out["S2e"][k] = E[s].var(0, ddof=1)
        out["Cov"][k] = ((Z[s] - Z[s].mean(0)) * (Zt[s] - Zt[s].mean(0))).sum(0) / (n - 1)
        out["sbar2"][k] = r2[s].mean(0); out["S2ztA"][k] = Ez[s].var(0, ddof=1)
    out["S2zA"] = out["S2ztA"] + out["sbar2"]                                  # E_m S2_z (plain), exact under the model
    out["T"] = T; out["truth"] = est.value(T); out["psi"] = psi; out["Fh"] = np.array([psi[strata == h].sum(0) for h in ids])
    return out


def var_of(Nh, nh, S2):
    """sum_h N_h^2 (1/n_h - 1/N_h) S2_h, per target (S2: (H, n_t))."""
    return (Nh[:, None] ** 2 * (1 / nh[:, None] - 1 / Nh[:, None]) * S2).sum(0)


def predictions(P, nh):
    Nh = P["Nh"]
    Vz, Ve = var_of(Nh, nh, P["S2z"]), var_of(Nh, nh, P["S2e"])
    Vzt = var_of(Nh, nh, P["S2zt"])
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(P["S2zt"] > 1e-300, P["Cov"] / np.maximum(P["S2zt"], 1e-300), 0.0)
        r2 = np.where((P["S2zt"] > 1e-300) & (P["S2z"] > 1e-300), P["Cov"] ** 2 / np.maximum(P["S2zt"] * P["S2z"], 1e-300), 0.0)
    Vopt = var_of(Nh, nh, P["S2z"] * (1 - r2))                                  # unclipped optimal lambda_h = beta_h
    lam = np.clip(beta, 0.0, 1.0)
    Vclip = var_of(Nh, nh, P["S2z"] - 2 * lam * P["Cov"] + lam ** 2 * P["S2zt"])    # lambda_h = clip(beta_h, 0, 1)
    c = (Nh[:, None] ** 2 * (1 / nh[:, None] - 1 / Nh[:, None]))
    beta_bar = (c * P["Cov"]).sum(0) / np.maximum((c * P["S2zt"]).sum(0), 1e-300)  # design-weighted slope of z on ztilde
    VzA, VeA = var_of(Nh, nh, P["S2zA"]), var_of(Nh, nh, P["sbar2"])
    return dict(Vz=Vz, Ve=Ve, Vopt=Vopt, VzA=VzA, VeA=VeA,
                lin=Vz / np.maximum(Ve, 1e-300), ant=VzA / np.maximum(VeA, 1e-300), opt=Ve / np.maximum(Vopt, 1e-300),
                optclip=Ve / np.maximum(Vclip, 1e-300), beta_bar=beta_bar)


# ------------------------------------------------------------------ Part 0: exact enumeration
def exact_check():
    rng = np.random.default_rng(7)
    Nh = np.array([6, 8]); nh = np.array([2, 3]); N = Nh.sum(); C = 3
    st = np.repeat([0, 1], Nh)
    u = rng.integers(0, 2, (N, C)).astype(float); v = rng.integers(0, 2, (N, C)).astype(float); y = rng.integers(0, 2, N).astype(float)
    t = y[:, None] * u + (1 - y[:, None]) * v; T = t.sum(0)
    rows = []
    for name, psi in (("random mixture of u and v", (lambda w: w[:, None] * u + (1 - w)[:, None] * v)(rng.random(N))),
                      ("arbitrary bounded proxy", rng.uniform(-1, 1, (N, C))),
                      ("anti-correlated proxy 1 - t", 1 - t)):
        for lam in (0.0, 0.5, 1.0):
            est_all, vhat_all = [], []
            idx0 = np.flatnonzero(st == 0); idx1 = np.flatnonzero(st == 1)
            for c0 in itertools.combinations(idx0, nh[0]):
                for c1 in itertools.combinations(idx1, nh[1]):
                    That = lam * psi.sum(0); vh = np.zeros(C)
                    for k, c in enumerate((c0, c1)):
                        r = t[list(c)] - lam * psi[list(c)]
                        That = That + Nh[k] * r.mean(0)
                        vh += Nh[k] ** 2 * (1 / nh[k] - 1 / Nh[k]) * r.var(0, ddof=1)
                    est_all.append(That); vhat_all.append(vh)
            est_all = np.array(est_all); vhat_all = np.array(vhat_all)
            Vform = np.zeros(C)
            for k in range(2):
                r = (t - lam * psi)[st == k]
                Vform += Nh[k] ** 2 * (1 / nh[k] - 1 / Nh[k]) * r.var(0, ddof=1)
            rows.append(dict(proxy=name, lam=lam, n_samples=len(est_all), max_abs_bias=float(np.abs(est_all.mean(0) - T).max()),
                             max_abs_var_err=float(np.abs(est_all.var(0) - Vform).max()),
                             max_abs_vhat_bias=float(np.abs(vhat_all.mean(0) - Vform).max()), var_scale=float(Vform.max())))
    return rows


# ------------------------------------------------------------------ Monte Carlo worker
def mc_rep(args):
    key, seed = args
    d = S[key]; rng = np.random.default_rng(seed); out = {}
    U, V, y, pi, ve, est = d["U"], d["V"], d["y"], d["pi"], d["ve"], d["est"]
    for arm, (st, ids, Nh, nh) in d["arms"].items():
        idx = draw(st, ids, nh, rng)
        thp, sep, _, _ = estimate(st, ids, Nh, nh, U, V, y[idx], idx, est)
        thd, sed, _, _ = ppi_estimate(st, ids, Nh, nh, idx, y, U, V, pi, ve, tune=False)
        res = dict(thp=thp, sep=sep, thd=thd, sed=sed)
        if d["totals"] and arm == "plain_alloc":
            hh = st[idx]; t = y[idx, None] * U[idx] + (1 - y[idx, None]) * V[idx]; f = d["psi"][idx]
            Tp = np.zeros(U.shape[1]); Td = d["Fh"].sum(0).copy()
            for k, h in enumerate(ids):
                s = hh == h
                Tp += Nh[k] * t[s].mean(0); Td += Nh[k] * (t[s].mean(0) - f[s].mean(0))
            res.update(Tp=Tp, Td=Td)
        out[arm] = res
    return out


def run_mc(key, R, base_seed):
    with Pool(NPROC) as pool:
        res = pool.map(mc_rep, [(key, base_seed + r) for r in range(R)], chunksize=20)
    return res


def mc_summary(d, P, res, arm, live):
    """Observed quantities for one arm. Returns dict of per-target arrays."""
    truth = P["truth"]
    stack = lambda k: np.array([r[arm][k] for r in res])
    thp, sep, thd, sed = stack("thp"), stack("sep"), stack("thd"), stack("sed")
    ok = np.isfinite(thp).all(1) & np.isfinite(thd).all(1) & np.isfinite(sep).all(1) & np.isfinite(sed).all(1)
    thp, sep, thd, sed = thp[ok], sep[ok], thd[ok], sed[ok]
    mse_p = ((thp - truth) ** 2).mean(0); mse_d = ((thd - truth) ** 2).mean(0)
    tol = 1e-9 * (1 + np.abs(truth))
    cov = lambda th, se: ((np.abs(th - truth) <= Z95 * se + tol)).mean(0)
    return dict(n_ok=int(ok.sum()), mse_p=mse_p, mse_d=mse_d, eff=mse_p / np.maximum(mse_d, 1e-30),
                var_d=thd.var(0, ddof=1), vhat_d=(sed ** 2).mean(0), var_p=thp.var(0, ddof=1), vhat_p=(sep ** 2).mean(0),
                cover_p=cov(thp, sep), cover_d=cov(thd, sed),
                bias_z_d=(thd.mean(0) - truth) / np.maximum(thd.std(0, ddof=1) / np.sqrt(ok.sum()), 1e-30))


def total_bias(res, P):
    out = {}
    for nm, k in (("plain", "Tp"), ("dif", "Td")):
        X = np.array([r["plain_alloc"][k] for r in res]); sd = X.std(0, ddof=1); m = sd > 1e-12
        z = (X.mean(0) - P["T"])[m] / (sd[m] / np.sqrt(len(X)))
        out[nm] = dict(n_coords=int(m.sum()), n_degenerate=int((~m).sum()), max_abs_z=float(np.abs(z).max()),
                       share_abs_z_gt_196=float((np.abs(z) > 1.96).mean()), mean_z2=float((z ** 2).mean()),
                       max_rel_bias=float((np.abs(X.mean(0) - P["T"])[m] / sd[m]).max()))
    return out


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def per_kind(x, kinds, mask):
    return {k: med(np.asarray(x)[mask & (kinds == k)]) for k in ("dTPR", "dFPR", "dERR")} | {"all": med(np.asarray(x)[mask])}


def analyse(key, d, P, nh_plain, nh_dif, res, live):
    """Compare MC with predictions for the plain-allocation arm (and the dif-allocation arm if present)."""
    kinds = d["kinds"]
    pr = predictions(P, nh_plain)
    ob = mc_summary(d, P, res, "plain_alloc", live)
    m = live & (pr["Ve"] > 1e-30) & (ob["mse_d"] > 1e-30)
    lr = np.log(ob["eff"][m] / pr["lin"][m])
    row = dict(key=key, R=len(res), n_targets=int(len(kinds)), n_compared=int(m.sum()),
               obs=per_kind(ob["eff"], kinds, m), lin=per_kind(pr["lin"], kinds, m), ant=per_kind(pr["ant"], kinds, m),
               opt=per_kind(pr["opt"], kinds, m), optclip=per_kind(pr["optclip"], kinds, m), beta_bar=per_kind(pr["beta_bar"], kinds, m),
               ratio_obs_lin=dict(median=float(np.exp(np.median(lr))), q10=float(np.exp(np.quantile(lr, .1))), q90=float(np.exp(np.quantile(lr, .9)))),
               corr_log_obs_lin=float(np.corrcoef(np.log(ob["eff"][m]), np.log(pr["lin"][m]))[0, 1]),
               corr_log_obs_ant=float(np.corrcoef(np.log(ob["eff"][m]), np.log(pr["ant"][m]))[0, 1]),
               ratio_obs_ant_median=med(ob["eff"][m] / pr["ant"][m]),
               # variance formula and variance estimator for the difference estimator
               varratio_mc_lin_dif=per_kind(ob["var_d"] / np.maximum(pr["Ve"], 1e-300), kinds, m),
               varratio_mc_lin_plain=per_kind(ob["var_p"] / np.maximum(pr["Vz"], 1e-300), kinds, m),
               vhat_over_var_dif=per_kind(ob["vhat_d"] / np.maximum(ob["var_d"], 1e-300), kinds, m),
               cover_plain=float(ob["cover_p"][live].mean()), cover_dif=float(ob["cover_d"][live].mean()),
               cover_dif_min=float(ob["cover_d"][live].min()), cover_plain_min=float(ob["cover_p"][live].min()),
               share_bias_gt2=float((np.abs(ob["bias_z_d"][live]) > 2).mean()),
               per_target=dict(obs=ob["eff"][m].tolist(), lin=pr["lin"][m].tolist(), ant=pr["ant"][m].tolist(),
                               opt=pr["opt"][m].tolist(), kind=kinds[m].tolist()))
    if "dif_alloc" in res[0]:
        ob2 = mc_summary(d, P, res, "dif_alloc", live)
        Ve2 = var_of(P["Nh"], nh_dif, P["S2e"])
        m2 = m & (ob2["mse_d"] > 1e-30)
        row["alloc"] = dict(
            nh_plain=nh_plain.tolist(), nh_dif=nh_dif.tolist(), Nh=P["Nh"].tolist(),
            gain_dif_alloc_obs=med(ob["mse_d"][m2] / ob2["mse_d"][m2]),                 # MSE_dif(plain alloc) / MSE_dif(dif alloc)
            gain_dif_alloc_lin=med((pr["Ve"] / np.maximum(Ve2, 1e-300))[m2]),
            eff_dif_vs_plain_dif_alloc_obs=med(ob2["mse_p"][m2] / ob2["mse_d"][m2]),   # on the dif-allocation sample
            cover_dif_dif_alloc=float(ob2["cover_d"][live].mean()), cover_dif_min_dif_alloc=float(ob2["cover_d"][live].min()))
    if d["totals"]:
        row["totals"] = total_bias(res, P)
    return row


# ------------------------------------------------------------------ Part A: synthetic pools
def make_pool(a, rng, N=60000, M=5, sig=0.8):
    """Y ~ Bernoulli(pi), pi = sigmoid(s), s = a * N(0,1) + group offset; model m predicts 1{s + sig * eps_m > 0}."""
    g = (rng.random(N) < 0.35).astype(int)
    s = a * rng.normal(size=N) + np.where(g == 1, -0.3, -0.8)
    pi = 1 / (1 + np.exp(-s)); y = (rng.random(N) < pi).astype(float)
    pred = (s[:, None] + sig * rng.normal(size=(N, M)) > 0).astype(int)
    return pred, g, y, s, pi


def setup_pool(pred, g, y, proxy, L, with_dif_alloc=False, totals=False):
    N, M = pred.shape
    U, V, names = total_columns(pred, g); est = Estimands(names, M, N); ve = VecEstimands(est)
    sp = make_strata(pred, g, "pattern")
    ids, Nh, nh = neyman_allocation(sp, U, V, proxy, est, L)
    P = pop_stats(U, V, y, proxy, est, sp)
    arms = {"plain_alloc": (sp, ids, Nh, nh)}
    nh2 = None
    if with_dif_alloc:
        _, _, nh2 = stratppi_allocation(sp, U, V, proxy, ve, L); arms["dif_alloc"] = (sp, ids, Nh, nh2)
    kinds = np.array([k for (k, _, _) in est.labels])
    d = dict(U=U, V=V, y=y, pi=proxy, ve=ve, est=est, arms=arms, psi=P["psi"], Fh=P["Fh"], totals=totals, kinds=kinds, pred=pred, g=g)
    return d, P, nh, nh2


def part_a(R, rng_master):
    out = dict(sharpness=[], miscal=[])
    L = 600
    for a in (0.5, 1.0, 2.0, 4.0):
        seed = int(rng_master.integers(1 << 30)); rng = np.random.default_rng(seed)
        pred, g, y, s, pi = make_pool(a, rng)
        key = f"syn_a{a}"; d, P, nh, _ = setup_pool(pred, g, y, pi, L, totals=True); S[key] = d
        live = np.ones(len(d["kinds"]), bool) & (predictions(P, nh)["Ve"] > 1e-30)
        res = run_mc(key, R, seed + 1)
        row = analyse(key, d, P, nh, None, res, live)
        row.update(a=a, kappa=1.0, N=len(y), L=L, seed=seed, share_any=float(np.mean((pred.sum(1) > 0) & (pred.sum(1) < pred.shape[1]))),
                   pos_rate=float(y.mean()), mean_pi_1mpi=float(np.mean(pi * (1 - pi))))
        row.pop("per_target"); out["sharpness"].append(row); del S[key]
        print(key, "obs", row["obs"], "lin", row["lin"], "ant", row["ant"], "cov", round(row["cover_plain"], 3), round(row["cover_dif"], 3), flush=True)
    a = 2.0
    seed = int(rng_master.integers(1 << 30)); rng = np.random.default_rng(seed)
    pred, g, y, s, pi = make_pool(a, rng)
    for kappa in (-1.0, 0.0, 0.5, 1.0, 2.0, 4.0):
        proxy = 1 / (1 + np.exp(-kappa * s))
        key = f"syn_k{kappa}"; d, P, nh, _ = setup_pool(pred, g, y, proxy, L, totals=(kappa in (-1.0, 1.0))); S[key] = d
        live = np.ones(len(d["kinds"]), bool) & (predictions(P, nh)["Ve"] > 1e-30)
        res = run_mc(key, R, seed + 7)
        row = analyse(key, d, P, nh, None, res, live)
        row.update(a=a, kappa=kappa, N=len(y), L=L, seed=seed)
        row.pop("per_target"); out["miscal"].append(row); del S[key]
        print(key, "obs", row["obs"], "lin", row["lin"], "ant", row["ant"], "opt", row["opt"], flush=True)
    return out


# ------------------------------------------------------------------ Part B/C: real pools
def load_real(key):
    z = np.load(f"results/pool_cache/{key}.npz", allow_pickle=True)
    return z["pred"], z["g"].astype(int), z["y"].astype(float), z["p"].mean(1)


def live_mask(pred, y, est):
    live = []
    for (m, j) in est.pairs:
        dif = pred[:, m] != pred[:, j]
        live += [bool((dif & (y == 1)).any()), bool((dif & (y == 0)).any()), bool(dif.any())]
    return np.array(live)


def part_b(R):
    out = []
    for key, L in REAL.items():
        pred, g, y, p = load_real(key)
        d, P, nh, nh2 = setup_pool(pred, g, y, p, L, with_dif_alloc=True, totals=True); S[key] = d
        live = live_mask(pred, y, d["est"])
        res = run_mc(key, R, zlib.crc32(key.encode()) % 9973 + 31 * L + 555555)
        row = analyse(key, d, P, nh, nh2, res, live)
        row.update(L=L, N=len(y), share_any=float(np.mean((pred.sum(1) > 0) & (pred.sum(1) < pred.shape[1]))))
        out.append(row); del S[key]
        a = row["alloc"]
        print(key, L, "obs", row["obs"]["all"], "lin", row["lin"]["all"], "ant", row["ant"]["all"], "opt", row["opt"]["all"],
              "| alloc gain obs/lin", a["gain_dif_alloc_obs"], a["gain_dif_alloc_lin"], "| cover", round(row["cover_plain"], 3),
              round(row["cover_dif"], 3), "| bias", row["totals"]["dif"]["max_abs_z"], flush=True)
    return out


def part_c(json_rows):
    jr = {(r["dataset"], r["L"]): r for r in json_rows}
    cells = []
    for key in ALL_KEYS:
        pred, g, y, p = load_real(key); N = len(y)
        U, V, names = total_columns(pred, g); est = Estimands(names, pred.shape[1], N)
        sp = make_strata(pred, g, "pattern"); P = pop_stats(U, V, y, p, est, sp)
        kinds = np.array([k for (k, _, _) in est.labels]); live = live_mask(pred, y, est)
        ve = VecEstimands(est)
        # weighted-variance allocation loss needs the plain weights w_t = 1 / S_t^2 (anticipated, as in the design)
        for L in (250, 500, 1000, 2000):
            if (key, L) not in jr: continue
            ids, Nh, nh = neyman_allocation(sp, U, V, p, est, L)
            _, _, nh2 = stratppi_allocation(sp, U, V, p, ve, L)
            pr = predictions(P, nh); pr2 = predictions(P, nh2)
            r = jr[(key, L)]
            obs = r["eff_pattern_dif_vs_pattern_all"]; obs_ppi = r["eff_pattern_ppi_vs_pattern_all"]
            m = live & (pr["Ve"] > 1e-30) & (pr["Vz"] > 1e-30)
            cells.append(dict(dataset=key, L=L, N=N, n_compared=int(m.sum()),
                              obs_dif_vs_plain=obs, obs_ppi_vs_plain=obs_ppi,
                              pred_lin=med(pr["lin"][m]) if m.any() else None, pred_ant=med(pr["ant"][m]) if m.any() else None,
                              pred_opt=med(pr["opt"][m]) if m.any() else None, pred_optclip=med(pr["optclip"][m]) if m.any() else None,
                              beta_bar=med(pr["beta_bar"][m]) if m.any() else None,
                              pred_dif_alloc_gain=med((pr["Ve"] / np.maximum(pr2["Ve"], 1e-300))[m]) if m.any() else None,
                              cover_plain=r["cover_pattern"], cover_dif=r["cover_pattern_dif"], cover_ppi=r["cover_pattern_ppi"],
                              kinds_lin={k: med(pr["lin"][m & (kinds == k)]) for k in ("dTPR", "dFPR", "dERR")},
                              kinds_obs={k: (r[f"eff_pattern_dif_vs_srs_{k}"] / r[f"eff_pattern_vs_srs_{k}"]
                                             if r[f"eff_pattern_vs_srs_{k}"] and r[f"eff_pattern_vs_srs_{k}"] < 1e6 else None)
                                         for k in ("dTPR", "dFPR", "dERR")}))
            print(key, L, "obs", None if obs is None else round(obs, 2), "lin", cells[-1]["pred_lin"] and round(cells[-1]["pred_lin"], 2),
                  "ant", cells[-1]["pred_ant"] and round(cells[-1]["pred_ant"], 2), flush=True)
    return cells


def cells_summary(cells):
    c = [x for x in cells if x["obs_dif_vs_plain"] and x["obs_dif_vs_plain"] > 0 and x["pred_lin"] and x["pred_lin"] < 1e6
         and x["obs_dif_vs_plain"] < 1e6]
    obs = np.array([x["obs_dif_vs_plain"] for x in c]); lin = np.array([x["pred_lin"] for x in c]); ant = np.array([x["pred_ant"] for x in c])
    opt = np.array([x["pred_opt"] for x in c]); optc = np.array([x["pred_optclip"] for x in c]); bb = np.array([x["beta_bar"] for x in c])
    q = lambda r: dict(median=float(np.median(r)), q10=float(np.quantile(r, .1)), q90=float(np.quantile(r, .9)),
                       min=float(r.min()), max=float(r.max()))
    ob_ppi = np.array([x["obs_ppi_vs_plain"] for x in c])
    return dict(n_cells=len(c), obs=q(obs), lin=q(lin), ant=q(ant),
                ratio_obs_lin=q(obs / lin), ratio_obs_ant=q(obs / ant),
                corr_log_obs_lin=float(np.corrcoef(np.log(obs), np.log(lin))[0, 1]),
                corr_log_obs_ant=float(np.corrcoef(np.log(obs), np.log(ant))[0, 1]),
                spearman_obs_lin=float(np.corrcoef(np.argsort(np.argsort(obs)), np.argsort(np.argsort(lin)))[0, 1]),
                share_within_20pct_lin=float(np.mean(np.abs(np.log(obs / lin)) < np.log(1.2))),
                ratio_ppi_dif_obs=q(ob_ppi / obs), pred_opt_over_dif=q(opt), pred_optclip_over_dif=q(optc), beta_bar=q(bb),
                lawschool_excluded=dict(ratio_obs_ant=q(np.array([o / a for o, a, x in zip(obs, ant, c) if x["dataset"] != "law-school"])),
                                        corr_log_obs_ant=float(np.corrcoef(*[np.log(np.array(v)) for v in zip(*[(o, a) for o, a, x in zip(obs, ant, c) if x["dataset"] != "law-school"])])[0, 1]),
                                        corr_log_obs_lin=float(np.corrcoef(*[np.log(np.array(v)) for v in zip(*[(o, l) for o, l, x in zip(obs, lin, c) if x["dataset"] != "law-school"])])[0, 1]),
                                        ratio_obs_lin=q(np.array([o / l for o, l, x in zip(obs, lin, c) if x["dataset"] != "law-school"]))),
                dif_alloc_gain_pred=q(np.array([x["pred_dif_alloc_gain"] for x in c])),
                cover_mean=dict(plain=float(np.mean([x["cover_plain"] for x in cells])), dif=float(np.mean([x["cover_dif"] for x in cells])),
                                ppi=float(np.mean([x["cover_ppi"] for x in cells]))),
                cover_frac_below_93=dict(plain=float(np.mean([x["cover_plain"] < .93 for x in cells])),
                                         dif=float(np.mean([x["cover_dif"] < .93 for x in cells])),
                                         ppi=float(np.mean([x["cover_ppi"] < .93 for x in cells]))))



# ------------------------------------------------------------------ Part D: zero-variance strata and the proxy variance floor
def dif_floor_estimate(strata, ids, Nh, nh, idx, y, U, V, pi, ve, psi_all, Fh, ud, pq):
    """Difference estimator (lambda = 1) with Wald standard errors: plain residual (se), and with the proxy variance floor
    (se_floor): a stratum whose sample variance of the residual influence values is exactly zero gets the anticipated
    value sigma_tilde_h^2 = mean_{U_h} pi (1 - pi) dhat_i^2 (dhat_i = grad f(T_dif)' (U_i - V_i)). ud = U - V, pq = pi (1 - pi)."""
    yk = y[idx]; t = yk[:, None] * U[idx] + (1 - yk[:, None]) * V[idx]; f = psi_all[idx]; h_of = strata[idx]
    H = len(ids); T = Fh.sum(0).copy(); sel = []
    for k in range(H):
        s = h_of == ids[k]; sel.append(s)
        T += Nh[k] * (t[s].mean(0) - f[s].mean(0))
    if not ve.den_ok(T): return None
    th = ve.value(T); G = ve.grad(T)
    e = (t - f) @ G.T
    dh = ud @ G.T; sig = pq[:, None] * dh ** 2
    var = np.zeros(ve.nt); varf = np.zeros(ve.nt); nzero = np.zeros(ve.nt)
    for k in range(H):
        s = sel[k]
        if nh[k] >= 2 and nh[k] < Nh[k]:
            c = Nh[k] ** 2 * (1 - nh[k] / Nh[k]) / nh[k]
            s2 = e[s].var(0, ddof=1); var += c * s2
            zero = s2 <= 1e-24
            sg = sig[strata == ids[k]].mean(0)
            nzero += zero & (sg > 0)                                           # floor actually adds variance (not a structural zero)
            varf += c * np.where(zero, sg, s2)
    return th, np.sqrt(var), np.sqrt(varf), nzero


def floor_rep(args):
    key, seed = args
    d = S[key]; rng = np.random.default_rng(seed)
    st, ids, Nh, nh = d["arms"]["plain_alloc"]; idx = draw(st, ids, nh, rng)
    r = dif_floor_estimate(st, ids, Nh, nh, idx, d["y"], d["U"], d["V"], d["pi"], d["ve"], d["psi"], d["Fh"], d["ud"], d["pq"])
    if r is None: return None
    ref = ppi_estimate(st, ids, Nh, nh, idx, d["y"], d["U"], d["V"], d["pi"], d["ve"], tune=False)
    return r + (ref[0], ref[1])


def part_d(R):
    out = []
    for key, L in (("churn", 250), ("law-school", 250), ("adult-sex", 250), ("credit-card", 250)):
        pred, g, y, p = load_real(key)
        d, P, nh, _ = setup_pool(pred, g, y, p, L); d["ud"] = d["U"] - d["V"]; d["pq"] = p * (1 - p); S[key] = d
        live = live_mask(pred, y, d["est"]); truth = P["truth"]
        with Pool(NPROC) as pool:
            res = pool.map(floor_rep, [(key, zlib.crc32(key.encode()) % 9973 + 91 * L + 777777 + r) for r in range(R)], chunksize=10)
        res = [r for r in res if r is not None and np.isfinite(r[0]).all()]
        th = np.array([r[0] for r in res]); se = np.array([r[1] for r in res]); sef = np.array([r[2] for r in res])
        nz = np.array([r[3] for r in res]); chk = max(float(np.nanmax(np.abs(np.array([r[1] for r in res]) - np.array([r[5] for r in res])))), 0.0)
        tol = 1e-9 * (1 + np.abs(truth))
        cov = lambda s_: (np.abs(th - truth) <= Z95 * s_ + tol).mean(0)
        c0, c1 = cov(se)[live], cov(sef)[live]
        wr = sef[:, live].mean(0) / np.maximum(se[:, live].mean(0), 1e-300)           # per-target ratio of mean standard errors
        row = dict(dataset=key, L=L, R=len(res), n_live=int(live.sum()), cover_nofloor=float(c0.mean()), cover_floor=float(c1.mean()),
                   cover_min_nofloor=float(c0.min()), cover_min_floor=float(c1.min()),
                   share_zero_se=float((se[:, live] <= 1e-12).mean()), share_floor_active=float((nz[:, live] > 0).mean()),            # (replication, target) pairs with a floored stratum
                   width_ratio_median=float(np.median(wr)), width_ratio_q90=float(np.quantile(wr, .9)),
                   max_abs_se_diff_vs_ppi_estimate=chk)
        out.append(row); del S[key]
        print("floor", key, L, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return out


# ------------------------------------------------------------------ figures
def figures(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    os.makedirs("manuscript_v2/figures", exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    col = {"dTPR": "#2a6fbb", "dFPR": "#d0782a", "dERR": "#3a9a5b"}
    mk = {"adult-sex": "o", "bank": "s", "eeg": "^", "spambase": "D"}

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for row in res["real"]:
        pt = row["per_target"]; kd = np.array(pt["kind"])
        for k in col:
            s = kd == k
            ax[0].scatter(np.array(pt["lin"])[s], np.array(pt["obs"])[s], s=7, alpha=.5, color=col[k], marker=mk[row["key"]], lw=0)
    lim = [0.8, 4]
    ax[0].plot(lim, lim, color="0.4", lw=.8, ls="--"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("predicted ratio LIN"); ax[0].set_ylabel("observed MSE(plain) / MSE(difference)")
    ax[0].set_title("four real pools, per target ($R=2000$)", fontsize=9)
    h = [Line2D([], [], color=col[k], marker="o", ls="", ms=4) for k in col] + [Line2D([], [], color="0.3", marker=mk[k], ls="", ms=4) for k in mk]
    ax[0].legend(h, [r"$\theta^{TPR}$", r"$\theta^{FPR}$", r"$\theta^{ERR}$"] + list(mk), frameon=False, fontsize=6.5, loc="upper left", ncol=2)
    cells = [c for c in res["cells"] if c["obs_dif_vs_plain"] and 0 < c["obs_dif_vs_plain"] < 1e6 and c["pred_lin"] and c["pred_lin"] < 1e6]
    ox = np.array([c["obs_dif_vs_plain"] for c in cells])
    ax[1].scatter([c["pred_lin"] for c in cells], ox, s=14, color="#1b4f9c", lw=0, label="LIN (true labels)")
    ax[1].scatter([c["pred_ant"] for c in cells], ox, s=14, facecolors="none", edgecolors="#c0392b", lw=.8, label="ANT (proxy only)")
    lim2 = [0.8, 20]
    ax[1].plot(lim2, lim2, color="0.4", lw=.8, ls="--"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlim(lim2); ax[1].set_ylim(lim2)
    ax[1].set_xlabel("predicted ratio"); ax[1].set_ylabel("observed, label_baselines.json ($R=200$)")
    ax[1].set_title("%d pool-budget cells (median over targets)" % len(cells), fontsize=9); ax[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"manuscript_v2/figures/fig_sim_dif_real.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for a_, rows, xk, xl in ((ax[0], res["synthetic"]["sharpness"], "a", r"sharpness $a$ of the latent score"),
                            (ax[1], res["synthetic"]["miscal"], "kappa", r"proxy dispersion $\kappa$ (1 = calibrated)")):
        x = np.array([r[xk] for r in rows])
        for k in col:
            a_.plot(x, [r["obs"][k] for r in rows], "o", color=col[k], ms=4)
            a_.plot(x, [r["lin"][k] for r in rows], "-", color=col[k], lw=1)
            a_.plot(x, [r["ant"][k] for r in rows], ":", color=col[k], lw=1.2)
        a_.axhline(1, color="0.5", lw=.8, ls="--"); a_.set_xlabel(xl); a_.set_ylabel("MSE(plain) / MSE(difference)")
    ax[0].set_xscale("log"); ax[0].set_xticks([0.5, 1, 2, 4]); ax[0].set_xticklabels(["0.5", "1", "2", "4"]); ax[0].minorticks_off()
    h = [Line2D([], [], color=col[k], lw=2) for k in col] + [Line2D([], [], color="k", marker="o", ls=""), Line2D([], [], color="k", lw=1),
         Line2D([], [], color="k", ls=":")]
    ax[0].legend(h, [r"$\theta^{TPR}$", r"$\theta^{FPR}$", r"$\theta^{ERR}$", "observed", "LIN", "ANT"], frameon=False, fontsize=7, loc="center", ncol=2, bbox_to_anchor=(0.5, 0.33))
    ax[0].set_ylim(0.9, 2.6)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"manuscript_v2/figures/fig_sim_dif_synth.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "figs":                                # redraw the figures from the stored json
        figures(json.load(open("results/theory_check_dif.json"))); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "floor":                              # python experiments/theory_check_dif.py floor [R=1000]
        res = json.load(open("results/theory_check_dif.json"))
        res["floor"] = part_d(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
        json.dump(res, open("results/theory_check_dif.json", "w"), indent=1); sys.exit(0)
    R_syn = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    R_real = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    rng_master = np.random.default_rng(20261105)
    res = dict(R_syn=R_syn, R_real=R_real)
    res["exact"] = exact_check()
    print("exact", max(r["max_abs_bias"] for r in res["exact"]), max(r["max_abs_var_err"] / r["var_scale"] for r in res["exact"]), flush=True)
    res["synthetic"] = part_a(R_syn, rng_master)
    res["real"] = part_b(R_real)
    res["cells"] = part_c(json.load(open("results/label_baselines.json")))
    try: res["floor"] = json.load(open("results/theory_check_dif.json")).get("floor")     # keep Part D (separate run)
    except Exception: pass
    res["cells_summary"] = cells_summary(res["cells"])
    print(json.dumps(res["cells_summary"], indent=1))
    json.dump(res, open("results/theory_check_dif.json", "w"), indent=1)
    figures(res)
