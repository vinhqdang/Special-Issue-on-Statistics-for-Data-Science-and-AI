"""Check the efficiency theory of the stratified labelling design (manuscript_v2, Sections theory/simulation).

Three predictions of the efficiency MSE_SRS / MSE_PATTERN of each comparison estimand are computed from
the full pool (labels used only to evaluate the formulas) and compared with Monte Carlo efficiencies.

  LIN   exact linearised design effect for the allocation actually used (Proposition on the
        linearised variance):  N^2 (1/L - 1/N) S_t^2  /  sum_h N_h^2 (1/n_h - 1/N_h) S_ht^2 .
  OWN   own-target Neyman design effect on the same strata (no fpc):  S_t^2 / (sum_h W_h S_ht)^2 .
  FORM  closed-form two-region formula of the Corollary (pair-specific agreement / disagreement
        regions within each group), an explicit function of the disagreement rates rho_g, the
        positive-and-disagree shares r_g, the group positive rates p_g, the group shares zeta_g and
        the group-wise target components delta_g; plus the small-target rule of thumb 1 / rho_mj.

Part A (real pools). The 11 cached pools and the budgets of results/label_efficiency.json. The Monte
Carlo part re-runs the SRS and PATTERN arms of experiments/label_efficiency.py with the same seeds
(the GROUP and ORACLE samples are drawn as well so that the random stream is identical), which gives
per-target efficiencies; their medians are checked against the medians stored in the json.

Part B (synthetic). Pools of N = 200000 with a controlled disagreement rate rho in
{1, 2, 5, 10, 20, 40}%; three families (two models with moderate / large group-wise gaps, five
models). For two-model pools the strata (group x number of models predicting 1) separate agreement
from disagreement exactly, so the closed form is tested directly (an own-target Neyman arm is run as
well). Part C: proxy misspecification (five models, rho = 5%): default, oracle, uninformative and
inverted proxies, and proportional allocation on the same strata.

Usage: python experiments/theory_check.py [R_real=200] [R_syn=1000]
Writes results/theory_check.json and manuscript_v2/figures/fig_sim_*.{pdf,png}. Two worker processes.
"""
import sys, json, zlib, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs.labeling import (total_columns, Estimands, make_strata, neyman_allocation,
                          proportional_allocation, _round_alloc, estimate, draw)

KEYS = ["adult-sex", "adult-race", "bank", "credit-card", "law-school", "magic", "eeg", "phoneme",
        "spambase", "churn", "electricity"]
NPROC = 2
G = {}   # global state shared with forked workers


# ------------------------------------------------------------------ population quantities
def influence(pred, g, y, est, U, V):
    T = (y[:, None] * U + (1 - y[:, None]) * V).sum(0)
    Z = (y[:, None] * U + (1 - y[:, None]) * V) @ est.gradient(T).T            # (N, n_targets)
    return T, Z


def strat_var(Z, strata):
    ids = np.unique(strata)
    Nh = np.array([(strata == h).sum() for h in ids])
    Sh2 = np.array([Z[strata == h].var(0, ddof=1) if n > 1 else np.zeros(Z.shape[1]) for h, n in zip(ids, Nh)])
    return ids, Nh, Sh2                                                          # Sh2: (H, n_targets)


def eff_lin(Z, strata, nh, L):
    """Exact linearised efficiency of STSI with allocation nh vs SRS of size L (both with fpc)."""
    N = len(Z); ids, Nh, Sh2 = strat_var(Z, strata)
    num = N ** 2 * (1 / L - 1 / N) * Z.var(0, ddof=1)
    den = (Nh[:, None] ** 2 * (1 / nh[:, None] - 1 / Nh[:, None]) * Sh2).sum(0)
    return num / np.maximum(den, 1e-30), num, den


def eff_own(Z, strata):
    """Own-target Neyman efficiency on the given strata (no fpc): S_t^2 / (sum W_h S_ht)^2."""
    N = len(Z); ids, Nh, Sh2 = strat_var(Z, strata)
    S2 = Z.var(0, ddof=1); den = ((Nh[:, None] / N) * np.sqrt(Sh2)).sum(0) ** 2
    return S2 / np.maximum(den, 1e-30)


def eff_formula(pred, g, y, est):
    """Closed-form two-region formula (Corollary) per target, and rho_mj per target."""
    N = len(y); out, rho_t = [], []
    for (m, j) in est.pairs:
        D = pred[:, m] - pred[:, j]; ad = np.abs(D)
        for kind in ("dTPR", "dFPR"):
            yy = y if kind == "dTPR" else 1 - y
            num = den = 0.0
            for gg in (0, 1):
                s = g == gg; zeta = s.mean(); p = yy[s].mean()
                rho = ad[s].mean(); r = (yy[s] * ad[s]).mean()
                delta = (yy[s] * D[s]).sum() / yy[s].sum()
                Sg2 = r - delta ** 2 * p                                        # group variance of Y(D - delta)
                if rho > 0:
                    piD = r / rho; cD = delta * p / rho
                    sD2 = max(piD - cD ** 2 + delta ** 2 * piD * (1 - piD) - 2 * delta * cD * (1 - piD), 0)
                else:
                    sD2 = 0.0
                piA = (p - r) / (1 - rho) if rho < 1 else 0.0
                sA = abs(delta) * np.sqrt(max(piA * (1 - piA), 0))
                num += Sg2 / (zeta * p ** 2)
                den += (rho * np.sqrt(sD2) + (1 - rho) * sA) / p
            out.append(num / max(den ** 2, 1e-30)); rho_t.append(ad.mean())
        e = np.abs(pred[:, m] - y) - np.abs(pred[:, j] - y)                    # = D (1 - 2Y), zero where m, j agree
        rho = ad.mean(); theta = e.mean(); den = 0.0
        for gg in (0, 1):
            s = g == gg; zeta = s.mean(); rho_g = ad[s].mean()
            if rho_g > 0:
                mu = e[s].sum() / ad[s].sum()                                   # mean of D(1-2Y) in the group's disagreement set
                den += zeta * rho_g * np.sqrt(max(1 - mu ** 2, 0))
        out.append((rho - theta ** 2) / max(den ** 2, 1e-30) if rho > 0 else 0.0); rho_t.append(rho)
    return np.array(out), np.array(rho_t)


# ------------------------------------------------------------------ Monte Carlo workers
def mc_real(args):
    key, L, seed = args
    d = G[key]; dz = G[(key, L)]; rng = np.random.default_rng(seed); res = {}
    for name in ("srs", "group", "pattern", "oracle"):                         # same order as label_efficiency.py
        st, ids, Nh, nh = dz[name]
        idx = draw(st, ids, nh, rng)
        if name in ("srs", "pattern"):
            res[name] = estimate(st, ids, Nh, nh, d["U"], d["V"], d["y"][idx], idx, d["est"])[0]
    return res


def mc_syn(args):
    key, seed = args
    d = G[key]; rng = np.random.default_rng(seed); res = {}
    for name, (st, ids, Nh, nh) in d["designs"].items():
        idx = draw(st, ids, nh, rng)
        res[name] = estimate(st, ids, Nh, nh, d["U"], d["V"], d["y"][idx], idx, d["est"])[0]
    return res


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None


def mse_eff(th_srs, th_m, truth):
    both = np.isfinite(th_srs).all(1) & np.isfinite(th_m).all(1)
    a = ((th_srs[both] - truth) ** 2).mean(0); b = ((th_m[both] - truth) ** 2).mean(0)
    return a / np.maximum(b, 1e-30), a, b


def summ(x):
    x = np.asarray(x, float)
    if len(x) == 0: return None
    return dict(median=float(np.median(x)), q10=float(np.quantile(x, .1)), q90=float(np.quantile(x, .9)))


# ------------------------------------------------------------------ Part A: real pools
def part_a(R):
    rows_json = json.load(open("results/label_efficiency.json"))
    out = []
    for key in KEYS:
        z = np.load(f"results/pool_cache/{key}.npz", allow_pickle=True)
        pred, g, y, p = z["pred"], z["g"].astype(int), z["y"].astype(float), z["p"].mean(1)
        U, V, names = total_columns(pred, g); N, M = pred.shape
        est = Estimands(names, M, N)
        T, Z = influence(pred, g, y, est, U, V); truth = est.value(T)
        G[key] = dict(U=U, V=V, y=y, est=est)
        sp = make_strata(pred, g, "pattern"); sg = make_strata(pred, g, "group"); one = np.zeros(N, int)
        f_form, rho_t = eff_formula(pred, g, y, est)
        f_own = eff_own(Z, sp)
        kinds = np.array([k for (k, _, _) in est.labels])
        nonzero = Z.var(0) > 1e-20                                                # targets with non-degenerate influence
        for L in sorted({r["L"] for r in rows_json if r["dataset"] == key}):
            dz = {"srs": (one,) + proportional_allocation(one, L),
                  "group": (sg,) + neyman_allocation(sg, U, V, p, est, L),
                  "pattern": (sp,) + neyman_allocation(sp, U, V, p, est, L),
                  "oracle": (sp,) + neyman_allocation(sp, U, V, y, est, L)}
            G[(key, L)] = dz
            nh = dz["pattern"][3]
            f_lin, num, den = eff_lin(Z, sp, nh, L)
            with Pool(NPROC) as pool:
                res = pool.map(mc_real, [(key, L, zlib.crc32(key.encode()) % 9973 + 17 * L + r) for r in range(R)])
            th_s = np.array([x["srs"] for x in res]); th_p = np.array([x["pattern"] for x in res])
            obs, mse_s, mse_p = mse_eff(th_s, th_p, truth)
            jr = [r for r in rows_json if r["dataset"] == key and r["L"] == L][0]
            census = bool(np.any(nh >= dz["pattern"][2]))
            ok = nonzero & (den > 1e-12) & (mse_p > 1e-30)                       # per-target comparisons
            lr = np.log(obs[ok] / f_lin[ok])
            row = dict(dataset=key, N=N, L=L, R=R, census_stratum=census,
                       n_targets=int(len(obs)), n_compared=int(ok.sum()),
                       rho_any=float(np.mean((pred.sum(1) > 0) & (pred.sum(1) < M))),
                       rho_pair_median=float(np.median(rho_t[kinds == "dERR"])),
                       json_eff_pattern_all=jr["eff_pattern_all"], repro_eff_pattern_all=float(np.median(obs)),
                       obs_all=float(np.median(obs)), lin_all=float(np.median(f_lin)), own_all=float(np.median(f_own)),
                       form_all=float(np.median(f_form)), inv_rho_all=float(np.median(1 / np.maximum(rho_t, 1e-12))),
                       obs_harm=float(ok.sum() / np.sum(1 / obs[ok])) if ok.any() else None,
                       lin_harm=float(ok.sum() / np.sum(1 / f_lin[ok])) if ok.any() else None,
                       ratio_obs_lin=summ(np.exp(lr)), mad_log_obs_lin=float(np.median(np.abs(lr))) if ok.any() else None,
                       ratio_obs_form=summ(obs[ok] / f_form[ok]), ratio_obs_own=summ(obs[ok] / f_own[ok]),
                       corr_log_obs_lin=corr(np.log(obs[ok]), np.log(f_lin[ok])),
                       corr_log_obs_form=corr(np.log(obs[ok]), np.log(f_form[ok])),
                       corr_log_obs_invrho=corr(np.log(obs[ok]), -np.log(rho_t[ok])))
            for k in ("dTPR", "dFPR", "dERR"):
                s = kinds == k
                row[f"obs_{k}"] = float(np.median(obs[s])); row[f"lin_{k}"] = float(np.median(f_lin[s]))
                row[f"form_{k}"] = float(np.median(f_form[s])); row[f"json_{k}"] = jr[f"eff_pattern_{k}"]
            row["per_target"] = dict(obs=obs[ok].tolist(), lin=f_lin[ok].tolist(), form=f_form[ok].tolist(),
                                     own=f_own[ok].tolist(), rho=rho_t[ok].tolist(), kind=kinds[ok].tolist())
            out.append(row)
            print(key, L, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                           if k in ("json_eff_pattern_all", "repro_eff_pattern_all", "lin_all", "own_all", "form_all",
                                    "inv_rho_all", "mad_log_obs_lin", "corr_log_obs_lin", "census_stratum")}, flush=True)
        for k in [k for k in G if (k == key or (isinstance(k, tuple) and k[0] == key))]:
            del G[k]
    return out


# ------------------------------------------------------------------ Part B/C: synthetic pools
def make_pool(rho, M, q, rng, N=200_000, zeta1=0.3, pg=(0.30, 0.20), acc_agree=0.85, beta=1.73):
    """q[m, g]: P(model m predicts the true label | disagreement region, group g)."""
    g = (rng.random(N) < zeta1).astype(int)
    y = (rng.random(N) < np.where(g == 1, pg[1], pg[0])).astype(float)
    dis = rng.random(N) < rho
    common = np.where(rng.random(N) < acc_agree, y, 1 - y)
    pred = np.repeat(common[:, None], M, 1)
    idx = np.flatnonzero(dis)
    if M == 2:
        right = rng.random(len(idx)) < q[0, g[idx]]
        a = np.where(right, y[idx], 1 - y[idx]); pred[idx, 0] = a; pred[idx, 1] = 1 - a
    else:
        todo = idx
        while len(todo):
            right = rng.random((len(todo), M)) < q[:, g[todo]].T
            pr = np.where(right, y[todo, None], 1 - y[todo, None]); pred[todo] = pr
            k = pr.sum(1); todo = todo[(k == 0) | (k == M)]
    pm = 1 / (1 + np.exp(-(beta * (2 * pred - 1) + rng.normal(0, 1, (N, M)))))
    return pred.astype(int), pm.mean(1), g, y


def syn_setup(key, pred, p, g, y, extra_own=False, proxies=None, L=1000):
    N, M = pred.shape
    U, V, names = total_columns(pred, g); est = Estimands(names, M, N)
    T, Z = influence(pred, g, y, est, U, V); truth = est.value(T)
    sp = make_strata(pred, g, "pattern"); one = np.zeros(N, int)
    designs = {"srs": (one,) + proportional_allocation(one, L),
               "pattern": (sp,) + neyman_allocation(sp, U, V, p, est, L)}
    if extra_own:                                                                 # own-target Neyman with true S_ht
        ids, Nh, Sh2 = strat_var(Z, sp)
        for t in range(Z.shape[1]):
            designs[f"own{t}"] = (sp, ids, Nh, _round_alloc(Nh * np.sqrt(Sh2[:, t]), Nh, L, 10))
    if proxies:
        for name, pp in proxies.items():
            designs[name] = (sp,) + neyman_allocation(sp, U, V, pp, est, L)
        designs["proportional"] = (sp,) + proportional_allocation(sp, L)
    G[key] = dict(U=U, V=V, y=y, est=est, designs=designs)
    return est, Z, truth, sp, designs


def part_b(R, rng_master):
    out = []
    fams = {"two_moderate": (2, np.array([[0.65, 0.35]])),
            "two_large": (2, np.array([[0.95, 0.05]])),
            "five": (5, np.array([[0.70, 0.50], [0.60, 0.60], [0.55, 0.45], [0.50, 0.65], [0.40, 0.55]]))}
    L = 1000
    for fam, (M, q) in fams.items():
        for rho in (0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
            seed = int(rng_master.integers(1 << 30)); rng = np.random.default_rng(seed)
            pred, p, g, y = make_pool(rho, M, q, rng)
            key = f"{fam}_{rho}"
            est, Z, truth, sp, designs = syn_setup(key, pred, p, g, y, extra_own=(M == 2), L=L)
            f_form, rho_t = eff_formula(pred, g, y.astype(float), est)
            f_lin, _, _ = eff_lin(Z, sp, designs["pattern"][3], L)
            f_own = eff_own(Z, sp)
            with Pool(NPROC) as pool:
                res = pool.map(mc_syn, [(key, seed + 1 + r) for r in range(R)])
            th = {k: np.array([x[k] for x in res]) for k in designs}
            obs, _, _ = mse_eff(th["srs"], th["pattern"], truth)
            kinds = [k for (k, _, _) in est.labels]
            row = dict(family=fam, M=M, rho_target=rho, rho_any=float(np.mean((pred.sum(1) > 0) & (pred.sum(1) < M))),
                       N=len(y), L=L, R=R, seed=seed, kinds=kinds, truth=truth.tolist(), rho_pair=rho_t.tolist(),
                       obs=obs.tolist(), lin=f_lin.tolist(), form=f_form.tolist(), own_formula=f_own.tolist(),
                       inv_rho=(1 / rho_t).tolist(),
                       census_stratum=bool(np.any(designs["pattern"][3] >= designs["pattern"][2])))
            if M == 2:
                obs_own = [mse_eff(th["srs"], th[f"own{t}"], truth)[0][t] for t in range(len(truth))]
                lin_own = [eff_lin(Z, sp, designs[f"own{t}"][3], L)[0][t] for t in range(len(truth))]
                row.update(obs_own=[float(v) for v in obs_own], lin_own=[float(v) for v in lin_own])
            out.append(row)
            print(fam, rho, "obs", np.round(obs, 2), "lin", np.round(f_lin, 2), "form", np.round(f_form, 2),
                  "own_obs", np.round(row.get("obs_own", []), 2), flush=True)
            del G[key]
    return out


def part_c(R, rng_master):
    seed = int(rng_master.integers(1 << 30)); rng = np.random.default_rng(seed)
    q = np.array([[0.70, 0.50], [0.60, 0.60], [0.55, 0.45], [0.50, 0.65], [0.40, 0.55]])
    pred, p, g, y = make_pool(0.05, 5, q, rng)
    proxies = {"oracle": y.astype(float), "uninformative": np.full(len(y), y.mean()), "inverted": 1 - p}
    est, Z, truth, sp, designs = syn_setup("proxy", pred, p, g, y, proxies=proxies)
    L = 1000; ids, Nh, Sh2 = strat_var(Z, sp)
    S2 = Z.var(0, ddof=1); w = 1 / S2
    B = np.sqrt((Sh2 * w).sum(1))                                                 # true A_h for weights w_t = 1/S_t^2

    def phi(nh):                                                                  # weighted linearised variance (with fpc)
        return float((Nh ** 2 * (1 / nh - 1 / Nh) * B ** 2).sum())
    with Pool(NPROC) as pool:
        res = pool.map(mc_syn, [("proxy", seed + 1 + r) for r in range(R)])
    th = {k: np.array([x[k] for x in res]) for k in designs}
    out = dict(N=len(y), L=L, R=R, seed=seed, arms={})
    phi_or = phi(designs["oracle"][3])
    mse_or = mse_eff(th["srs"], th["oracle"], truth)[2]
    for name in ("pattern", "oracle", "uninformative", "inverted", "proportional"):
        nh = designs[name][3]
        eff_o, _, mse_m = mse_eff(th["srs"], th[name], truth)
        a = np.sqrt(np.maximum(nh, 1e-12)) / Nh                                   # implied A_h (n_h ~ N_h a_h)
        out["arms"][name] = dict(
            loss_pred=phi(nh) / phi_or,                                           # weighted variance relative to oracle
            loss_obs=float((w * mse_m).sum() / (w * mse_or).sum()),
            eff_obs_median=float(np.median(eff_o)), eff_lin_median=float(np.median(eff_lin(Z, sp, nh, L)[0])),
            eff_obs_harm=float(len(eff_o) / np.sum(1 / eff_o)),
            eff_lin_harm=float(len(eff_o) / np.sum(1 / eff_lin(Z, sp, nh, L)[0])),
            nh=nh.tolist())
        print("proxy", name, {k: (round(v, 3) if isinstance(v, float) else None) for k, v in out["arms"][name].items()}, flush=True)
    # Kantorovich bound for the default proxy, ratio a_h / b_h over strata (no-fpc version of the loss)
    nh = designs["pattern"][3]; x = (nh / Nh) / B
    out["kantorovich_bound_default"] = float((x.max() + x.min()) ** 2 / (4 * x.max() * x.min()))
    out["Nh"] = Nh.tolist(); out["A_true"] = B.tolist()
    return out


# ------------------------------------------------------------------ figures
def figures(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("manuscript_v2/figures", exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    col = {"dTPR": "#2a6fbb", "dFPR": "#d0782a", "dERR": "#3a9a5b"}

    # Figure 1: real pools, observed vs linearised and vs closed form (per target)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for row in res["real"]:
        pt = row["per_target"]
        for k in col:
            s = np.array(pt["kind"]) == k
            ax[0].scatter(np.array(pt["lin"])[s], np.array(pt["obs"])[s], s=4, alpha=.35, color=col[k], lw=0)
            ax[1].scatter(np.array(pt["form"])[s], np.array(pt["obs"])[s], s=4, alpha=.35, color=col[k], lw=0)
    for a, lab in zip(ax, ("linearised design effect (actual allocation)", "closed-form two-region formula")):
        a.set_xscale("log"); a.set_yscale("log"); lim = [0.3, 300]
        a.plot(lim, lim, color="0.4", lw=.8, ls="--"); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel(f"predicted: {lab}"); a.set_ylabel("observed efficiency (Monte Carlo)")
    for k in col: ax[0].scatter([], [], color=col[k], s=10, label=k.replace("d", "$\\theta^{") + "}$")
    ax[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"manuscript_v2/figures/fig_sim_real.{ext}", dpi=200)
    plt.close(fig)

    # Figure 2: synthetic, efficiency against rho
    fams = [("two_moderate", "two models, moderate gaps"), ("two_large", "two models, large gaps"), ("five", "five models")]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for a, (fam, title) in zip(ax, fams):
        rows = [r for r in res["synthetic"] if r["family"] == fam]
        rh = np.array([r["rho_target"] for r in rows])
        for k in col:
            ix = [i for i, kk in enumerate(rows[0]["kinds"]) if kk == k]
            med = lambda f: np.array([np.median(np.array(r[f])[ix]) for r in rows])
            a.plot(rh, med("obs"), "o", color=col[k], ms=4)
            a.plot(rh, med("lin"), "-", color=col[k], lw=1)
            a.plot(rh, med("form"), ":", color=col[k], lw=1.2)
        a.plot(rh, 1 / rh, color="0.5", lw=.8, ls="--")
        a.set_xscale("log"); a.set_yscale("log"); a.set_title(title, fontsize=9)
        a.set_xlabel(r"disagreement rate $\rho$")
    ax[0].set_ylabel("efficiency vs SRS")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=col[k], lw=2) for k in col] + [Line2D([], [], color="k", marker="o", ls=""),
         Line2D([], [], color="k", lw=1), Line2D([], [], color="k", ls=":"), Line2D([], [], color="0.5", ls="--")]
    ax[2].legend(h, [r"$\theta^{TPR}$", r"$\theta^{FPR}$", r"$\theta^{ERR}$", "observed", "linearised", "closed form", r"$1/\rho$"],
                 frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"manuscript_v2/figures/fig_sim_rho.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    R_real = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    R_syn = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rng_master = np.random.default_rng(20261104)
    res = dict(R_real=R_real, R_syn=R_syn)
    res["synthetic"] = part_b(R_syn, rng_master)
    res["proxy"] = part_c(R_syn, rng_master)
    res["real"] = part_a(R_real)
    json.dump(res, open("results/theory_check.json", "w"), indent=1)
    figures(res)
