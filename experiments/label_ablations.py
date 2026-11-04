"""Ablation and robustness studies for the PATTERN labelling design (results/label_ablations.json).

For each of the 11 settings (cached pools) and budget L in {250, 500, 1000} with L <= N/4, R replications:
  A1 stratifier   : kcount (default, group x K) | binary (group x agree/disagree) |
                    joint3 (group x joint pattern of the 3 models agreeing most with the ensemble) |
                    kproxy2 / kproxy3 (group x K x within-cell proxy bins)
  A2 proxy        : ens (default) | single0 (LR) | single7 (Gaussian NB) | const 0.5 | shuffle (within group) |
                    noise (logit + N(0, 2^2)) | flip (1 - ens) | prop (proportional, no proxy) | oracle (true y)
  A3 models M     : random subsets of size 2, 3, 5 (3 subsets each, fixed seed) and all 9; SRS vs PATTERN
  A4 inference    : Wald | t with Satterthwaite df | stratified bootstrap percentile (default design and SRS);
                    design grid min_n in {5, 10, 20} x collapse threshold in {25, 50, 100}
  A5 EO gaps      : plug-in EO_m = max(|dTPR_m|, |dFPR_m|) under SRS, GROUP, PATTERN, PATTERN-GAP (allocation
                    targeting the per-model gaps) and ORACLE-ALLOC; RMSE, P(argmin correct), Kendall tau.
Usage: python experiments/label_ablations.py R [keys] [B]
"""
import sys, json, zlib, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import t as tdist
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, draw
from tcs.labeling_variants import (strata_kcount, strata_binary, strata_joint, strata_kproxy, top_agreement_models,
                                   proxies, allocate, estimate_full, bootstrap_percentile, eo_gap, GapEstimands,
                                   kendall_tau)

SETTINGS = ["adult-sex", "adult-race", "bank", "credit-card", "law-school", "magic", "eeg", "phoneme",
            "spambase", "churn", "electricity"]
BUDGETS = (250, 500, 1000)
SUBSET_SIZES = (2, 3, 5)
N_SUBSETS = 3
EO_DESIGNS = ["srs", "group", "pattern", "pattern_gap", "oracle"]
BOOT_DESIGNS = ["srs", "pattern"]
Z = 1.959964


def load(key):
    z = np.load(f"results/pool_cache/{key}.npz", allow_pickle=True)
    return z["pred"].astype(int), z["p"].astype(float), z["g"].astype(int), z["y"].astype(float)


def config(pred, p, g, y):
    U, V, names = total_columns(pred, g); N, M = pred.shape
    est = Estimands(names, M, N)
    T = (y[:, None] * U + (1 - y[:, None]) * V).sum(0)
    K = pred.sum(1)
    dis = [(pred[:, a] != pred[:, b]).mean() for a in range(M) for b in range(a + 1, M)]
    return dict(pred=pred, p=p, pi=p.mean(1), U=U, V=V, est=est, names=names, M=M, N=N,
                truth=est.value(T), Ttrue=T, nonunan=float(np.mean((K > 0) & (K < M))), pairdis=float(np.mean(dis)))


def build_designs(key, L):
    pred, p, g, y = load(key); N, M = pred.shape
    base = config(pred, p, g, y)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    prox = proxies(p, g, y, rng)
    U, V, est, pi = base["U"], base["V"], base["est"], base["pi"]
    gap_est = GapEstimands(base["names"], M)
    D = {}
    one = np.zeros(N, int)

    def add(name, strata, proxy=pi, min_n=10, kind="neyman", e=est, cfg="main"):
        ids, Nh, nh, mn = allocate(strata, U if cfg == "main" else SUB[cfg]["U"], V if cfg == "main" else SUB[cfg]["V"],
                                   proxy, e, L, min_n, kind)
        D[name] = dict(strata=strata, ids=ids, Nh=Nh, nh=nh, min_n=mn, H=len(ids), cfg=cfg, n=int(nh.sum()))

    SUB = {}
    add("srs", one, kind="prop")
    add("group", g.copy())
    sk = strata_kcount(pred, g)
    add("pattern", sk)
    # A1 stratifiers
    add("binary", strata_binary(pred, g))
    top3 = top_agreement_models(pred, p, 3)
    add("joint3", strata_joint(pred, g, top3))
    add("kproxy2", strata_kproxy(pred, g, pi, 2))
    add("kproxy3", strata_kproxy(pred, g, pi, 3))
    # A2 proxies
    for nm in ("single0", "single7", "const", "shuffle", "noise", "flip", "oracle"):
        add(nm if nm != "oracle" else "oracle", sk, proxy=prox[nm])
    add("prop", sk, kind="prop")
    # A4 design grid
    for mn in (5, 10, 20):
        for c in (25, 50, 100):
            if (mn, c) == (10, 50): continue
            add(f"grid_n{mn}_c{c}", strata_kcount(pred, g, c), min_n=mn)
    # A5 allocation targeting per-model gaps
    add("pattern_gap", sk, e=gap_est)
    # A3 subsets of models
    srng = np.random.default_rng(20261104)
    subsets = []
    for size in SUBSET_SIZES:
        seen = set()
        while len([s for s in subsets if len(s) == size]) < N_SUBSETS:
            s = tuple(sorted(srng.choice(M, size, replace=False).tolist()))
            if s not in seen: seen.add(s); subsets.append(s)
    for s in subsets:
        cname = "M" + "-".join(map(str, s))
        SUB[cname] = config(pred[:, list(s)], p[:, list(s)], g, y); SUB[cname]["models"] = list(s)
        add(f"{cname}|srs", one, kind="prop", e=SUB[cname]["est"], cfg=cname)
        add(f"{cname}|pattern", strata_kcount(pred[:, list(s)], g), proxy=SUB[cname]["pi"], e=SUB[cname]["est"], cfg=cname)
    return dict(base=base, sub=SUB, D=D, y=y, g=g, top3=top3, M=M, N=N)


def mse_ratio(a, b, truth):
    """Per-target MSE(a)/MSE(b) on samples where both are finite; NaN where both MSEs are 0."""
    both = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    ma = ((a[both] - truth) ** 2).mean(0); mb = ((b[both] - truth) ** 2).mean(0)
    r = ma / np.maximum(mb, 1e-30)
    return np.where((ma == 0) & (mb == 0), np.nan, r)


def coverage(lo, hi, truth, ok):
    c = ((lo[ok] <= truth) & (truth <= hi[ok])).mean(0)
    return float(c.mean()), float(c.min()), c


def run_cell(args):
    key, L, R, B = args
    t0 = time.time()
    S = build_designs(key, L)
    base, SUB, D, y = S["base"], S["sub"], S["D"], S["y"]
    names = list(D)
    th = {n: [] for n in names}; se = {n: [] for n in names}; df = {n: [] for n in names}
    eo = {n: [] for n in EO_DESIGNS}; blo = {n: [] for n in BOOT_DESIGNS}; bhi = {n: [] for n in BOOT_DESIGNS}
    crc = zlib.crc32(key.encode()) % 9973
    for r in range(R):
        for di, n in enumerate(names):
            d = D[n]; cfg = base if d["cfg"] == "main" else SUB[d["cfg"]]
            rng = np.random.default_rng([crc, L, r, di])
            idx = draw(d["strata"], d["ids"], d["nh"], rng)
            a, s, f, T = estimate_full(d["strata"], d["ids"], d["Nh"], d["nh"], cfg["U"], cfg["V"], y[idx], idx, cfg["est"])
            th[n].append(a); se[n].append(s); df[n].append(f)
            if n in EO_DESIGNS:
                eo[n].append(eo_gap(T, base["est"].ix, base["M"]))
            if n in BOOT_DESIGNS:
                lo_, hi_ = bootstrap_percentile(d["strata"], d["ids"], d["Nh"], cfg["U"], cfg["V"], y[idx], idx,
                                                cfg["est"], np.random.default_rng([crc, L, r, di, 1]), B=B)
                blo[n].append(lo_); bhi[n].append(hi_)
    out = dict(dataset=key, L=L, R=R, B=B, N=S["N"], top3=S["top3"], nonunan=base["nonunan"], pairdis=base["pairdis"],
               designs={}, subsets={}, eo={})
    kinds = np.array([k for (k, _, _) in base["est"].labels])
    for n in names:
        d = D[n]; cfg = base if d["cfg"] == "main" else SUB[d["cfg"]]
        A = np.array(th[n]); Sd = np.array(se[n]); F = np.array(df[n]); truth = cfg["truth"]
        ok = np.isfinite(A).all(1)
        q = np.where(np.isfinite(F), tdist.ppf(0.975, np.where(np.isfinite(F), F, 1e6)), Z)
        cz = coverage(A - Z * Sd, A + Z * Sd, truth, ok); ct = coverage(A - q * Sd, A + q * Sd, truth, ok)
        rec = dict(H=d["H"], min_n=d["min_n"], n=d["n"], fail=float(1 - ok.mean()),
                   cover_z=cz[0], cover_z_min=cz[1], cover_t=ct[0], cover_t_min=ct[1],
                   width_t_over_z=float(np.nanmean(q[ok]) / Z) if ok.any() else None,
                   degenerate=float(np.mean(Sd[ok][:, np.abs(truth) > 1e-12] == 0)),
                   worst_target=str(cfg["est"].labels[int(np.argmin(cz[2]))]),
                   worst_target_truth=float(truth[int(np.argmin(cz[2]))]))
        srs_name = "srs" if d["cfg"] == "main" else f"{d['cfg']}|srs"
        if n != srs_name:
            e = 1 / mse_ratio(A, np.array(th[srs_name]), truth)          # MSE_srs / MSE_design
            rec["eff_vs_srs"] = float(np.nanmedian(e))
            if d["cfg"] == "main":
                for k in ("dTPR", "dFPR", "dERR"):
                    rec[f"eff_vs_srs_{k}"] = float(np.nanmedian(e[kinds == k]))
                if n != "pattern":
                    rr = 1 / mse_ratio(A, np.array(th["pattern"]), truth)  # MSE_pattern / MSE_design
                    rec["eff_vs_pattern"] = float(np.nanmedian(rr))
                    rec["eff_vs_pattern_gmean"] = float(np.exp(np.nanmean(np.log(np.clip(rr, 1e-6, 1e6)))))
        if n in BOOT_DESIGNS:
            cb = coverage(np.array(blo[n]), np.array(bhi[n]), truth, ok)
            rec["cover_boot"] = cb[0]; rec["cover_boot_min"] = cb[1]
        if d["cfg"] == "main":
            out["designs"][n] = rec
        else:
            out["subsets"].setdefault(d["cfg"], dict(models=cfg["models"], M=len(cfg["models"]), nonunan=cfg["nonunan"],
                                                     pairdis=cfg["pairdis"]))[n.split("|")[1]] = rec
    # A5: EO gaps
    eo_true = eo_gap(base["Ttrue"], base["est"].ix, base["M"])
    best = int(np.argmin(eo_true)); srt = np.sort(eo_true)
    out["eo_truth"] = eo_true.tolist(); out["eo_margin"] = float(srt[1] - srt[0])
    for n in EO_DESIGNS:
        E = np.array(eo[n]); ok = np.isfinite(E).all(1); E = E[ok]
        out["eo"][n] = dict(fail=float(1 - ok.mean()),
                            rmse=float(np.sqrt(((E - eo_true) ** 2).mean(0)).mean()),
                            bias=float((E - eo_true).mean()),
                            p_best=float(np.mean(E.argmin(1) == best)),
                            p_best_top2=float(np.mean(np.isin(E.argmin(1), np.argsort(eo_true)[:2]))),
                            tau=float(np.mean([kendall_tau(e, eo_true) for e in E])))
    for n in EO_DESIGNS[1:]:
        Es = np.array(eo["srs"]); Ed = np.array(eo[n]); both = np.isfinite(Es).all(1) & np.isfinite(Ed).all(1)
        out["eo"][n]["eff_vs_srs"] = float(np.median(((Es[both] - eo_true) ** 2).mean(0) / np.maximum(((Ed[both] - eo_true) ** 2).mean(0), 1e-30)))
    # Monte Carlo reference for the minimum over targets of R-replicate coverage when every target covers 95%
    nt = len(base["truth"]); mc = np.random.default_rng(1).binomial(R, 0.95, size=(2000, nt)).min(1) / R
    out["mc_min_ref"] = dict(mean=float(mc.mean()), q05=float(np.quantile(mc, 0.05)))
    out["seconds"] = time.time() - t0
    print(key, L, f"{out['seconds']:.0f}s", "pattern cover", round(out["designs"]["pattern"]["cover_z_min"], 3), flush=True)
    return out


if __name__ == "__main__":
    R = int(sys.argv[1]); keys = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else SETTINGS
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 999
    tasks = []
    for key in keys:
        N = len(np.load(f"results/pool_cache/{key}.npz")["y"])
        tasks += [(key, L, R, B) for L in BUDGETS if L <= N / 4]
    tasks.sort(key=lambda a: -a[1])
    with Pool(2) as pool:
        rows = pool.map(run_cell, tasks, chunksize=1)
    order = {k: i for i, k in enumerate(SETTINGS)}
    rows.sort(key=lambda r: (order.get(r["dataset"], 99), r["L"]))
    out = sys.argv[4] if len(sys.argv) > 4 else "results/label_ablations.json"
    json.dump(rows, open(out, "w"), indent=1)
