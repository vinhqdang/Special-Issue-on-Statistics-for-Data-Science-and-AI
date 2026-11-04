"""Repeated train/pool splits for the main label-efficiency study (SRS, GROUP, PATTERN, PATTERN-DIF).

For the six largest settings and new split seeds (default 1, 2, 3) the nine models are refitted on a new
60/40 train/pool split (prepare(key, seed) of realdata_validation / realdata_multi; the random_state of the
individual learners is unchanged), the pool is cached in results/pool_cache_seed<k>/<key>.npz, and the
protocol of experiments/label_efficiency_dif.py is run (budgets 500 and 1000, R = 100, same estimators and
efficiency definition). Seed 0 comes from results/label_efficiency_dif.json (R = 200).
Usage: python experiments/label_efficiency_splits.py build SEEDS   |  run R SEEDS  |  report
"""
import sys, os, json, zlib, warnings
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, make_strata, proportional_allocation, estimate, draw, neyman_allocation
from tcs.labeling_baselines import VecEstimands
from tcs.labeling_dif import DifContext, estimate_dif_all
warnings.filterwarnings("ignore")

KEYS = ["adult-sex", "adult-race", "bank", "credit-card", "magic", "electricity"]
BUDGETS = (500, 1000)
METHODS = ["srs", "srs_dif", "group", "pattern", "pdif_plain"]
OUT = "results/label_efficiency_splits.json"
Z95 = 1.959964
S = {}


def cache_dir(seed): return f"results/pool_cache_seed{seed}"


def build(key, seed):
    f = f"{cache_dir(seed)}/{key}.npz"
    if os.path.exists(f): return
    from experiments import realdata_validation as rv, realdata_multi as rm
    os.makedirs(cache_dir(seed), exist_ok=True)
    d = rv.prepare(key, seed) if key in ("adult-sex", "adult-race", "bank") else rm.prepare(key, seed)
    np.savez_compressed(f, **{k: np.asarray(v) for k, v in d.items()})
    print("built", key, seed, len(d["y"]), flush=True)


def setup(key, seed):
    z = np.load(f"{cache_dir(seed)}/{key}.npz", allow_pickle=True)
    pred, g, y, p = z["pred"], z["g"].astype(int), z["y"].astype(float), z["p"].mean(1)
    U, V, names = total_columns(pred, g); N, M = pred.shape
    est = Estimands(names, M, N)
    truth = est.value((y[:, None] * U + (1 - y[:, None]) * V).sum(0))
    live = []
    for (m, j) in est.pairs:
        dif = pred[:, m] != pred[:, j]
        live += [bool((dif & (y == 1)).any()), bool((dif & (y == 0)).any()), bool(dif.any())]
    kinds = np.array([k for (k, _, _) in est.labels])
    return dict(pred=pred, g=g, y=y, p=p, U=U, V=V, est=est, truth=truth, N=N, M=M, live=np.array(live), kinds=kinds,
                pos_min=min(int(((g == gg) & (y == yy)).sum()) for gg in (0, 1) for yy in (0, 1)))


def designs(d, L):
    U, V, p, est = d["U"], d["V"], d["p"], d["est"]; out = {}
    one = np.zeros(d["N"], int); out["srs"] = (one,) + proportional_allocation(one, L)
    sg = make_strata(d["pred"], d["g"], "group"); out["group"] = (sg,) + neyman_allocation(sg, U, V, p, est, L)
    sp = make_strata(d["pred"], d["g"], "pattern"); out["pattern"] = (sp,) + neyman_allocation(sp, U, V, p, est, L)
    ctx = {k: DifContext(out[k][0], out[k][1], U, V, p, est) for k in ("srs", "pattern")}
    return out, ctx


def run(args):
    k, L, seed = args
    d = S[k]; D, ctx = S[(k, L)]; rng = np.random.default_rng(seed); U, V, y, p, est = d["U"], d["V"], d["y"], d["p"], d["est"]; res = {}
    for name, plug, difname in (("srs", "srs", "srs_dif"), ("group", "group", None), ("pattern", "pattern", "pdif_plain")):
        st, ids, Nh, nh = D[name]; idx = draw(st, ids, nh, rng)
        res[plug] = estimate(st, ids, Nh, nh, U, V, y[idx], idx, est)
        if difname:
            th, ses = estimate_dif_all(st, ids, Nh, nh, U, V, y[idx], idx, est, p, ctx[name])
            res[difname] = (th, ses["none"], th - Z95 * ses["none"], th + Z95 * ses["none"])
    return {m: tuple(np.asarray(x, float) for x in res[m]) for m in METHODS}


def summarise(d, key, L, seed, R, res):
    truth, live, kinds = d["truth"], d["live"], d["kinds"]
    th = {m: np.array([x[m][0] for x in res]) for m in METHODS}; se = {m: np.array([x[m][1] for x in res]) for m in METHODS}
    lo = {m: np.array([x[m][2] for x in res]) for m in METHODS}; hi = {m: np.array([x[m][3] for x in res]) for m in METHODS}
    fin = {m: np.isfinite(th[m]).all(1) & np.isfinite(se[m]).all(1) for m in METHODS}
    row = dict(dataset=key, split=seed, N=d["N"], L=L, R=R, n_live=int(live.sum()), pos_min=d["pos_min"],
               disagree=float(np.mean([(d["pred"][:, m] != d["pred"][:, j]).mean() for m, j in d["est"].pairs])))
    tol = 1e-9 * (1 + np.abs(truth))
    for m in METHODS:
        ok = fin[m]; row[f"fail_{m}"] = float(1 - ok.mean())
        cov = ((lo[m][ok] - tol <= truth) & (truth <= hi[m][ok] + tol)).mean(0)[live]
        row[f"cover_{m}"] = float(cov.mean()); row[f"cover_{m}_min"] = float(cov.min())
        if m != "srs":
            both = fin["srs"] & ok
            mse = ((th[m][both] - truth) ** 2).mean(0); ref = ((th["srs"][both] - truth) ** 2).mean(0)
            eff = ref / np.maximum(mse, 1e-30); row[f"eff_{m}"] = float(np.median(eff[live]))
            for kk in ("dTPR", "dFPR", "dERR"): row[f"eff_{m}_{kk}"] = float(np.median(eff[live & (kinds == kk)]))
    return row


def main_run(R, seeds):
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    for seed in seeds:
        for key in KEYS:
            rows = [r for r in rows if not (r["dataset"] == key and r["split"] == seed)]
            d = setup(key, seed); S[key] = d
            if d["pos_min"] < 30: print(key, seed, "skipped (group cell < 30)", flush=True); continue
            for L in BUDGETS:
                S[(key, L)] = designs(d, L)
                with Pool(3) as pool:
                    res = pool.map(run, [(key, L, zlib.crc32(key.encode()) % 9973 + 17 * L + 100003 * seed + r) for r in range(R)], chunksize=4)
                row = summarise(d, key, L, seed, R, res); rows.append(row)
                print(key, seed, L, {m: (round(row.get(f"eff_{m}", 0), 2), round(row[f"cover_{m}"], 3)) for m in METHODS if m != "srs"}, flush=True)
                json.dump(rows, open(OUT, "w"), indent=1)
            del S[key]


NAME = {"adult-sex": "Adult (sex)", "adult-race": "Adult (race)", "bank": "Bank marketing", "credit-card": "Credit card",
        "magic": "MAGIC", "electricity": "Electricity"}


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    new = json.load(open(OUT)); old = json.load(open("results/label_efficiency_dif.json"))
    rows = []
    for r in old:
        if r["dataset"] in KEYS and r["L"] in BUDGETS:
            rows.append(dict(dataset=r["dataset"], split=0, L=r["L"], R=r["R"], N=r["N"],
                             eff_pdif_plain=r["eff_pdif_plain_vs_srs_all"], eff_pattern=r["eff_pattern_vs_srs_all"],
                             eff_srs_dif=r["eff_srs_dif_vs_srs_all"], eff_group=r["eff_group_vs_srs_all"],
                             cover_pdif_plain=r["cover_pdif_plain"], cover_pdif_plain_min=r["cover_pdif_plain_min"],
                             cover_srs=r["cover_srs"], cover_srs_min=r["cover_srs_min"]))
    rows += new
    splits = sorted({r["split"] for r in rows}); Ls = BUDGETS
    E = {(k, L, s): [r for r in rows if r["dataset"] == k and r["L"] == L and r["split"] == s][0]["eff_pdif_plain"]
         for k in KEYS for L in Ls for s in splits if any(r["dataset"] == k and r["L"] == L and r["split"] == s for r in rows)}
    summ = {"splits": splits, "per_setting": {}, "median_over_settings": {}}
    for k in KEYS:
        per = {}
        for L in Ls:
            v = np.array([E[(k, L, s)] for s in splits if (k, L, s) in E])
            per[str(L)] = dict(by_split={str(s): E[(k, L, s)] for s in splits if (k, L, s) in E}, mean=float(v.mean()), sd=float(v.std(ddof=1)),
                               cv=float(v.std(ddof=1) / v.mean()), min=float(v.min()), max=float(v.max()),
                               seed0=float(E[(k, L, 0)]), new_median=float(np.median([E[(k, L, s)] for s in splits if s != 0 and (k, L, s) in E])))
        summ["per_setting"][k] = per
    for L in Ls:
        summ["median_over_settings"][str(L)] = {str(s): float(np.median([E[(k, L, s)] for k in KEYS if (k, L, s) in E])) for s in splits}
        summ["median_over_settings"][str(L)]["pooled_all_splits"] = float(np.median([E[(k, L, s)] for k in KEYS for s in splits if (k, L, s) in E]))
    summ["median_over_settings_and_budgets"] = {str(s): float(np.median([E[(k, L, s)] for k in KEYS for L in Ls if (k, L, s) in E])) for s in splits}
    summ["median_over_settings_and_budgets"]["pooled_all_splits"] = float(np.median(list(E.values())))
    cs = {}
    for m in ("pdif_plain", "srs"):
        for s in splits:
            sel = [r for r in rows if r["split"] == s]
            cs[f"{m}_split{s}"] = dict(mean=float(np.mean([r[f"cover_{m}"] for r in sel])), worst=float(min(r[f"cover_{m}_min"] for r in sel)))
        sel = rows; cs[f"{m}_all"] = dict(mean=float(np.mean([r[f"cover_{m}"] for r in sel])), worst=float(min(r[f"cover_{m}_min"] for r in sel)))
    summ["coverage"] = cs
    summ["cv_median"] = float(np.median([summ["per_setting"][k][str(L)]["cv"] for k in KEYS for L in Ls]))
    summ["ratio_max_over_min_median"] = float(np.median([summ["per_setting"][k][str(L)]["max"] / summ["per_setting"][k][str(L)]["min"] for k in KEYS for L in Ls]))
    json.dump(summ, open("results/label_efficiency_splits_summary.json", "w"), indent=1)
    # table
    f = lambda x: f"{x:.1f}"
    out = [r"\begin{tabular}{lr" + "r" * len(splits) + r"rrr}\toprule",
           r"Setting & $L$ & " + " & ".join(f"split {s}" for s in splits) + r" & Mean & SD & Range \\\midrule"]
    for k in KEYS:
        for L in Ls:
            per = summ["per_setting"][k][str(L)]
            out.append((NAME[k] if L == Ls[0] else "") + f" & {L} & " + " & ".join(f(E[(k, L, s)]) for s in splits)
                       + f" & {f(per['mean'])} & {f(per['sd'])} & {f(per['min'])}--{f(per['max'])}" + r" \\")
        out.append(r"\addlinespace")
    for L in Ls:
        out.append(f"Median over settings & {L} & " + " & ".join(f(summ["median_over_settings"][str(L)][str(s)]) for s in splits) + r" & \multicolumn{3}{c}{} \\")
    out.append(r"\bottomrule\end{tabular}")
    os.makedirs("manuscript_v2/tables", exist_ok=True); os.makedirs("manuscript_v2/figures", exist_ok=True)
    open("manuscript_v2/tables/splits.tex", "w").write("\n".join(out) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    cm = plt.get_cmap("tab10")
    for ax, L in zip(axes, Ls):
        for i, k in enumerate(KEYS):
            for s in splits:
                if (k, L, s) in E:
                    ax.scatter(i + (0 if s else -0.0) + 0.06 * (s - 1.5), E[(k, L, s)], s=34 if s else 60, marker="D" if s == 0 else "o",
                               color="black" if s == 0 else cm(s), zorder=3, label=(f"split {s}" + (" (original)" if s == 0 else "")) if i == 0 else None)
        for s in splits:
            ax.axhline(summ["median_over_settings"][str(L)][str(s)], color="black" if s == 0 else cm(s), lw=0.6, ls=":")
        ax.set_xticks(range(len(KEYS))); ax.set_xticklabels([NAME[k] for k in KEYS], rotation=40, ha="right", fontsize=8)
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.set_title(f"L = {L}", fontsize=10)
    axes[0].set_ylabel("MSE(SRS) / MSE(PATTERN-DIF)"); axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"manuscript_v2/figures/splits_efficiency.{ext}", dpi=200)
    print(json.dumps(summ, indent=1)[:4000])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        for s in [int(x) for x in sys.argv[2].split(",")]:
            for k in KEYS: build(k, s)
    elif cmd == "run": main_run(int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")])
    else: report()
