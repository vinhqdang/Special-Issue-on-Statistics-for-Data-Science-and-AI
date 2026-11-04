"""Label efficiency with the difference estimator (PATTERN-DIF): design versus estimator.

Replicates experiments/label_efficiency.py (11 settings = preference_validation.SETTINGS minus nomao,
L in {250, 500, 1000, 2000} with L <= N/4, R replications, same seed scheme seed = crc32(key) % 9973 + 17 L + r,
same truth) and draws the designs in the order srs, group, pattern (so these three samples coincide with the
ones of label_efficiency.py), then pattern with residual allocation, then StratPPI. Estimators on the SAME indices:
  srs            SRS, plug-in (expansion) estimator
  srs_dif        SRS sample, difference estimator = PPI (lambda = 1)
  group          strata = group, 'plain' allocation, plug-in
  group_dif      same sample, difference estimator
  pattern        PATTERN, plug-in (existing)
  pdif_plain     same PATTERN sample, difference estimator  (PATTERN-DIF-plain)
  pdif_resid     PATTERN strata, residual allocation, difference estimator (PATTERN-DIF-resid)
  stratppi       StratPPI-style comparator of tcs.labeling_baselines (group x 10 proxy bins, tuned lambda)
Extra interval variants (same estimates, different SE): *_fz = zero-variance guard, *_fm = max(s2, proxy floor)
for pdif_plain and pdif_resid, and pattern_fz for the plug-in with the proxy variance floor of
tcs.labeling_variants.estimate_full(sigma).
Estimands whose numerator is identically zero in the pool (estimated exactly by every design) are excluded
from every summary (as in label_baselines.py); only law school is affected.
Usage: python experiments/label_efficiency_dif.py run R [keys]   |   report
"""
import sys, os, json, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, make_strata, proportional_allocation, estimate, draw, neyman_allocation
from tcs.labeling_baselines import VecEstimands, ppi_estimate, stratppi_allocation, score_strata
from tcs.labeling_variants import proxy_cov, estimate_full
from tcs.labeling_dif import DifContext, dif_allocation, estimate_dif_all
from experiments.preference_validation import SETTINGS

KEYS = [k for k in SETTINGS if k != "nomao"]
BUDGETS = (250, 500, 1000, 2000)
CORE = ["srs", "srs_dif", "group", "group_dif", "pattern", "pdif_plain", "pdif_resid", "stratppi"]
EXTRA = ["pdif_plain_fz", "pdif_plain_fm", "pdif_resid_fz", "pdif_resid_fm", "pattern_fz"]
METHODS = CORE + EXTRA
SE_VARIANT = {"pdif_plain_fz": ("pdif_plain", "zero"), "pdif_plain_fm": ("pdif_plain", "max"),
              "pdif_resid_fz": ("pdif_resid", "zero"), "pdif_resid_fm": ("pdif_resid", "max")}
Z95 = 1.959964
OUT = os.environ.get("LD_OUT", "results/label_efficiency_dif.json")
S = {}


def setup(key):
    z = np.load(f"results/pool_cache/{key}.npz", allow_pickle=True)
    pred, g, y, p = z["pred"], z["g"].astype(int), z["y"].astype(float), z["p"].mean(1)
    U, V, names = total_columns(pred, g); N, M = pred.shape
    est = Estimands(names, M, N); ve = VecEstimands(est)
    truth = est.value((y[:, None] * U + (1 - y[:, None]) * V).sum(0))
    err = (pred != y[:, None]).mean(0); top = np.argsort(err)[:3]           # pool labels: reporting only
    top_pairs = [k for k, (m, j) in enumerate(est.pairs) if m in top and j in top]
    live = []
    for (m, j) in est.pairs:
        dif = pred[:, m] != pred[:, j]
        live += [bool((dif & (y == 1)).any()), bool((dif & (y == 0)).any()), bool(dif.any())]
    return dict(pred=pred, g=g, y=y, p=p, U=U, V=V, est=est, ve=ve, truth=truth, N=N, M=M, live=np.array(live),
                top_targets=np.array([3 * k + r for k in top_pairs for r in range(3)]),
                kinds=np.array([k for (k, _, _) in est.labels]),
                Pg=[int(((g == gg) & (y == 1)).sum()) for gg in (0, 1)], Qg=[int(((g == gg) & (y == 0)).sum()) for gg in (0, 1)])


def designs(d, L):
    out = {}; U, V, p, est, ve = d["U"], d["V"], d["p"], d["est"], d["ve"]
    one = np.zeros(d["N"], int)
    out["srs"] = (one,) + proportional_allocation(one, L)
    sg = make_strata(d["pred"], d["g"], "group"); out["group"] = (sg,) + neyman_allocation(sg, U, V, p, est, L)
    sp = make_strata(d["pred"], d["g"], "pattern"); out["pattern"] = (sp,) + neyman_allocation(sp, U, V, p, est, L)
    out["resid"] = (sp,) + dif_allocation(sp, U, V, p, est, L, "resid")
    ss = score_strata(p, d["g"], 10); out["stratppi"] = (ss,) + stratppi_allocation(ss, U, V, p, ve, L)
    ctx = {k: DifContext(out[k][0], out[k][1], U, V, p, est) for k in ("srs", "group", "pattern", "resid")}
    sigma = proxy_cov(sp, out["pattern"][1], U, V, p)
    return out, ctx, sigma


def run(args):
    key, L, seed = args
    d = S[key]; D, ctx, sigma = S[(key, L)]; rng = np.random.default_rng(seed); res = {}
    U, V, y, p, est, ve = d["U"], d["V"], d["y"], d["p"], d["est"], d["ve"]

    def dif(name, st, ids, Nh, nh, idx, c):
        th, ses = estimate_dif_all(st, ids, Nh, nh, U, V, y[idx], idx, est, p, c)
        return th, ses

    for name, plug, difname in (("srs", "srs", "srs_dif"), ("group", "group", "group_dif"), ("pattern", "pattern", "pdif_plain")):
        st, ids, Nh, nh = D[name]; idx = draw(st, ids, nh, rng)
        res[plug] = estimate(st, ids, Nh, nh, U, V, y[idx], idx, est)
        th, ses = dif(name, st, ids, Nh, nh, idx, ctx[name])
        res[difname] = (th, ses["none"], th - Z95 * ses["none"], th + Z95 * ses["none"])
        if name == "pattern":
            for nm, (src, fl) in SE_VARIANT.items():
                if src == "pdif_plain":
                    res[nm] = (th, ses[fl], th - Z95 * ses[fl], th + Z95 * ses[fl])
            out = estimate_full(st, ids, Nh, nh, U, V, y[idx], idx, est, sigma)
            thp, sef = out[0], out[4]
            res["pattern_fz"] = (thp, sef, thp - Z95 * sef, thp + Z95 * sef)
    st, ids, Nh, nh = D["resid"]; idx = draw(st, ids, nh, rng)
    th, ses = dif("resid", st, ids, Nh, nh, idx, ctx["resid"])
    res["pdif_resid"] = (th, ses["none"], th - Z95 * ses["none"], th + Z95 * ses["none"])
    for nm, (src, fl) in SE_VARIANT.items():
        if src == "pdif_resid":
            res[nm] = (th, ses[fl], th - Z95 * ses[fl], th + Z95 * ses[fl])
    st, ids, Nh, nh = D["stratppi"]; idx = draw(st, ids, nh, rng)
    res["stratppi"] = ppi_estimate(st, ids, Nh, nh, idx, y, U, V, p, ve, tune=True)
    return {m: tuple(np.asarray(x, float) for x in res[m]) for m in METHODS}


def summarise(d, key, L, R, res):
    truth = d["truth"]; live = d["live"]; top = np.array([t for t in d["top_targets"] if live[t]], int)
    th = {m: np.array([x[m][0] for x in res]) for m in METHODS}
    se = {m: np.array([x[m][1] for x in res]) for m in METHODS}
    lo = {m: np.array([x[m][2] for x in res]) for m in METHODS}
    hi = {m: np.array([x[m][3] for x in res]) for m in METHODS}
    fin = {m: np.isfinite(th[m]).all(1) & np.isfinite(se[m]).all(1) for m in METHODS}
    row = dict(dataset=key, N=d["N"], L=L, R=R, n_targets=int(len(truth)), n_live=int(live.sum()), n_top_live=int(len(top)))
    kinds = d["kinds"]
    for m in METHODS:
        row[f"fail_{m}"] = float(1 - fin[m].mean()); ok = fin[m]
        if ok.sum() == 0: continue
        tol = 1e-9 * (1 + np.abs(truth))                                   # exact estimates with zero-width CIs count as covered
        cov = ((lo[m][ok] - tol <= truth) & (truth <= hi[m][ok] + tol)).mean(0)[live]
        row[f"cover_{m}"] = float(cov.mean()); row[f"cover_{m}_min"] = float(cov.min()); row[f"cover_{m}_p05"] = float(np.quantile(cov, 0.05))
        row[f"bias2share_{m}"] = float(np.median(((th[m][ok].mean(0) - truth) ** 2 / np.maximum(((th[m][ok] - truth) ** 2).mean(0), 1e-30))[live]))
        row[f"width_{m}"] = float(np.median((hi[m][ok] - lo[m][ok]).mean(0)[live]))
    for ref in ("srs", "pattern", "pdif_plain"):
        for m in CORE + ["pdif_plain_fz"]:
            if m == ref: continue
            both = fin[ref] & fin[m]
            if both.sum() < 10: row[f"eff_{m}_vs_{ref}_all"] = None; row[f"eff_{m}_vs_{ref}_top"] = None; continue
            eff = ((th[ref][both] - truth) ** 2).mean(0) / np.maximum(((th[m][both] - truth) ** 2).mean(0), 1e-30)
            row[f"eff_{m}_vs_{ref}_all"] = float(np.median(eff[live]))
            row[f"eff_{m}_vs_{ref}_top"] = float(np.median(eff[top])) if len(top) else None
            if ref == "srs":
                for k in ("dTPR", "dFPR", "dERR"):
                    row[f"eff_{m}_vs_srs_{k}"] = float(np.median(eff[live & (kinds == k)]))
    # who has the smallest MSE: per estimand over replications where all CORE designs are defined
    allok = np.all([fin[m] for m in CORE], axis=0)
    if allok.sum() >= 10:
        mse = np.array([((th[m][allok] - truth) ** 2).mean(0)[live] for m in CORE])       # (designs, live targets)
        best = mse.argmin(0)
        for k, m in enumerate(CORE):
            row[f"best_share_{m}"] = float((best == k).mean())
        row["best_cell"] = CORE[int(np.argmax([np.median(mse[0] / np.maximum(mse[k], 1e-30)) for k in range(len(CORE))]))]
    return row


def main_run(R, keys):
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    rows = [r for r in rows if r["dataset"] not in keys]
    for key in keys:
        d = setup(key)
        if min(d["Pg"] + d["Qg"]) < 30:
            print(key, "skipped: a group has fewer than 30 positives or negatives", flush=True); continue
        S[key] = d
        for L in BUDGETS:
            if L > d["N"] / 4: continue
            S[(key, L)] = designs(d, L)
            with Pool(3) as pool:
                res = pool.map(run, [(key, L, zlib.crc32(key.encode()) % 9973 + 17 * L + r) for r in range(R)], chunksize=4)
            row = summarise(d, key, L, R, res); rows.append(row)
            print(key, L, {m: (round(row.get(f"eff_{m}_vs_srs_all") or 0, 2), round(row.get(f"cover_{m}", 0), 3))
                           for m in ("srs_dif", "pattern", "pdif_plain", "pdif_resid", "stratppi")}, flush=True)
            json.dump(rows, open(OUT, "w"), indent=1)
        del S[key]


# ---------------------------------------------------------------- reporting
LABEL = {"srs": "SRS", "srs_dif": "SRS-DIF", "group": "GROUP", "group_dif": "GROUP-DIF", "pattern": "PATTERN",
         "pdif_plain": "PATTERN-DIF (plain)", "pdif_resid": "PATTERN-DIF (resid)", "stratppi": "StratPPI",
         "pdif_plain_fz": "DIF (plain) + floor", "pdif_plain_fm": "DIF (plain) + max floor",
         "pdif_resid_fz": "DIF (resid) + floor", "pdif_resid_fm": "DIF (resid) + max floor", "pattern_fz": "PATTERN + floor"}
NAME = {"adult-sex": "Adult (sex)", "adult-race": "Adult (race)", "bank": "Bank marketing", "credit-card": "Credit card",
        "law-school": "Law school", "magic": "MAGIC", "eeg": "EEG eye state", "phoneme": "Phoneme", "spambase": "Spambase",
        "churn": "Churn", "electricity": "Electricity"}


def fmt(x, big=100):
    if x is None or not np.isfinite(x): return "--"
    if x >= big: return f"$>${big}"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def _med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def _g(r, k):
    v = r.get(k); return np.nan if v is None else v


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = json.load(open(OUT)); rows.sort(key=lambda r: (KEYS.index(r["dataset"]), r["L"]))
    os.makedirs("manuscript_v2/figures", exist_ok=True); os.makedirs("manuscript_v2/tables", exist_ok=True)
    Ls = [L for L in BUDGETS if any(r["L"] == L for r in rows)]
    sub = {L: [r for r in rows if r["L"] == L] for L in Ls}
    # ---- table 1: per setting and budget
    cols = ["srs_dif", "group_dif", "pattern", "pdif_plain", "pdif_resid", "stratppi"]
    out = [r"\begin{tabular}{lr" + "r" * len(cols) + r"r rrr}\toprule",
           r"Setting & $L$ & \multicolumn{6}{c}{Efficiency vs SRS} & vs PATTERN & \multicolumn{3}{c}{PATTERN-DIF (plain) by estimand} \\",
           r" & & SRS-DIF & GROUP-DIF & PATTERN & DIF plain & DIF resid & StratPPI & DIF plain & $\theta^{TPR}$ & $\theta^{FPR}$ & $\theta^{ERR}$ \\\midrule"]
    last = None
    for r in rows:
        if last is not None and r["dataset"] != last: out.append(r"\addlinespace")
        vals = [_g(r, f"eff_{m}_vs_srs_all") for m in cols]
        best = np.nanargmax([v if np.isfinite(v) else -1 for v in vals])
        cells = [(r"\textbf{" + fmt(v) + "}") if k == best else fmt(v) for k, v in enumerate(vals)]
        cells += [fmt(_g(r, "eff_pdif_plain_vs_pattern_all"))] + [fmt(_g(r, f"eff_pdif_plain_vs_srs_{k}")) for k in ("dTPR", "dFPR", "dERR")]
        out.append((NAME[r["dataset"]] if r["dataset"] != last else "") + f" & {r['L']} & " + " & ".join(cells) + r" \\")
        last = r["dataset"]
    out.append(r"\bottomrule\end{tabular}")
    open("manuscript_v2/tables/efficiency_dif.tex", "w").write("\n".join(out) + "\n")
    # ---- table 2: inference by budget
    imeth = ["srs", "pattern", "pdif_plain", "pdif_resid", "pdif_plain_fz", "pdif_resid_fz", "pattern_fz"]
    out = [r"\resizebox{\textwidth}{!}{\begin{tabular}{rr" + "rr" * len(imeth) + r"r}\toprule",
           r"$L$ & Cells & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{LABEL[m]}}}" for m in imeth) + r" & Fail \\",
           r" & & " + " & ".join("mean & worst" for m in imeth) + r" & (max) \\\midrule"]
    for L in Ls:
        c = []
        for m in imeth:
            c += [f"{np.mean([r[f'cover_{m}'] for r in sub[L]]):.3f}", f"{min(r[f'cover_{m}_min'] for r in sub[L]):.2f}"]
        fail = max(max(r[f"fail_{m}"] for m in CORE) for r in sub[L])
        out.append(f"{L} & {len(sub[L])} & " + " & ".join(c) + f" & {fail:.3f}" + r" \\")
    out.append(r"\bottomrule\end{tabular}}")
    open("manuscript_v2/tables/inference_dif.tex", "w").write("\n".join(out) + "\n")
    # ---- table 3: decomposition (medians over cells of per-cell median efficiencies)
    steps = [("SRS-DIF vs SRS", "srs_dif", "srs"), ("PATTERN vs SRS", "pattern", "srs"), ("PATTERN-DIF vs PATTERN", "pdif_plain", "pattern"),
             ("PATTERN-DIF vs SRS-DIF", None, None), ("PATTERN-DIF vs SRS", "pdif_plain", "srs"), ("DIF resid vs DIF plain", "pdif_resid", "pdif_plain")]
    out = [r"\begin{tabular}{l" + "r" * len(Ls) + r"}\toprule", r"Step (MSE ratio) & " + " & ".join(f"$L={L}$" for L in Ls) + r" \\\midrule"]
    for lab, m, ref in steps:
        cells = []
        for L in Ls:
            if m is None:
                v = _med([_g(r, "eff_pdif_plain_vs_srs_all") / _g(r, "eff_srs_dif_vs_srs_all") for r in sub[L]])
            else:
                v = _med([_g(r, f"eff_{m}_vs_{ref}_all") for r in sub[L]])
            cells.append(fmt(v))
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    out.append(r"\bottomrule\end{tabular}")
    open("manuscript_v2/tables/decomposition_dif.tex", "w").write("\n".join(out) + "\n")
    # ---- summary json
    summ = {"note": "efficiency = MSE ratio, median over non-degenerate (live) estimands within a cell, then median over cells; "
                    "coverage = mean over estimands then over cells; worst = min over cells of the worst-estimand coverage",
            "cells": {str(L): len(sub[L]) for L in Ls}, "by_budget": {}, "overall": {}}
    for L in Ls:
        b = {}
        for m in METHODS:
            e = {"fail_max": max(r[f"fail_{m}"] for r in sub[L]),
                 "cover_mean": float(np.mean([r[f"cover_{m}"] for r in sub[L]])),
                 "cover_worst": float(min(r[f"cover_{m}_min"] for r in sub[L])),
                 "cover_p05_median": _med([r[f"cover_{m}_p05"] for r in sub[L]]),
                 "bias2share_median": _med([r[f"bias2share_{m}"] for r in sub[L]]),
                 "width_median": _med([r[f"width_{m}"] for r in sub[L]])}
            for ref in ("srs", "pattern", "pdif_plain"):
                for suf in ("all", "top"):
                    e[f"eff_vs_{ref}_{suf}"] = None if m == ref else _med([_g(r, f"eff_{m}_vs_{ref}_{suf}") for r in sub[L]])
            for k in ("dTPR", "dFPR", "dERR"):
                e[f"eff_vs_srs_{k}"] = None if m == "srs" else _med([_g(r, f"eff_{m}_vs_srs_{k}") for r in sub[L]])
            if m in CORE:
                e["best_share_estimand_median"] = _med([_g(r, f"best_share_{m}") for r in sub[L]])
                e["cells_best_by_median_eff"] = int(sum(r.get("best_cell") == m for r in sub[L]))
            b[m] = e
        summ["by_budget"][str(L)] = b
    rr = [(r["dataset"], r["L"], _g(r, "eff_pdif_resid_vs_pdif_plain_all")) for r in rows]
    fin = [x for x in rr if np.isfinite(x[2])]
    summ["overall"]["resid_vs_plain"] = {"cells_resid_better": int(sum(x[2] > 1 for x in fin)), "cells": len(fin),
                                        "median_ratio": _med([x[2] for x in fin]),
                                        "cells_resid_better_by_over_10pct": int(sum(x[2] > 1.1 for x in fin)),
                                        "cells_plain_better_by_over_10pct": int(sum(x[2] < 1 / 1.1 for x in fin)),
                                        "per_cell": [[a, int(b), None if not np.isfinite(c) else float(c)] for a, b, c in rr]}
    summ["overall"]["dif_plain_vs_pattern"] = {"cells_better": int(sum(_g(r, "eff_pdif_plain_vs_pattern_all") > 1 for r in rows)), "cells": len(rows)}
    summ["overall"]["cover_all_cells"] = {m: {"mean": float(np.mean([r[f"cover_{m}"] for r in rows])),
                                              "worst": float(min(r[f"cover_{m}_min"] for r in rows)),
                                              "worst_cell": min(((r[f"cover_{m}_min"], r["dataset"], r["L"]) for r in rows))[1:]}
                                          for m in METHODS}
    json.dump(summ, open("results/label_efficiency_dif_summary.json", "w"), indent=1)
    # ---- figure
    colr = {"srs": "black", "srs_dif": "#8d6cab", "group_dif": "#edae49", "pattern": "#1b4f9c", "pdif_plain": "#d1495b",
            "pdif_resid": "#00798c", "stratppi": "#66a182"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1, 1.35, 1]))
    ax = axes[0]
    for m in ["srs_dif", "group_dif", "pattern", "pdif_plain", "pdif_resid", "stratppi"]:
        med = [np.clip(_med([_g(r, f"eff_{m}_vs_srs_all") for r in sub[L]]), 0.5, 1e3) for L in Ls]
        q1 = [np.nanquantile(np.clip([_g(r, f"eff_{m}_vs_srs_all") for r in sub[L]], 0.5, 1e3), 0.25) for L in Ls]
        q3 = [np.nanquantile(np.clip([_g(r, f"eff_{m}_vs_srs_all") for r in sub[L]], 0.5, 1e3), 0.75) for L in Ls]
        ax.plot(Ls, med, "-o", color=colr[m], label=LABEL[m], ms=4); ax.fill_between(Ls, q1, q3, color=colr[m], alpha=0.10)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(Ls); ax.set_xticklabels(Ls); ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("labels L"); ax.set_ylabel("MSE(SRS) / MSE(design)"); ax.grid(alpha=0.3)
    ax.set_title("Efficiency vs SRS: median over cells, IQR band", fontsize=9); ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]; xs = np.arange(len(KEYS)); off = {"srs_dif": -0.24, "pattern": -0.08, "pdif_plain": 0.08, "pdif_resid": 0.24}
    for m in ("srs_dif", "pattern", "pdif_plain", "pdif_resid"):
        for i, key in enumerate(KEYS):
            v = [np.clip(_g(r, f"eff_{m}_vs_srs_all"), 0.5, 300) for r in rows if r["dataset"] == key]
            ax.scatter(np.full(len(v), i) + off[m], v, s=16, color=colr[m], label=LABEL[m] if i == 0 else None, alpha=0.85, zorder=3)
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels([NAME[k] for k in KEYS], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3); ax.axhline(1, color="grey", lw=0.8, ls="--"); ax.set_ylabel("MSE(SRS) / MSE(design)")
    ax.set_title("Efficiency vs SRS by setting (one point per budget; clipped at 300)", fontsize=9); ax.legend(fontsize=7, ncol=2)
    ax = axes[2]; ms = ["srs", "pattern", "pdif_plain", "pdif_resid", "pdif_plain_fz", "pdif_resid_fz"]
    for k, m in enumerate(ms):
        c = [r[f"cover_{m}"] for r in rows]
        ax.scatter(np.full(len(c), k) + np.random.default_rng(k).uniform(-0.18, 0.18, len(c)), c, s=9, color="#555", alpha=0.6)
        ax.scatter([k], [np.median([r[f"cover_{m}_min"] for r in rows])], marker="_", s=170, color="black", zorder=4)
    ax.axhline(0.95, color="grey", lw=0.8, ls="--"); ax.set_ylim(0.4, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([LABEL[m] for m in ms], rotation=45, ha="right", fontsize=7.5)
    ax.set_title("Coverage: dots = mean over estimands per cell;\nbar = median worst-estimand coverage", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"manuscript_v2/figures/efficiency_dif.{ext}", dpi=200)
    print(json.dumps({L: {m: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ["by_budget"][str(L)][m].items()
                              if k in ("eff_vs_srs_all", "eff_vs_pattern_all", "cover_mean", "cover_worst", "fail_max")} for m in METHODS} for L in Ls}, indent=0))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        main_run(int(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else KEYS)
    else:
        report()
