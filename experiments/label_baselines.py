"""Baselines for label-efficient comparison of fixed classifiers on fairness contrasts.

Runs, on the cached pools of the 11 settings (preference_validation.SETTINGS minus 'nomao') and
budgets L in {250, 500, 1000, 2000} (only L <= N/4), R replications of
  srs          simple random sampling, plug-in, linearisation CI          (tcs.labeling)
  pattern      PATTERN strata, proxy-Neyman allocation                    (tcs.labeling, proposed)
  is           B1  importance sampling, Sawade et al. (2012) style         (tcs.labeling_baselines)
  score10      B2  group x 10 proxy-quantile strata, same Neyman rule
  score5       B2  group x 5 proxy-quantile strata
  dis25, dis50 B3  SRS of 25% / 50% of L, rest by SRS on the any-disagreement set
  ppi, ppipp   B4  PPI (difference estimator) and PPI++ (power-tuned) on the srs sample
  stratppi     B4  StratPPI: group x 10 proxy bins, residual-Neyman allocation, per-stratum tuning
  pattern_dif  B5  PATTERN sample, stratified difference estimator (lam = 1)
  pattern_ppi  B5  PATTERN sample, stratified power-tuned difference estimator
Labels in the pool are used only to answer label queries and to compute the truth.

Usage:  python experiments/label_baselines.py run R [keys]     -> results/label_baselines.json
        python experiments/label_baselines.py report            -> figures, manuscript_v2/tables/baselines_*.tex
"""
import sys, os, json, zlib, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, make_strata, neyman_allocation, proportional_allocation, estimate, draw
from tcs.labeling_baselines import (VecEstimands, is_distribution, is_draw, is_estimate, score_strata, dis_draw,
                                    ppi_estimate, stratppi_allocation)
from experiments.preference_validation import SETTINGS

KEYS = [k for k in SETTINGS if k != "nomao"]
BUDGETS = (250, 500, 1000, 2000)
METHODS = ["srs", "pattern", "is", "score10", "score5", "dis25", "dis50", "ppi", "ppipp", "stratppi", "pattern_dif", "pattern_ppi"]
OUT = os.environ.get("LB_OUT", "results/label_baselines.json")
S = {}


def setup(key):
    z = np.load(f"results/pool_cache/{key}.npz", allow_pickle=True)
    pred, g, y, p = z["pred"], z["g"].astype(int), z["y"].astype(float), z["p"].mean(1)
    U, V, names = total_columns(pred, g); N, M = pred.shape
    est = Estimands(names, M, N); ve = VecEstimands(est)
    truth = est.value((y[:, None] * U + (1 - y[:, None]) * V).sum(0))
    err = (pred != y[:, None]).mean(0); top = np.argsort(err)[:3]        # pool labels: reporting only
    top_pairs = [k for k, (m, j) in enumerate(est.pairs) if m in top and j in top]
    # targets estimated exactly by every design (numerator identically zero in the pool): excluded from summaries
    live = []
    for (m, j) in est.pairs:
        dif = pred[:, m] != pred[:, j]
        live += [bool((dif & (y == 1)).any()), bool((dif & (y == 0)).any()), bool(dif.any())]
    K = pred.sum(1)
    return dict(pred=pred, g=g, y=y, p=p, U=U, V=V, est=est, ve=ve, truth=truth, N=N, M=M, live=np.array(live),
                top_targets=np.array([3 * k + r for k in top_pairs for r in range(3)]),
                kinds=np.array([k for (k, _, _) in est.labels]),
                share_D=float(((K > 0) & (K < M)).mean()))


def designs(d, L):
    t0 = time.perf_counter(); out = {}; tm = {}
    one = np.zeros(d["N"], int)
    out["srs"] = (one,) + proportional_allocation(one, L); tm["srs"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sp = make_strata(d["pred"], d["g"], "pattern"); out["pattern"] = (sp,) + neyman_allocation(sp, d["U"], d["V"], d["p"], d["est"], L)
    tm["pattern"] = time.perf_counter() - t0
    t0 = time.perf_counter(); out["is"] = is_distribution(d["U"], d["V"], d["p"], d["ve"]); tm["is"] = time.perf_counter() - t0
    for b in (10, 5):
        t0 = time.perf_counter()
        ss = score_strata(d["p"], d["g"], b); out[f"score{b}"] = (ss,) + neyman_allocation(ss, d["U"], d["V"], d["p"], d["est"], L)
        tm[f"score{b}"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ss = score_strata(d["p"], d["g"], 10); out["stratppi"] = (ss,) + stratppi_allocation(ss, d["U"], d["V"], d["p"], d["ve"], L)
    tm["stratppi"] = time.perf_counter() - t0
    return out, tm


def run(args):
    key, L, seed = args
    d = S[key]; D = S[(key, L)]; rng = np.random.default_rng(seed); res = {}; tm = {}
    U, V, y, p, ve = d["U"], d["V"], d["y"], d["p"], d["ve"]
    for name in ("srs", "pattern", "score10", "score5"):
        t0 = time.perf_counter()
        st, ids, Nh, nh = D[name]; idx = draw(st, ids, nh, rng)
        res[name] = estimate(st, ids, Nh, nh, U, V, y[idx], idx, d["est"]); tm[name] = time.perf_counter() - t0
        if name == "srs":
            for nm, tune in (("ppi", False), ("ppipp", True)):
                t0 = time.perf_counter()
                res[nm] = ppi_estimate(st, ids, Nh, nh, idx, y, U, V, p, ve, tune=tune); tm[nm] = time.perf_counter() - t0 + tm["srs"]
        if name == "pattern":
            for nm, tune in (("pattern_dif", False), ("pattern_ppi", True)):
                t0 = time.perf_counter()
                res[nm] = ppi_estimate(st, ids, Nh, nh, idx, y, U, V, p, ve, tune=tune); tm[nm] = time.perf_counter() - t0 + tm["pattern"]
    t0 = time.perf_counter()
    q = D["is"]; dr = is_draw(q, L, rng); res["is"] = is_estimate(q, dr, y, U, V, ve); tm["is"] = time.perf_counter() - t0
    ndraw = len(dr)
    for nm, fr in (("dis25", 0.25), ("dis50", 0.5)):
        t0 = time.perf_counter()
        idx, st, ids, Nh, nh = dis_draw(d["pred"], L, fr, rng)
        res[nm] = estimate(st, ids, Nh, nh, U, V, y[idx], idx, d["est"]); tm[nm] = time.perf_counter() - t0
    t0 = time.perf_counter()
    st, ids, Nh, nh = D["stratppi"]; idx = draw(st, ids, nh, rng)
    res["stratppi"] = ppi_estimate(st, ids, Nh, nh, idx, y, U, V, p, ve, tune=True); tm["stratppi"] = time.perf_counter() - t0
    return {m: tuple(np.asarray(x, float) for x in res[m]) for m in METHODS}, tm, ndraw


def summarise(d, key, L, R, res, dtime):
    truth = d["truth"]; live = d["live"]; top = np.array([t for t in d["top_targets"] if live[t]], int)
    th = {m: np.array([x[0][m][0] for x in res]) for m in METHODS}
    lo = {m: np.array([x[0][m][2] for x in res]) for m in METHODS}
    hi = {m: np.array([x[0][m][3] for x in res]) for m in METHODS}
    se = {m: np.array([x[0][m][1] for x in res]) for m in METHODS}
    fin = {m: np.isfinite(th[m]).all(1) & np.isfinite(se[m]).all(1) for m in METHODS}
    row = dict(dataset=key, N=d["N"], L=L, R=R, n_targets=int(len(truth)), n_live=int(live.sum()), n_top_live=int(len(top)),
               share_D=d["share_D"], is_draws_mean=float(np.mean([x[2] for x in res])))
    for m in METHODS:
        row[f"fail_{m}"] = float(1 - fin[m].mean())
        row[f"time_{m}"] = float(np.mean([x[1][m] for x in res]))
        row[f"design_time_{m}"] = float(dtime.get(m, dtime.get({"ppi": "srs", "ppipp": "srs", "pattern_dif": "pattern", "pattern_ppi": "pattern"}.get(m, m), 0.0)))
        ok = fin[m]
        if ok.sum() == 0:
            continue
        tol = 1e-9 * (1 + np.abs(truth))                                  # exact estimates with zero-width CIs count as covered
        cov = ((lo[m][ok] - tol <= truth) & (truth <= hi[m][ok] + tol)).mean(0)[live]
        row[f"cover_{m}"] = float(cov.mean()); row[f"cover_{m}_min"] = float(cov.min())
        row[f"cover_{m}_top"] = float(cov[np.isin(np.flatnonzero(live), top)].mean()) if len(top) else None
        row[f"bias2share_{m}"] = float(np.median(((th[m][ok].mean(0) - truth) ** 2 / np.maximum(((th[m][ok] - truth) ** 2).mean(0), 1e-30))[live]))
        row[f"width_{m}"] = float(np.median((hi[m][ok] - lo[m][ok]).mean(0)[live]))
    for ref in ("srs", "pattern"):
        for m in METHODS:
            if m == ref: continue
            both = fin[ref] & fin[m]
            if both.sum() < 10: row[f"eff_{m}_vs_{ref}_all"] = None; row[f"eff_{m}_vs_{ref}_top"] = None; continue
            eff = ((th[ref][both] - truth) ** 2).mean(0) / np.maximum(((th[m][both] - truth) ** 2).mean(0), 1e-30)
            row[f"eff_{m}_vs_{ref}_all"] = float(np.median(eff[live]))
            row[f"eff_{m}_vs_{ref}_top"] = float(np.median(eff[top])) if len(top) else None
            if ref == "srs":
                for k in ("dTPR", "dFPR", "dERR"):
                    row[f"eff_{m}_vs_srs_{k}"] = float(np.median(eff[live & (d["kinds"] == k)]))
    return row


def main_run(R, keys):
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    rows = [r for r in rows if r["dataset"] not in keys]
    for key in keys:
        d = setup(key); S[key] = d
        for L in BUDGETS:
            if L > d["N"] / 4: continue
            S[(key, L)], dtime = designs(d, L)
            t0 = time.perf_counter()
            with Pool(2) as pool:
                res = pool.map(run, [(key, L, zlib.crc32(key.encode()) % 9973 + 31 * L + r + 100003) for r in range(R)])
            row = summarise(d, key, L, R, res, dtime); row["wall_seconds"] = time.perf_counter() - t0
            rows.append(row)
            print(key, L, {m: (round(row.get(f"eff_{m}_vs_srs_all") or 0, 2), round(row.get(f"cover_{m}", 0), 3), round(row[f"fail_{m}"], 3))
                           for m in METHODS if m != "srs"}, "pattern-vs-srs", round(row["eff_pattern_vs_srs_all"], 2), round(row["wall_seconds"]), "s", flush=True)
            json.dump(rows, open(OUT, "w"), indent=1)
        del S[key]


# ---------------------------------------------------------------- reporting
LABEL = {"srs": "SRS", "pattern": "PATTERN", "is": "IS (B1)", "score10": "Score-10 (B2)", "score5": "Score-5 (B2)",
         "dis25": "Dis-25 (B3)", "dis50": "Dis-50 (B3)", "ppi": "PPI (B4)", "ppipp": "PPI++ (B4)", "stratppi": "StratPPI (B4)",
         "pattern_dif": "PATTERN-DIF (B5)", "pattern_ppi": "PATTERN-PPI (B5)"}
NAME = {"adult-sex": "Adult-sex", "adult-race": "Adult-race", "bank": "Bank", "credit-card": "Credit-card", "law-school": "Law school",
        "magic": "MAGIC", "eeg": "EEG", "phoneme": "Phoneme", "spambase": "Spambase", "churn": "Churn", "electricity": "Electricity"}


def fmt(x, big=100):
    if x is None or not np.isfinite(x): return "--"
    if x >= big: return f"$>${big}"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def _med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = json.load(open(OUT)); os.makedirs("manuscript_v2/figures", exist_ok=True); os.makedirs("manuscript_v2/tables", exist_ok=True)
    rows.sort(key=lambda r: (KEYS.index(r["dataset"]), r["L"]))
    order = ["pattern", "is", "score10", "score5", "dis25", "dis50", "ppi", "ppipp", "stratppi", "pattern_dif", "pattern_ppi"]
    # ---- Table 1: by method and budget (medians over settings of the per-setting medians over targets)
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Baselines by label budget $L$ (number of settings in brackets). Efficiency: ratio of mean squared errors, "
           r"median over the non-degenerate comparison estimands within a setting, then median over settings; ``vs SRS'' is "
           r"MSE(SRS)/MSE(method), ``vs PATTERN'' is MSE(PATTERN)/MSE(method), so values above 1 favour the method. "
           r"Coverage: nominal 95\% Wald intervals, mean over estimands and settings; ``worst'': lowest single-estimand coverage over all "
           r"settings and budgets. $R=200$ replications per setting and budget. Source: \texttt{results/label\_baselines.json}.}\label{tab:base-budget}",
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "rr" * 4 + r"rrr}\toprule"]
    Ls = [L for L in BUDGETS if any(r["L"] == L for r in rows)]
    out.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$L={L}$ ({sum(r['L'] == L for r in rows)})}}" for L in Ls)
               + r" & \multicolumn{3}{c}{All $(s,L)$} \\")
    out.append(" ".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(Ls))) + rf" \cmidrule(lr){{{2 + 2 * len(Ls)}-{4 + 2 * len(Ls)}}}")
    out.append("Method & " + " & ".join(["vs SRS & vs PAT."] * len(Ls)) + r" & Cover. & Worst & ms \\ \midrule")
    for m in ["srs"] + order:
        cells = []
        for L in Ls:
            sub = [r for r in rows if r["L"] == L]
            cells.append("1" if m == "srs" else fmt(_med([r[f"eff_{m}_vs_srs_all"] for r in sub])))
            cells.append("1" if m == "pattern" else fmt(_med([r[f"eff_{m}_vs_pattern_all"] for r in sub])))
        cov = np.mean([r[f"cover_{m}"] for r in rows]); worst = min(r[f"cover_{m}_min"] for r in rows)
        ms = np.median([r[f"time_{m}"] * 1e3 for r in rows])
        if m == "pattern_dif": out.append(r"\midrule \multicolumn{" + str(4 + 2 * len(Ls)) + r"}{l}{\emph{Extension (B5): PATTERN sample, model-assisted estimator}} \\")
        out.append(f"{LABEL[m]} & " + " & ".join(cells) + f" & {cov:.3f} & {worst:.2f} & {ms:.0f}" + r" \\")
    out += [r"\bottomrule\end{tabular}}\end{table}"]
    open("manuscript_v2/tables/baselines_budget.tex", "w").write("\n".join(out) + "\n")
    # ---- Table 2: by setting at L = 500 (L = 250 where 500 > N/4), vs PATTERN, with coverage of the best competitor
    main = ["is", "score10", "dis25", "dis50", "ppipp", "stratppi", "pattern_dif", "pattern_ppi"]
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Efficiency relative to PATTERN, MSE(PATTERN)/MSE(method), median over estimands, by setting at $L=500$ "
           r"($L=250$ for Spambase, whose pool is too small for $L=500$). Values above 1 mean that the method beats PATTERN. "
           r"Second column: efficiency of PATTERN relative to SRS. Last two columns: mean coverage of the 95\% intervals of StratPPI "
           r"and of PATTERN-PPI. Source: \texttt{results/label\_baselines.json}.}\label{tab:base-setting}",
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\resizebox{\textwidth}{!}{\begin{tabular}{l r" + "r" * len(main) + r"rr}\toprule",
           r" & PAT. & \multicolumn{6}{c}{Baselines (B1--B4) vs PATTERN} & \multicolumn{2}{c}{B5 vs PATTERN} & \multicolumn{2}{c}{Coverage} \\",
           r"\cmidrule(lr){3-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
           r"Setting & vs SRS & " + " & ".join(LABEL[m].split(" (")[0] for m in main) + r" & StratPPI & PAT-PPI \\ \midrule"]
    for key in KEYS:
        cand = [r for r in rows if r["dataset"] == key and r["L"] == 500] or [r for r in rows if r["dataset"] == key and r["L"] == 250]
        if not cand: continue
        r = cand[0]
        cells = [fmt(r["eff_pattern_vs_srs_all"])] + [(r"\textbf{" + fmt(r[f"eff_{m}_vs_pattern_all"]) + "}") if (r[f"eff_{m}_vs_pattern_all"] or 0) > 1 and m not in ("pattern_dif", "pattern_ppi") else fmt(r[f"eff_{m}_vs_pattern_all"]) for m in main]
        cells += [f"{r['cover_stratppi']:.3f}", f"{r['cover_pattern_ppi']:.3f}"]
        out.append(NAME[key] + (r"$^{\ast}$" if r["L"] != 500 else "") + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule\end{tabular}}\end{table}"]
    open("manuscript_v2/tables/baselines_setting.tex", "w").write("\n".join(out) + "\n")
    # ---- figure: efficiency vs PATTERN by setting (all budgets), and coverage
    cols = {"is": "#d1495b", "score10": "#edae49", "dis25": "#66a182", "dis50": "#2e8b57", "ppipp": "#8d6cab",
            "stratppi": "#3b6fb6", "pattern_dif": "#7f7f7f", "pattern_ppi": "#00798c", "srs": "black", "pattern": "#1b4f9c"}
    marks = {"is": "v", "score10": "s", "dis25": "<", "dis50": ">", "ppipp": "D", "stratppi": "P", "pattern_dif": "x", "pattern_ppi": "*"}
    show = ["is", "score10", "dis25", "ppipp", "stratppi", "pattern_ppi"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axes[0]; xs = np.arange(len(KEYS))
    for k, m in enumerate(show):
        for key_i, key in enumerate(KEYS):
            sub = [r for r in rows if r["dataset"] == key]
            v = [np.clip(r[f"eff_{m}_vs_pattern_all"], 1e-2, 1e2) for r in sub]
            ax.scatter(np.full(len(v), key_i) + (k - 2.5) * 0.12, v, s=22, marker=marks[m], color=cols[m],
                       label=LABEL[m] if key_i == 0 else None, zorder=3, alpha=0.85)
    ax.axhline(1, color="grey", lw=0.9, ls="--"); ax.set_yscale("log"); ax.set_ylim(0.02, 30)
    ax.set_xticks(xs); ax.set_xticklabels([NAME[k] for k in KEYS], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MSE(PATTERN) / MSE(method)"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Efficiency relative to PATTERN (median over estimands), one point per budget", fontsize=9)
    ax.legend(fontsize=7.5, ncol=3, loc="lower left", frameon=True)
    ax = axes[1]; ms = ["srs", "pattern"] + show
    for k, m in enumerate(ms):
        c = [r[f"cover_{m}"] for r in rows]
        ax.scatter(np.full(len(c), k) + np.random.default_rng(k).uniform(-0.18, 0.18, len(c)), c, s=10, color=cols[m], alpha=0.7)
        ax.scatter([k], [np.median([r[f"cover_{m}_min"] for r in rows])], marker="_", s=180, color="black", zorder=4)
    ax.axhline(0.95, color="grey", lw=0.8, ls="--"); ax.set_ylim(0.4, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([LABEL[m].split(" (")[0] for m in ms], rotation=45, ha="right", fontsize=8)
    ax.set_title("Coverage of 95% intervals: dots = mean over estimands\nper (setting, L); bar = median worst-estimand coverage", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"manuscript_v2/figures/base_efficiency.{ext}", dpi=200)
    print(open("manuscript_v2/tables/baselines_budget.tex").read()); print(open("manuscript_v2/tables/baselines_setting.tex").read())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        main_run(int(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else KEYS)
    else:
        report()
