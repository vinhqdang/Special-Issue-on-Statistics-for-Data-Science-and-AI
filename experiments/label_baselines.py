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
        python experiments/label_baselines.py report            -> figures and LaTeX table
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
        cov = ((lo[m][ok] <= truth) & (truth <= hi[m][ok])).mean(0)[live]
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


def fmt(x):
    if x is None or not np.isfinite(x): return "--"
    if x >= 100: return "$>$100"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = json.load(open(OUT)); os.makedirs("manuscript_v2/figures", exist_ok=True)
    main = ["pattern", "is", "score10", "dis25", "dis50", "ppipp", "stratppi", "pattern_ppi"]
    # ---- table: L = 500 (all settings), efficiency vs SRS (all targets), coverage
    L0 = 500
    sub = [r for r in rows if r["L"] == L0]
    lines = []
    for r in sub:
        lines.append(NAME[r["dataset"]] + " & " + " & ".join(fmt(r[f"eff_{m}_vs_srs_all"]) for m in main) + r" \\")
    med = lambda m, k: float(np.median([r[f"eff_{m}_vs_{k}"] for r in rows if r.get(f"eff_{m}_vs_{k}") is not None]))
    lines.append(r"\midrule")
    lines.append(r"Median, all $(s,L)$ & " + " & ".join(fmt(med(m, "srs_all")) for m in main) + r" \\")
    lines.append(r"Median, top-3 pairs & " + " & ".join(fmt(med(m, "srs_top")) for m in main) + r" \\")
    lines.append(r"Median vs PATTERN & " + " & ".join("1" if m == "pattern" else fmt(med(m, "pattern_all")) for m in main) + r" \\")
    cov = lambda m: float(np.mean([r[f"cover_{m}"] for r in rows if f"cover_{m}" in r]))
    covmin = lambda m: float(np.min([r[f"cover_{m}_min"] for r in rows if f"cover_{m}_min" in r]))
    fail = lambda m: float(np.max([r[f"fail_{m}"] for r in rows]))
    lines.append(r"\midrule")
    lines.append(r"Coverage, mean & " + " & ".join(f"{cov(m):.3f}" for m in main) + r" \\")
    lines.append(r"Coverage, worst target & " + " & ".join(f"{covmin(m):.2f}" for m in main) + r" \\")
    lines.append(r"Failure rate, max & " + " & ".join(f"{fail(m):.3f}" for m in main) + r" \\")
    tms = lambda m: float(np.median([r[f"time_{m}"] * 1e3 for r in rows]))
    lines.append(r"Time per replicate (ms) & " + " & ".join(f"{tms(m):.0f}" for m in main) + r" \\")
    open("results/label_baselines_table.tex", "w").write("\n".join(lines) + "\n")
    # ---- figure: efficiency vs SRS by setting, L = 500 and L = 1000 panels + coverage panel
    cols = {"pattern": "#1b4f9c", "is": "#d1495b", "score10": "#edae49", "dis25": "#66a182", "dis50": "#2e8b57",
            "ppipp": "#8d6cab", "stratppi": "#b07aa1", "pattern_ppi": "#00798c"}
    marks = {"pattern": "o", "is": "v", "score10": "s", "dis25": "<", "dis50": ">", "ppipp": "D", "stratppi": "P", "pattern_ppi": "*"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    for ax, L in zip(axes[:2], (500, 1000)):
        sub = [r for r in rows if r["L"] == L]; xs = np.arange(len(sub))
        for k, m in enumerate(main):
            v = [min(r[f"eff_{m}_vs_srs_all"] or np.nan, 200) for r in sub]
            ax.scatter(xs + (k - 3.5) * 0.08, v, s=28 if m != "pattern" else 44, marker=marks[m], color=cols[m], label=LABEL[m], zorder=3,
                       edgecolor="black" if m == "pattern" else "none", linewidth=0.5)
        ax.axhline(1, color="grey", lw=0.8, ls="--"); ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels([NAME[r["dataset"]] for r in sub], rotation=55, ha="right", fontsize=8)
        ax.set_title(f"L = {L}: MSE efficiency vs SRS (median over targets)", fontsize=9); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("MSE(SRS) / MSE(method)")
    ax = axes[2]
    for k, m in enumerate(["srs"] + main):
        c = [r[f"cover_{m}"] for r in rows if f"cover_{m}" in r]; w = [r[f"cover_{m}_min"] for r in rows if f"cover_{m}_min" in r]
        ax.scatter(np.full(len(c), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(c)), c, s=10, color=cols.get(m, "black"), alpha=0.7)
        ax.scatter([k], [np.median(w)], marker="_", s=150, color="black")
    ax.axhline(0.95, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(main) + 1)); ax.set_xticklabels([LABEL[m].split(" (")[0] for m in ["srs"] + main], rotation=55, ha="right", fontsize=8)
    ax.set_title("95% CI coverage: mean over targets per (setting, L);\nbar = median of worst-target coverage", fontsize=9)
    ax.set_ylim(0.5, 1.0); ax.grid(axis="y", alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"manuscript_v2/figures/base_efficiency.{ext}", dpi=200)
    print(open("results/label_baselines_table.tex").read())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        main_run(int(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else KEYS)
    else:
        report()
