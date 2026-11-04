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
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, zlib, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import t as tdist
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, draw
from tcs.labeling_variants import (strata_kcount, strata_binary, strata_joint, strata_kproxy, top_agreement_models,
                                   proxies, allocate, estimate_full, bootstrap_percentile, eo_gap, GapEstimands,
                                   kendall_tau, proxy_cov)

SETTINGS = ["adult-sex", "adult-race", "bank", "credit-card", "law-school", "magic", "eeg", "phoneme",
            "spambase", "churn", "electricity"]
BUDGETS = (250, 500, 1000)
SUBSET_SIZES = (2, 3, 5)
N_SUBSETS = 3
EO_DESIGNS = ["srs", "group", "pattern", "pattern_gap", "oracle"]
BOOT_DESIGNS = ["srs", "pattern"]
FLOOR_DESIGNS = ["srs", "pattern", "grid_n5_c25", "grid_n20_c100", "kproxy2", "binary"]   # proxy variance floor
CLIP = 1e4   # per-target MSE ratios are clipped to [1/CLIP, CLIP] (a design can make a target exact)
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
    for n in FLOOR_DESIGNS:
        D[n]["sigma"] = proxy_cov(D[n]["strata"], D[n]["ids"], U, V, pi)
    return dict(base=base, sub=SUB, D=D, y=y, g=g, top3=top3, M=M, N=N)


def mse_ratio(a, b, truth):
    """Per-target MSE(a)/MSE(b) on samples where both are finite; NaN where both MSEs are 0."""
    both = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    ma = ((a[both] - truth) ** 2).mean(0); mb = ((b[both] - truth) ** 2).mean(0)
    r = np.clip(ma / np.maximum(mb, 1e-300), 1 / CLIP, CLIP)
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
    th = {n: [] for n in names}; se = {n: [] for n in names}; df = {n: [] for n in names}; sef = {n: [] for n in FLOOR_DESIGNS}
    eo = {n: [] for n in EO_DESIGNS}; blo = {n: [] for n in BOOT_DESIGNS}; bhi = {n: [] for n in BOOT_DESIGNS}
    crc = zlib.crc32(key.encode()) % 9973
    for r in range(R):
        for di, n in enumerate(names):
            d = D[n]; cfg = base if d["cfg"] == "main" else SUB[d["cfg"]]
            rng = np.random.default_rng([crc, L, r, di])
            idx = draw(d["strata"], d["ids"], d["nh"], rng)
            res = estimate_full(d["strata"], d["ids"], d["Nh"], d["nh"], cfg["U"], cfg["V"], y[idx], idx, cfg["est"],
                                d.get("sigma"))
            a, s, f, T = res[:4]
            if n in FLOOR_DESIGNS: sef[n].append(res[4])
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
        if n in FLOOR_DESIGNS:
            SF = np.array(sef[n]); cf = coverage(A - Z * SF, A + Z * SF, truth, ok)
            rec["cover_floor"] = cf[0]; rec["cover_floor_min"] = cf[1]
            rec["width_floor_over_z"] = float(np.nanmean(SF[ok]) / np.nanmean(Sd[ok]))
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
    srt = np.sort(eo_true); bestset = np.flatnonzero(eo_true <= srt[0] + 1e-12)
    top2 = np.flatnonzero(eo_true <= srt[min(1, len(srt) - 1)] + 1e-12)
    out["eo_truth"] = eo_true.tolist(); out["eo_n_best"] = int(len(bestset))
    out["eo_margin"] = float(srt[len(bestset)] - srt[0]) if len(bestset) < len(srt) else 0.0
    for n in EO_DESIGNS:
        E = np.array(eo[n]); ok = np.isfinite(E).all(1); E = E[ok]
        out["eo"][n] = dict(fail=float(1 - ok.mean()),
                            rmse=float(np.sqrt(((E - eo_true) ** 2).mean(0)).mean()),
                            bias=float((E - eo_true).mean()),
                            p_best=float(np.mean(np.isin(E.argmin(1), bestset))),
                            p_best_top2=float(np.mean(np.isin(E.argmin(1), top2))),
                            tau=float(np.mean([kendall_tau(e, eo_true) for e in E])))
    for n in EO_DESIGNS[1:]:
        Es = np.array(eo["srs"]); Ed = np.array(eo[n]); both = np.isfinite(Es).all(1) & np.isfinite(Ed).all(1)
        out["eo"][n]["eff_vs_srs"] = float(np.nanmedian(1 / mse_ratio(Ed, Es, eo_true)))
    # Monte Carlo reference for the minimum over targets of R-replicate coverage when every target covers 95%
    nt = len(base["truth"]); mc = np.random.default_rng(1).binomial(R, 0.95, size=(2000, nt)).min(1) / R
    out["mc_min_ref"] = dict(mean=float(mc.mean()), q05=float(np.quantile(mc, 0.05)))
    out["seconds"] = time.time() - t0
    print(key, L, f"{out['seconds']:.0f}s", "pattern cover", round(out["designs"]["pattern"]["cover_z_min"], 3), flush=True)
    return out


# ---------------------------------------------------------------- report: tables (printed LaTeX rows) and figures
SMALL = ["churn", "spambase", "phoneme", "law-school"]
SHORT = {"adult-sex": "Adult-sex", "adult-race": "Adult-race", "bank": "Bank", "credit-card": "Credit", "law-school": "Law",
         "magic": "MAGIC", "eeg": "EEG", "phoneme": "Phoneme", "spambase": "Spambase", "churn": "Churn", "electricity": "Elec."}


def report(path="results/label_ablations.json", figdir="manuscript_v2/figures"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = json.load(open(path)); f2 = lambda x: "--" if x is None or not np.isfinite(x) else f"{x:.2f}"
    f3 = lambda x: "--" if x is None or not np.isfinite(x) else f"{x:.3f}"
    byL = lambda L: [r for r in rows if r["L"] == L]
    med = lambda v: float(np.median(v)) if len(v) else np.nan
    print("cells", len(rows), "R", rows[0]["R"], "B", rows[0]["B"], "mc_min_ref", rows[0]["mc_min_ref"])
    # ---- A1/A2 summary table
    print("\n% TABLE A1/A2 summary: median eff vs SRS by L | median rel. default (all cells) | share cells rel>1 | mean cov | mean worst | min worst")
    for n in ["pattern", "binary", "joint3", "kproxy2", "kproxy3", "single0", "single7", "const", "shuffle", "noise", "flip",
              "prop", "oracle", "group", "pattern_gap"]:
        e = [f2(med([r["designs"][n]["eff_vs_srs"] for r in byL(L)])) for L in BUDGETS]
        rel = [r["designs"][n].get("eff_vs_pattern", 1.0) for r in rows]
        cz = [r["designs"][n]["cover_z"] for r in rows]; cm = [r["designs"][n]["cover_z_min"] for r in rows]
        print(f"{n} & " + " & ".join(e) + f" & {f2(med(rel))} & {np.mean(np.array(rel) > 1):.2f} & {f3(np.mean(cz))} & {f3(np.mean(cm))} & {f3(np.min(cm))} \\\\")
    # ---- A4 inference summary
    print("\n% TABLE A4: method | L=250 all: mean cov, mean worst, min worst | L=250 small pools: mean worst, min worst | L>=500: mean worst, min worst | width ratio")
    def a4(name, d, ck, wk=None):
        out = []
        for sel in (lambda r: r["L"] == 250, lambda r: r["L"] == 250 and r["dataset"] in SMALL, lambda r: r["L"] >= 500):
            rr = [r for r in rows if sel(r)]
            out.append((np.mean([r["designs"][d][ck] for r in rr]), np.mean([r["designs"][d][ck + "_min"] for r in rr]),
                        np.min([r["designs"][d][ck + "_min"] for r in rr])))
        w = np.mean([r["designs"][d][wk] for r in rows]) if wk else 1.0
        print(f"{name} & {f3(out[0][0])} & {f3(out[0][1])} & {f3(out[0][2])} & {f3(out[1][1])} & {f3(out[1][2])} & {f3(out[2][1])} & {f3(out[2][2])} & {f2(w)} \\\\")
    for d in ("pattern", "srs"):
        a4(d + " Wald", d, "cover_z"); a4(d + " t", d, "cover_t", "width_t_over_z"); a4(d + " boot", d, "cover_boot")
        a4(d + " floor", d, "cover_floor", "width_floor_over_z")
    for d in ("binary", "kproxy2", "grid_n5_c25", "grid_n20_c100"):
        a4(d + " floor", d, "cover_floor", "width_floor_over_z")
    for mn in (5, 10, 20):
        for c in (25, 50, 100):
            d = "pattern" if (mn, c) == (10, 50) else f"grid_n{mn}_c{c}"
            rel = [r["designs"][d].get("eff_vs_pattern", 1.0) for r in rows]
            print(f"grid n{mn} c{c} rel {f2(med(rel))}", end=" "); a4("", d, "cover_z")
    # ---- A5 EO table
    print("\n% TABLE A5 EO: design | by L: median eff (RMSE^2 ratio) | mean P(best) | mean tau")
    for n in EO_DESIGNS:
        cells = []
        for L in BUDGETS:
            rr = byL(L)
            cells += [f2(med([r["eo"][n].get("eff_vs_srs", 1.0) for r in rr])), f2(np.mean([r["eo"][n]["p_best"] for r in rr])),
                      f2(np.nanmean([r["eo"][n]["tau"] for r in rr]))]
        print(n + " & " + " & ".join(cells) + " \\\\")
    print("EO bias mean srs/pattern:", np.mean([r["eo"]["srs"]["bias"] for r in rows]), np.mean([r["eo"]["pattern"]["bias"] for r in rows]))
    # ---- A3
    pts = []
    for r in rows:
        for c, v in r["subsets"].items():
            if "eff_vs_srs" in v["pattern"] and v["nonunan"] > 0:
                pts.append((r["dataset"], r["L"], v["M"], v["nonunan"], v["pattern"]["eff_vs_srs"], v["pattern"]["cover_z"], v["pattern"]["cover_z_min"]))
        pts.append((r["dataset"], r["L"], 9, r["nonunan"], r["designs"]["pattern"]["eff_vs_srs"], r["designs"]["pattern"]["cover_z"], r["designs"]["pattern"]["cover_z_min"]))
    pts = np.array(pts, dtype=object)
    print("\n% TABLE A3: M | n configs | median non-unanimous share | median eff by L | mean cov | mean worst")
    for M in (2, 3, 5, 9):
        p = pts[pts[:, 2] == M]
        e = [f2(med(p[p[:, 1] == L][:, 4].astype(float))) for L in BUDGETS]
        print(f"{M} & {len(p)} & {f3(med(p[:, 3].astype(float)))} & " + " & ".join(e) + f" & {f3(np.mean(p[:, 5].astype(float)))} & {f3(np.mean(p[:, 6].astype(float)))} \\\\")
    from scipy.stats import spearmanr
    for L in BUDGETS:
        p = pts[pts[:, 1] == L]
        print("spearman(log eff, nonunan) L", L, spearmanr(p[:, 3].astype(float), np.log(p[:, 4].astype(float))))
    # within-setting: does eff fall as nonunan rises?
    rs = []
    for key in SETTINGS:
        p = pts[(pts[:, 0] == key) & (pts[:, 1] == 250)]
        if len(p) > 3: rs.append(spearmanr(p[:, 3].astype(float), np.log(p[:, 4].astype(float))).statistic)
    print("within-setting spearman L=250 median", np.median(rs), rs)
    # ---- Figures
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    cols = {2: "#2a78d6", 3: "#eb6834", 5: "#1baf7a", 9: "#6250d6"}; mk = {2: "o", 3: "s", 5: "^", 9: "D"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, L in zip(axes, BUDGETS):
        for M in (2, 3, 5, 9):
            p = pts[(pts[:, 1] == L) & (pts[:, 2] == M)]
            ax.scatter(p[:, 3].astype(float), np.minimum(p[:, 4].astype(float), CLIP), s=16, c=cols[M], marker=mk[M],
                       edgecolors="white", linewidths=0.5, label=f"M = {M}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(1, color="#9a9890", lw=0.8, ls=":")
        ax.set_title(f"L = {L}"); ax.set_xlabel("share of pool with disagreement")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("efficiency vs SRS (median over targets)"); axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{figdir}/abl_models_disagreement.pdf"); fig.savefig(f"{figdir}/abl_models_disagreement.png", dpi=200)
    # coverage of worst target at L = 250
    rr = byL(250); x = np.arange(len(rr))
    meths = [("Wald", "cover_z_min", "#2a78d6", "o"), ("t (Satterthwaite)", "cover_t_min", "#eb6834", "s"),
             ("bootstrap", "cover_boot_min", "#1baf7a", "^"), ("Wald + proxy floor", "cover_floor_min", "#6250d6", "D")]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    for k, (lab, key, c, m) in enumerate(meths):
        ax.scatter(x + (k - 1.5) * 0.15, [r["designs"]["pattern"][key] for r in rr], s=22, c=c, marker=m, label=lab, edgecolors="white", linewidths=0.5)
    ax.axhline(0.95, color="#52514e", lw=0.8, ls="--"); ax.axhline(rows[0]["mc_min_ref"]["mean"], color="#9a9890", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[r["dataset"]] for r in rr], rotation=30, ha="right")
    ax.set_ylabel("worst-target coverage"); ax.set_ylim(0.3, 1.0); ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(f"{figdir}/abl_coverage_L250.pdf"); fig.savefig(f"{figdir}/abl_coverage_L250.png", dpi=200)
    # ---- appendix per-cell rows
    print("\n% APPENDIX eff vs SRS per cell (A1, A2)")
    A12 = ["pattern", "binary", "joint3", "kproxy2", "kproxy3", "single0", "single7", "const", "shuffle", "noise", "flip", "prop", "oracle"]
    for r in rows:
        print(f"{SHORT[r['dataset']]} & {r['L']} & " + " & ".join(f2(r['designs'][n]['eff_vs_srs']) if r['designs'][n]['eff_vs_srs'] < 1000 else '$>$1000' for n in A12) + " \\\\")
    print("\n% APPENDIX coverage per cell (default design): mean Wald | worst Wald | worst t | worst boot | worst floor | floor width | worst SRS Wald | worst SRS floor | H | worst target")
    for r in rows:
        d = r["designs"]["pattern"]; s_ = r["designs"]["srs"]
        print(f"{SHORT[r['dataset']]} & {r['L']} & {d['H']} & {f3(d['cover_z'])} & {f3(d['cover_z_min'])} & {f3(d['cover_t_min'])} & {f3(d['cover_boot_min'])} & {f3(d['cover_floor_min'])} & {f2(d['width_floor_over_z'])} & {f3(s_['cover_z_min'])} & {f3(s_['cover_floor_min'])} \\\\  % {d['worst_target']} {d['worst_target_truth']:.4f}")
    print("\n% APPENDIX grid per cell: worst Wald coverage for (5,25) (5,50) (5,100) (10,25) (10,50) (10,100) (20,25) (20,50) (20,100); rel eff")
    for r in rows:
        cells = []
        for mn in (5, 10, 20):
            for c in (25, 50, 100):
                d = "pattern" if (mn, c) == (10, 50) else f"grid_n{mn}_c{c}"; cells.append(r["designs"][d]["cover_z_min"])
        print(f"{SHORT[r['dataset']]} & {r['L']} & " + " & ".join(f"{x:.2f}" for x in cells) + " \\\\")
    print("\n% APPENDIX EO per cell: margin, n_best | P(best) srs group pattern pattern_gap oracle | tau srs pattern | eff pattern")
    for r in rows:
        e = r["eo"]
        print(f"{SHORT[r['dataset']]} & {r['L']} & {r['eo_margin']:.3f} & " + " & ".join(f"{e[n]['p_best']:.2f}" for n in EO_DESIGNS)
              + f" & {e['srs']['tau']:.2f} & {e['pattern']['tau']:.2f} & {f2(e['pattern']['eff_vs_srs'])} & {f2(e['pattern_gap']['eff_vs_srs'])} \\\\  % nbest {r['eo_n_best']}")
    print("\n% APPENDIX A3 per setting (L=250): nonunan and eff for each subset")
    for r in byL(250):
        print(SHORT[r["dataset"]], [(v["M"], round(v["nonunan"], 3), round(v["pattern"].get("eff_vs_srs", np.nan), 1)) for v in r["subsets"].values()])
    print("\n% misc: strata counts, top3")
    for r in rows:
        print(r["dataset"], r["L"], {n: (r["designs"][n]["H"], r["designs"][n]["min_n"]) for n in ("pattern", "binary", "joint3", "kproxy2", "kproxy3", "grid_n5_c25", "grid_n20_c100")}, r["top3"])


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(*sys.argv[2:]); sys.exit()
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
