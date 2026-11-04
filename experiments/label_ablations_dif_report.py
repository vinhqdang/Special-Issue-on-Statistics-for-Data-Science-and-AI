"""Tables (manuscript_v2/tables/abl_dif_*.tex), figures (manuscript_v2/figures/abl_dif_*) and printed summary
numbers for results/label_ablations_dif.json.  Usage: python experiments/label_ablations_dif.py report"""
import json, os
import numpy as np
from scipy.stats import spearmanr

BUDGETS = (250, 500, 1000)
SHORT = {"adult-sex": "Adult-sex", "adult-race": "Adult-race", "bank": "Bank", "credit-card": "Credit", "law-school": "Law",
         "magic": "MAGIC", "eeg": "EEG", "phoneme": "Phoneme", "spambase": "Spambase", "churn": "Churn", "electricity": "Elec."}
SMALL = ["churn", "spambase", "phoneme", "law-school"]
TAB = "manuscript_v2/tables"; FIG = "manuscript_v2/figures"


def f2(x, big=1000):
    if x is None or not np.isfinite(x): return "--"
    return f"$>${big}" if x >= big else f"{x:.2f}"


def f3(x): return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


def f1(x): return "--" if x is None or not np.isfinite(x) else f"{x:.1f}"


def _e(d):
    x = d.get("eff_vs_srs", 1.0)
    return np.nan if x is None else x


def put(name, body):
    os.makedirs(TAB, exist_ok=True); open(f"{TAB}/{name}.tex", "w").write(body)


def report(path="results/label_ablations_dif.json"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
    rows = json.load(open(path))
    byL = lambda L: [r for r in rows if r["L"] == L]
    def vals(lab, key, rr=rows):
        return [r["ent"][lab][key] for r in rr if lab in r["ent"] and r["ent"][lab].get(key) is not None]
    med = lambda v: float(np.median(v)) if len(v) else np.nan
    mean = lambda v: float(np.mean(v)) if len(v) else np.nan
    print("cells", len(rows), "R", rows[0]["R"], "B", rows[0]["B"], "mc_min_ref", [r["mc_min_ref"]["mean"] for r in rows][:3])
    print("max fail", max(v["fail"] for r in rows for v in r["ent"].values()))
    # ------------------------------------------------------------ T1: design (A1)
    A1 = [("PATTERN-DIF (default)", "pattern|dif"), None,
          ("A1: stratifier", None),
          ("Group $\\times$ agree/disagree", "binary|dif"), ("Group $\\times$ pattern of 3 models", "joint3|dif"),
          ("Group $\\times K\\times$ 2 proxy bins", "kp2|dif"), ("Group $\\times K\\times$ 3 proxy bins", "kp3|dif"),
          ("PATTERN, residual allocation", "pattern_resid|dif"), None,
          ("References", None),
          ("GROUP-DIF (strata = group)", "group|dif"), ("SRS-DIF (PPI, $\\lambda=1$)", "srs|dif"),
          ("Proportional allocation (no proxy)", "prop|dif"), ("Gap-targeted allocation (A5)", "pattern_gap|dif"),
          ("PATTERN, plug-in (HT)", "pattern|ht")]
    out = ["\\begin{tabular}{lccccccc}", "\\toprule",
           " & \\multicolumn{3}{c}{Efficiency vs SRS} & & & \\multicolumn{2}{c}{Coverage} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){7-8}",
           "Variant (estimator DIF unless stated) & $L=250$ & 500 & 1000 & Rel. & Win & mean & worst \\\\", "\\midrule"]
    print("\n== A1/A1-refs: eff by L | rel | win | cov | worst mean | worst min")
    for it in A1:
        if it is None: out.append("\\addlinespace"); continue
        nm, lab = it
        if lab is None: out.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{nm}}}}}\\\\"); continue
        e = [med(vals(lab, "eff_vs_srs", byL(L))) for L in BUDGETS]
        rel = vals(lab, "eff_vs_def") if lab != "pattern|dif" else []
        win = np.mean(np.array(rel) > 1) if rel else None
        out.append(f"{nm} & " + " & ".join(f2(x) for x in e) + f" & {f2(med(rel)) if rel else '1.00'} & {f2(win) if rel else '--'} & "
                   f"{f3(mean(vals(lab, 'cover')))} & {f3(mean(vals(lab, 'cover_min')))} \\\\")
        print(lab, [round(x, 2) for x in e], round(med(rel), 3) if rel else 1, round(win, 2) if rel else None,
              round(mean(vals(lab, 'cover')), 3), round(mean(vals(lab, 'cover_min')), 3), round(min(vals(lab, 'cover_min')), 3))
    out += ["\\bottomrule", "\\end{tabular}"]
    put("abl_dif_design", "\n".join(out) + "\n")
    # ------------------------------------------------------------ T2: proxy (A2)
    P = [("Ensemble mean $\\pi$ (default)", "pattern"), ("Single model (logistic)", "single0"), ("Single model (naive Bayes)", "single7"),
         ("Constant 0.5", "const"), ("$\\pi$ shuffled within group", "shuffle"), ("$\\pi$ with logit noise (sd 2)", "noise"),
         ("Reversed, $1-\\pi$", "flip"), ("True labels (allocation only)", "oracle")]
    out = ["\\begin{tabular}{lcccccccccc}", "\\toprule",
           " & \\multicolumn{3}{c}{Proxy in allocation only (HT)} & \\multicolumn{4}{c}{Proxy in both (DIF)} & \\multicolumn{2}{c}{DIF, ensemble in estimator} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}",
           "Proxy & Eff. & Rel. & worst & Eff. & Rel. & mean & worst & Eff. & Rel. \\\\", "\\midrule"]
    print("\n== A2: HT(eff,rel,worst) | DIF both(eff,rel,mean,worst) | DIF alloc-only(eff,rel)")
    for nm, d in P:
        ht = d + "|ht"; both = d + "|dif" if d not in ("pattern", "oracle") else ("pattern|dif" if d == "pattern" else None)
        ao = d + "|difens" if d not in ("pattern", "oracle") else ("pattern|dif" if d == "pattern" else "oracle|difens")
        cells = []
        # HT: eff vs SRS, rel to default DIF, worst
        cells += [f2(med(vals(ht, "eff_vs_srs"))), f2(med(vals(ht, "eff_vs_def"))), f3(mean(vals(ht, "cover_min")))]
        if both:
            cells += [f2(med(vals(both, "eff_vs_srs"))), f2(med(vals(both, "eff_vs_def")) if both != "pattern|dif" else 1.0),
                      f3(mean(vals(both, "cover"))), f3(mean(vals(both, "cover_min")))]
        else: cells += ["--"] * 4
        cells += [f2(med(vals(ao, "eff_vs_srs"))), f2(med(vals(ao, "eff_vs_def")) if ao != "pattern|dif" else 1.0)]
        out.append(f"{nm} & " + " & ".join(cells) + " \\\\")
        print(d, cells)
    out.append("\\addlinespace")
    pr = ("Proportional allocation (no proxy in allocation)", "prop")
    cells = [f2(med(vals("prop|ht", "eff_vs_srs"))), f2(med(vals("prop|ht", "eff_vs_def"))), f3(mean(vals("prop|ht", "cover_min"))),
             f2(med(vals("prop|dif", "eff_vs_srs"))), f2(med(vals("prop|dif", "eff_vs_def"))), f3(mean(vals("prop|dif", "cover"))),
             f3(mean(vals("prop|dif", "cover_min"))), "--", "--"]
    out.append(f"{pr[0]} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    put("abl_dif_proxy", "\n".join(out) + "\n")
    # per-proxy win shares and cells (text numbers)
    for d in ["single0", "single7", "const", "shuffle", "noise", "flip"]:
        for lab in (d + "|dif", d + "|ht", d + "|difens", d + "|tuned"):
            rel = vals(lab, "eff_vs_def")
            print(lab, "rel med", round(med(rel), 3), "win", round(np.mean(np.array(rel) > 1), 2), "cells<0.5", int(np.sum(np.array(rel) < 0.5)),
                  "cov", round(mean(vals(lab, 'cover')), 3), "worst mean", round(mean(vals(lab, 'cover_min')), 3),
                  "min", round(min(vals(lab, 'cover_min')), 3))
    # ------------------------------------------------------------ T3: lambda
    LR = [("Ensemble $\\pi$, PATTERN", "pattern", "pattern", "pattern"), ("Ensemble $\\pi$, SRS", "srs", "srs", "srs"),
          ("Constant 0.5, PATTERN", "const", "const", "const"), ("Shuffled $\\pi$, PATTERN", "shuffle", "shuffle", "shuffle"),
          ("Noisy $\\pi$, PATTERN", "noise", "noise", "noise"), ("Reversed $1-\\pi$, PATTERN", "flip", "flip", "flip")]
    out = ["\\begin{tabular}{lccccccccc}", "\\toprule",
           " & \\multicolumn{3}{c}{$\\lambda=0$ (HT)} & \\multicolumn{3}{c}{$\\lambda=1$ (DIF)} & \\multicolumn{3}{c}{$\\lambda$ tuned} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
           "Proxy and design & Eff. & mean & worst & Eff. & mean & worst & Eff. & mean & worst \\\\", "\\midrule"]
    print("\n== lambda: HT | DIF | tuned : eff, cov, worst-mean, worst-min")
    for nm, d, _, _ in LR:
        cells = []
        for suf in ("ht", "dif", "tuned"):
            lab = f"{d}|{suf}"
            if d == "srs" and suf == "ht": cells += ["1.00", f3(mean(vals(lab, "cover"))), f3(mean(vals(lab, "cover_min")))]
            else: cells += [f2(med(vals(lab, "eff_vs_srs"))), f3(mean(vals(lab, "cover"))), f3(mean(vals(lab, "cover_min")))]
        out.append(f"{nm} & " + " & ".join(cells) + " \\\\")
        print(d, cells, "min worst", [round(min(vals(f"{d}|{s}", 'cover_min')), 3) for s in ("ht", "dif", "tuned")])
    out += ["\\bottomrule", "\\end{tabular}"]
    put("abl_dif_lambda", "\n".join(out) + "\n")
    for L in BUDGETS:
        print("lambda by L", L, {s: round(med(vals(f"pattern|{s}", "eff_vs_srs", byL(L))), 2) for s in ("ht", "dif", "tuned")},
              "tuned vs dif rel med", round(med(vals("pattern|tuned", "eff_vs_def", byL(L))), 3))
    rel = vals("pattern|tuned", "eff_vs_def"); print("tuned vs default: median", med(rel), "win", np.mean(np.array(rel) > 1), "cells", len(rel))
    print("pattern|dif eff by dTPR/dFPR/dERR", [round(med(vals("pattern|dif", f"eff_vs_srs_{k}")), 2) for k in ("dTPR", "dFPR", "dERR")],
          "ht", [round(med(vals("pattern|ht", f"eff_vs_srs_{k}")), 2) for k in ("dTPR", "dFPR", "dERR")])
    # share of cells where DIF beats HT on pattern design, per setting
    ratio = [(r["dataset"], r["L"], r["ent"]["pattern|dif"]["eff_vs_srs"] / r["ent"]["pattern|ht"]["eff_vs_srs"]) for r in rows]
    print("DIF/HT ratio on pattern: median", med([x[2] for x in ratio]), "min", min(ratio, key=lambda x: x[2]), "max", max(ratio, key=lambda x: x[2]))
    print("cells where ht >= dif:", [(a, b, round(c, 2)) for a, b, c in ratio if c < 1])
    # ------------------------------------------------------------ T4: models (A3)
    pts = []
    for r in rows:
        for c, v in r["subsets"].items():
            if v["n_live"] > 0 and "eff_vs_srs" in v["ent"].get("pattern|dif", {}):
                e = v["ent"]
                pts.append((r["dataset"], r["L"], v["M"], v["nonunan"], e["pattern|dif"]["eff_vs_srs"], e["pattern|ht"]["eff_vs_srs"],
                            e["pattern|dif"]["cover"], e["pattern|dif"]["cover_min"], e["pattern|dif"].get("eff_vs_def", np.nan),
                            e["srs|dif"]["eff_vs_srs"] if "eff_vs_srs" in e.get("srs|dif", {}) else np.nan))
        e = r["ent"]
        pts.append((r["dataset"], r["L"], 9, r["nonunan"], e["pattern|dif"]["eff_vs_srs"], e["pattern|ht"]["eff_vs_srs"],
                    e["pattern|dif"]["cover"], e["pattern|dif"]["cover_min"], np.nan, e["srs|dif"]["eff_vs_srs"]))
    pts = np.array(pts, dtype=object)
    out = ["\\begin{tabular}{rccccccccc}", "\\toprule",
           " & & & \\multicolumn{3}{c}{PATTERN-DIF vs SRS} & \\multicolumn{1}{c}{PATTERN-HT vs SRS} & & \\multicolumn{2}{c}{Coverage (DIF)} \\\\",
           "\\cmidrule(lr){4-6}\\cmidrule(lr){9-10}",
           "$M$ & Configs & Disagr. & $L=250$ & 500 & 1000 & all $L$ & DIF/HT & mean & worst \\\\", "\\midrule"]
    print("\n== A3")
    for M in (2, 3, 5, 9):
        p = pts[pts[:, 2] == M]
        e = [med(p[p[:, 1] == L][:, 4].astype(float)) for L in BUDGETS]
        ht = med(p[:, 5].astype(float)); rr = med(p[:, 4].astype(float) / p[:, 5].astype(float))
        out.append(f"{M} & {len(p)} & {f3(med(p[:, 3].astype(float)))} & " + " & ".join(f2(x) for x in e) +
                   f" & {f2(ht)} & {f2(rr)} & {f3(mean(p[:, 6].astype(float)))} & {f3(mean(p[:, 7].astype(float)))} \\\\")
        print(M, len(p), "nonunan", round(med(p[:, 3].astype(float)), 3), [round(x, 2) for x in e], "ht", round(ht, 2), "dif/ht", round(rr, 2),
              "cov", round(mean(p[:, 6].astype(float)), 3), round(mean(p[:, 7].astype(float)), 3))
    out += ["\\bottomrule", "\\end{tabular}"]
    put("abl_dif_models", "\n".join(out) + "\n")
    for L in BUDGETS:
        p = pts[pts[:, 1] == L]
        print("spearman(log eff DIF, nonunan) L", L, spearmanr(p[:, 3].astype(float), np.log(p[:, 4].astype(float))),
              "HT:", spearmanr(p[:, 3].astype(float), np.log(p[:, 5].astype(float))).statistic,
              "log DIF/HT vs nonunan:", spearmanr(p[:, 3].astype(float), np.log(p[:, 4].astype(float) / p[:, 5].astype(float))).statistic)
    rs = []
    for key in SHORT:
        p = pts[(pts[:, 0] == key) & (pts[:, 1] == 250)]
        if len(p) > 3: rs.append(spearmanr(p[:, 3].astype(float), np.log(p[:, 4].astype(float))).statistic)
    print("within-setting spearman L=250 median", np.median(rs), "n neg", int(np.sum(np.array(rs) < 0)), "of", len(rs))
    # figure: A3
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    cols = {2: "#2a78d6", 3: "#eb6834", 5: "#1baf7a", 9: "#6250d6"}; mk = {2: "o", 3: "s", 5: "^", 9: "D"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, L in zip(axes, BUDGETS):
        for M in (2, 3, 5, 9):
            p = pts[(pts[:, 1] == L) & (pts[:, 2] == M)]
            ax.scatter(p[:, 3].astype(float), np.minimum(p[:, 4].astype(float), 1e4), s=16, c=cols[M], marker=mk[M], edgecolors="white",
                       linewidths=0.5, label=f"M = {M}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(1, color="#9a9890", lw=0.8, ls=":"); ax.axhline(1e4, color="#9a9890", lw=0.6, ls="--")
        ax.set_xticks([0.05, 0.1, 0.2, 0.4, 0.8]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4", "0.8"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"L = {L}"); ax.set_xlabel("share of pool with disagreement"); ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("efficiency of PATTERN-DIF vs SRS")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{FIG}/abl_dif_models_disagreement.pdf"); fig.savefig(f"{FIG}/abl_dif_models_disagreement.png", dpi=200)
    plt.close(fig)
    # ------------------------------------------------------------ T5: inference (A4)
    meths = [("Wald (default)", "pattern|dif", "wald"), ("Wald + floor 'zero'", "pattern|dif", "zero"), ("Wald + floor 'max'", "pattern|dif", "max"),
             ("$t$, Satterthwaite df", "pattern|dif", "t"), ("$t$ + floor 'zero'", "pattern|dif", "t_zero"),
             ("Stratified bootstrap", "pattern|dif", "boot"), None,
             ("\\multicolumn{9}{l}{\\emph{References}}", None, None),
             ("SRS-DIF (PPI), Wald", "srs|dif", "wald"), ("SRS-DIF, Wald + floor 'max'", "srs|dif", "max"), ("SRS-DIF, bootstrap", "srs|dif", "boot"),
             ("PATTERN plug-in (HT), Wald", "pattern|ht", None)]
    sels = [lambda r: r["L"] == 250, lambda r: r["L"] == 250 and r["dataset"] in SMALL, lambda r: r["L"] >= 500]
    out = ["\\begin{tabular}{lcccccccc}", "\\toprule",
           " & \\multicolumn{3}{c}{$L=250$, all} & \\multicolumn{2}{c}{$L=250$, small} & \\multicolumn{2}{c}{$L\\ge 500$} & \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}", "Method & mean & worst & min & worst & min & worst & min & Width \\\\", "\\midrule"]
    print("\n== A4")
    for it in meths:
        if it is None: out.append("\\addlinespace"); continue
        nm, lab, k = it
        if lab is None: out.append(nm + " \\\\"); continue
        cells = []
        def get(r, key):
            e = r["ent"][lab]
            if k is None: return e["cover" if key == "cover" else "cover_min"]
            return e["inf"][k][key]
        for si, sel in enumerate(sels):
            rr = [r for r in rows if sel(r)]
            wm = mean([get(r, "cover_min") for r in rr]); mn = min(get(r, "cover_min") for r in rr)
            if si == 0: cells += [f3(mean([get(r, "cover") for r in rr])), f3(wm), f3(mn)]
            else: cells += [f3(wm), f3(mn)]
        w = f2(mean([r["ent"][lab]["inf"][k]["width"] for r in rows])) if k else "--"
        out.append(f"{nm} & " + " & ".join(cells) + f" & {w} \\\\")
        print(lab, k, cells, w)
    out += ["\\bottomrule", "\\end{tabular}"]
    put("abl_dif_inference", "\n".join(out) + "\n")
    # per-setting worst coverage at L=250, Wald, small pools
    for r in byL(250):
        e = r["ent"]
        print("L250", SHORT[r["dataset"]], "DIF wald", f3(e["pattern|dif"]["inf"]["wald"]["cover_min"]), "zero", f3(e["pattern|dif"]["inf"]["zero"]["cover_min"]),
              "max", f3(e["pattern|dif"]["inf"]["max"]["cover_min"]), "t", f3(e["pattern|dif"]["inf"]["t"]["cover_min"]),
              "boot", f3(e["pattern|dif"]["inf"]["boot"]["cover_min"]), "HT", f3(e["pattern|ht"]["cover_min"]), "deg", f3(e["pattern|dif"]["degenerate"]),
              "meanwald", f3(e["pattern|dif"]["cover"]), "worst target", e["pattern|dif"]["worst_target"])
    allc = [(r["dataset"], r["L"], r["ent"]["pattern|dif"]["inf"]["wald"]["cover_min"]) for r in rows]
    print("lowest 6 wald worst-cov:", sorted(allc, key=lambda x: x[2])[:6])
    print("floor zero ever differs from wald:", [(r["dataset"], r["L"], round(r["ent"]["pattern|dif"]["inf"]["zero"]["width"], 4)) for r in rows
                                                  if r["ent"]["pattern|dif"]["inf"]["zero"]["width"] > 1.0005])
    print("share degenerate se==0 (dif, pattern) max:", max(r["ent"]["pattern|dif"]["degenerate"] for r in rows),
          " HT max:", max(r["ent"]["pattern|ht"]["degenerate"] for r in rows))
    # figure A4
    rr = byL(250); x = np.arange(len(rr))
    ms = [("DIF Wald", lambda e: e["pattern|dif"]["inf"]["wald"]["cover_min"], "#2a78d6", "o"),
          ("DIF $t$", lambda e: e["pattern|dif"]["inf"]["t"]["cover_min"], "#eb6834", "s"),
          ("DIF bootstrap", lambda e: e["pattern|dif"]["inf"]["boot"]["cover_min"], "#1baf7a", "^"),
          ("DIF Wald + floor max", lambda e: e["pattern|dif"]["inf"]["max"]["cover_min"], "#6250d6", "D"),
          ("plug-in (HT) Wald", lambda e: e["pattern|ht"]["cover_min"], "#9a9890", "x")]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for k, (lab, fn, c, m) in enumerate(ms):
        kw = dict(s=22, c=c, marker=m, label=lab) if m == "x" else dict(s=22, c=c, marker=m, label=lab, edgecolors="white", linewidths=0.5)
        ax.scatter(x + (k - 2) * 0.13, [fn(r["ent"]) for r in rr], **kw)
    ax.axhline(0.95, color="#52514e", lw=0.8, ls="--"); ax.axhline(np.mean([r["mc_min_ref"]["mean"] for r in rr]), color="#9a9890", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[r["dataset"]] for r in rr], rotation=30, ha="right")
    ax.set_ylabel("worst-target coverage"); ax.set_ylim(0.3, 1.0); ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(f"{FIG}/abl_dif_coverage_L250.pdf"); fig.savefig(f"{FIG}/abl_dif_coverage_L250.png", dpi=200); plt.close(fig)
    # ------------------------------------------------------------ T6: EO (A5)
    EO = [("SRS (plug-in)", "srs|ht"), ("SRS-DIF (PPI)", "srs|dif"), ("GROUP, plug-in", "group|ht"), ("PATTERN, plug-in (HT)", "pattern|ht"),
          ("PATTERN-DIF (default)", "pattern|dif"), ("Gap-targeted, plug-in", "pattern_gap|ht"), ("Gap-targeted, DIF", "pattern_gap|dif"),
          ("ORACLE-ALLOC, DIF", "oracle|difens")]
    nanmean = lambda v: float(np.nanmean(v))
    out = ["\\begin{tabular}{lccccccccc}", "\\toprule",
           " & \\multicolumn{3}{c}{$L=250$} & \\multicolumn{3}{c}{$L=500$} & \\multicolumn{3}{c}{$L=1000$} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}", "Design & Eff. & $P$ & $\\tau$ & Eff. & $P$ & $\\tau$ & Eff. & $P$ & $\\tau$ \\\\", "\\midrule"]
    print("\n== A5")
    for nm, lab in EO:
        cells = []
        for L in BUDGETS:
            rr = byL(L)
            cells += [f2(float(np.nanmedian([_e(r["eo"][lab]) for r in rr]))), f2(mean([r["eo"][lab]["p_best"] for r in rr])),
                      f2(nanmean([r["eo"][lab]["tau"] for r in rr]))]
        out.append(f"{nm} & " + " & ".join(cells) + " \\\\")
        wins = np.nanmean([_e(r["eo"][lab]) > 1 for r in rows if np.isfinite(_e(r["eo"][lab]))])
        print(lab, cells, "bias", round(mean([r["eo"][lab]["bias"] for r in rows]), 4), "share cells eff>1", round(wins, 2),
              "p_best excl law/eeg L250", round(mean([r["eo"][lab]["p_best"] for r in byL(250) if r["dataset"] not in ("law-school", "eeg")]), 2),
              "cells eff<1:", [(r["dataset"], r["L"], round(_e(r["eo"][lab]), 2)) for r in rows if lab != "srs|ht" and _e(r["eo"][lab]) < 1][:12])
    out += ["\\bottomrule", "\\end{tabular}"]
    put("abl_dif_eo", "\n".join(out) + "\n")
    # ------------------------------------------------------------ figure A2: relative efficiency HT-only vs both
    names = [("single0", "logistic"), ("single7", "naive Bayes"), ("const", "constant"), ("shuffle", "shuffled"), ("noise", "noisy"), ("flip", "reversed")]
    fig, ax = plt.subplots(figsize=(7.2, 2.7)); xs = np.arange(len(names))
    for off, suf, c, lab, m in ((-0.2, "ht", "#9a9890", "proxy in allocation only (HT)", "o"), (0.0, "dif", "#2a78d6", "proxy in both (DIF)", "s"),
                                (0.2, "difens", "#eb6834", "alternative in allocation, ensemble in estimator", "^")):
        rel = [vals(f"{d}|{suf}", "eff_vs_def") for d, _ in names]
        ax.boxplot(rel, positions=xs + off, widths=0.17, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=c, alpha=0.5, lw=0.6), medianprops=dict(color="black", lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=c, alpha=0.5, edgecolor="black", lw=0.6, label=lab))
    ax.axhline(1, color="#52514e", lw=0.8, ls=":"); ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels([n for _, n in names])
    ax.set_ylabel("efficiency relative to PATTERN-DIF"); ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(f"{FIG}/abl_dif_proxy.pdf"); fig.savefig(f"{FIG}/abl_dif_proxy.png", dpi=200); plt.close(fig)
    # ------------------------------------------------------------ appendix per-cell tables
    def cell_table(name, cols, key, head, small=False):
        out = ["\\begin{tabular}{lr" + "r" * len(cols) + "}", "\\toprule", "Setting & $L$ & " + " & ".join(h for h, _ in cols) + " \\\\", "\\midrule"]
        for r in rows:
            out.append(f"{SHORT[r['dataset']]} & {r['L']} & " + " & ".join(f2(r["ent"][l].get(key)) if l in r["ent"] else "--" for _, l in cols) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]; put(name, "\n".join(out) + "\n")
    cell_table("abl_dif_app_eff_design", [("Pat.", "pattern|dif"), ("Pat.\\ HT", "pattern|ht"), ("Grp", "group|dif"), ("SRS", "srs|dif"), ("Bin.", "binary|dif"),
                                          ("J3", "joint3|dif"), ("KP2", "kp2|dif"), ("KP3", "kp3|dif"), ("Res.", "pattern_resid|dif"), ("Prop.", "prop|dif")],
               "eff_vs_srs", "")
    cell_table("abl_dif_app_eff_proxy", [("Pat.", "pattern|dif"), ("LR", "single0|dif"), ("NB", "single7|dif"), ("C", "const|dif"), ("Sh.", "shuffle|dif"),
                                         ("Noi.", "noise|dif"), ("Rev.", "flip|dif"), ("Orc.", "oracle|difens")], "eff_vs_srs", "")
    cell_table("abl_dif_app_eff_proxy_ht", [("Pat.", "pattern|ht"), ("LR", "single0|ht"), ("NB", "single7|ht"), ("C", "const|ht"), ("Sh.", "shuffle|ht"),
                                            ("Noi.", "noise|ht"), ("Rev.", "flip|ht"), ("Orc.", "oracle|ht")], "eff_vs_srs", "")
    cell_table("abl_dif_app_eff_tuned", [("$\\lambda=1$", "pattern|dif"), ("tuned", "pattern|tuned"), ("C tuned", "const|tuned"), ("Sh. tuned", "shuffle|tuned"),
                                         ("Rev. tuned", "flip|tuned"), ("SRS $\\lambda$=1", "srs|dif"), ("SRS tuned", "srs|tuned")], "eff_vs_srs", "")
    # coverage per cell
    out = ["\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
           " & & & & \\multicolumn{5}{c}{Worst target, PATTERN-DIF} & \\multicolumn{2}{c}{Worst, other} \\\\", "\\cmidrule(lr){5-9}\\cmidrule(lr){10-11}",
           "Setting & $L$ & $H$ & Mean & Wald & Zero & Max & $t$ & Boot. & HT & SRS-DIF \\\\", "\\midrule"]
    for r in rows:
        e = r["ent"]; d = e["pattern|dif"]; i = d["inf"]
        out.append(f"{SHORT[r['dataset']]} & {r['L']} & {r['design']['pattern']['H']} & {f3(d['cover'])} & {f3(i['wald']['cover_min'])} & {f3(i['zero']['cover_min'])} & "
                   f"{f3(i['max']['cover_min'])} & {f3(i['t']['cover_min'])} & {f3(i['boot']['cover_min'])} & {f3(e['pattern|ht']['cover_min'])} & {f3(e['srs|dif']['inf']['wald']['cover_min'])} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]; put("abl_dif_app_cov", "\n".join(out) + "\n")
    # width ratios per cell
    out = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Setting & $L$ & Zero & Max & $t$ & Boot. \\\\", "\\midrule"]
    for r in rows:
        i = r["ent"]["pattern|dif"]["inf"]
        out.append(f"{SHORT[r['dataset']]} & {r['L']} & {f3(i['zero']['width'])} & {f2(i['max']['width'])} & {f2(i['t']['width'])} & {f2(i['boot']['width'])} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]; put("abl_dif_app_width", "\n".join(out) + "\n")
    # EO per cell
    out = ["\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
           " & & & & \\multicolumn{4}{c}{$P(\\mathrm{best})$} & \\multicolumn{2}{c}{Kendall $\\tau$} & Eff. \\\\", "\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}",
           "Setting & $L$ & Margin & Ties & SRS & HT & DIF & Gap-DIF & SRS & DIF & DIF \\\\", "\\midrule"]
    for r in rows:
        e = r["eo"]
        out.append(f"{SHORT[r['dataset']]} & {r['L']} & {r['eo_margin']:.3f} & {r['eo_n_best']} & {e['srs|ht']['p_best']:.2f} & {e['pattern|ht']['p_best']:.2f} & "
                   f"{e['pattern|dif']['p_best']:.2f} & {e['pattern_gap|dif']['p_best']:.2f} & {e['srs|ht']['tau']:.2f} & {e['pattern|dif']['tau']:.2f} & "
                   f"{f2(_e(e['pattern|dif']))} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]; put("abl_dif_app_eo", "\n".join(out) + "\n")
    # A3 per setting at L=250
    out = ["\\begin{tabular}{lcccc}", "\\toprule", "Setting & $M=2$ & $M=3$ & $M=5$ & $M=9$ \\\\", "\\midrule"]
    for r in byL(250):
        cells = []
        for M in (2, 3, 5):
            cs = [f"{v['nonunan']:.2f}/{f1(v['ent']['pattern|dif'].get('eff_vs_srs'))}" if v["n_live"] > 0 else "exact"
                  for v in r["subsets"].values() if v["M"] == M]
            cells.append(", ".join(cs))
        cells.append(f"{r['nonunan']:.2f}/{f1(r['ent']['pattern|dif']['eff_vs_srs'])}")
        out.append(f"{SHORT[r['dataset']]} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]; put("abl_dif_app_models", "\n".join(out) + "\n")
    # misc
    print("\nstrata H range", min(r["design"]["pattern"]["H"] for r in rows), max(r["design"]["pattern"]["H"] for r in rows))
    print("floor used (kp2, kp3) min:", min(r["design"]["kp2"]["min_n"] for r in rows), min(r["design"]["kp3"]["min_n"] for r in rows))
