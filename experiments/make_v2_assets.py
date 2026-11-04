"""Tables and figures for the main empirical section of manuscript_v2 (from results/label_efficiency.json)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
d = json.load(open("results/label_efficiency.json"))
LAB = {"adult-sex": "Adult (sex)", "adult-race": "Adult (race)", "bank": "Bank marketing", "credit-card": "Credit card",
       "law-school": "Law school", "magic": "MAGIC", "eeg": "EEG eye state", "phoneme": "Phoneme", "spambase": "Spambase",
       "churn": "Churn", "electricity": "Electricity"}
GROUP = {"adult-sex": "sex", "adult-race": "race", "bank": "age", "credit-card": "sex", "law-school": "race"}
keys = list(dict.fromkeys(r["dataset"] for r in d))
fmt = lambda x: (">1000" if x > 1000 else f"{x:.1f}")

# Table 1: data settings
L = ["\\begin{tabular}{lcrrrr}\\toprule", "Setting & Group & Pool $N$ & Positives $P_0$ & Positives $P_1$ & Disagreement \\\\\\midrule"]
for k in keys:
    r = next(x for x in d if x["dataset"] == k)
    L.append(f"{LAB[k]} & {GROUP.get(k, 'arbitrary')} & {r['N']} & {r['Pg'][0]} & {r['Pg'][1]} & {r['disagree_top']:.3f} \\\\")
L.append("\\bottomrule\\end{tabular}"); open("manuscript_v2/tables/data.tex", "w").write("\n".join(L))

# Table 2: efficiency per setting and budget
L = ["\\begin{tabular}{lrrrrrrrr}\\toprule",
     "Setting & $L$ & \\multicolumn{3}{c}{Efficiency vs SRS} & vs GROUP & \\multicolumn{3}{c}{PATTERN by estimand} \\\\",
     " & & GROUP & PATTERN & ORACLE & PATTERN & $\\theta^{TPR}$ & $\\theta^{FPR}$ & $\\theta^{ERR}$ \\\\\\midrule"]
for k in keys:
    for r in [x for x in d if x["dataset"] == k]:
        L.append(f"{LAB[k] if r['L'] == min(x['L'] for x in d if x['dataset'] == k) else ''} & {r['L']} & {fmt(r['eff_group_all'])} & {fmt(r['eff_pattern_all'])} & {fmt(r['eff_oracle_all'])} & {fmt(r['eff_pattern_vs_group_all'])} & {fmt(r['eff_pattern_dTPR'])} & {fmt(r['eff_pattern_dFPR'])} & {fmt(r['eff_pattern_dERR'])} \\\\")
    L.append("\\addlinespace")
L.append("\\bottomrule\\end{tabular}"); open("manuscript_v2/tables/efficiency.tex", "w").write("\n".join(L))

# Table 3: inference quality
L = ["\\begin{tabular}{rrrrrrrrr}\\toprule",
     "$L$ & Cells & \\multicolumn{2}{c}{Coverage, mean} & \\multicolumn{2}{c}{Coverage, worst target} & \\multicolumn{2}{c}{Correct sign} & SRS failures \\\\",
     " & & SRS & PATTERN & SRS & PATTERN & SRS & PATTERN & (max) \\\\\\midrule"]
summary = {}
for Lb in sorted(set(r["L"] for r in d)):
    R = [r for r in d if r["L"] == Lb]
    med = lambda k: float(np.median([r[k] for r in R if r[k] is not None]))
    summary[Lb] = dict(cells=len(R), eff_pattern=med("eff_pattern_all"), eff_group=med("eff_group_all"), eff_oracle=med("eff_oracle_all"),
                       eff_pg=med("eff_pattern_vs_group_all"), eff_top=med("eff_pattern_top"),
                       min_eff=float(min(r["eff_pattern_all"] for r in R)), max_eff=float(max(r["eff_pattern_all"] for r in R if r["dataset"] != "law-school")),
                       cov_srs=med("cover_srs"), cov_pat=med("cover_pattern"), worst_srs=float(min(r["cover_srs_min"] for r in R)),
                       worst_pat=float(min(r["cover_pattern_min"] for r in R)), sign_srs=med("sign_srs"), sign_pat=med("sign_pattern"),
                       fail=float(max(r["fail_srs"] for r in R)))
    s = summary[Lb]
    L.append(f"{Lb} & {s['cells']} & {s['cov_srs']:.3f} & {s['cov_pat']:.3f} & {s['worst_srs']:.2f} & {s['worst_pat']:.2f} & {s['sign_srs']:.2f} & {s['sign_pat']:.2f} & {s['fail']:.3f} \\\\")
L.append("\\bottomrule\\end{tabular}"); open("manuscript_v2/tables/inference.tex", "w").write("\n".join(L))
json.dump(summary, open("results/label_efficiency_summary.json", "w"), indent=1)

# Figure: efficiency vs budget, and vs disagreement rate
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
for k in keys:
    R = [r for r in d if r["dataset"] == k and k != "law-school"]
    if R: ax[0].plot([r["L"] for r in R], [r["eff_pattern_all"] for r in R], marker="o", ms=3, lw=1, label=LAB[k])
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].axhline(1, color="k", lw=.6, ls=":")
ax[0].set_xlabel("Label budget $L$"); ax[0].set_ylabel("Efficiency vs SRS (median over estimands)")
ax[0].legend(fontsize=6, frameon=False, ncol=2)
xs = [r["disagree_top"] for r in d if r["L"] == 500 and r["dataset"] != "law-school"]
ys = [r["eff_pattern_all"] for r in d if r["L"] == 500 and r["dataset"] != "law-school"]
ax[1].scatter(xs, ys, s=14, color="#1f4e79")
for r in d:
    if r["L"] == 500 and r["dataset"] != "law-school": ax[1].annotate(LAB[r["dataset"]], (r["disagree_top"], r["eff_pattern_all"]), fontsize=6)
ax[1].set_xlabel("Disagreement rate among the three best models"); ax[1].set_ylabel("Efficiency vs SRS at $L=500$"); ax[1].set_yscale("log")
fig.tight_layout(); fig.savefig("manuscript_v2/figures/efficiency.pdf"); fig.savefig("manuscript_v2/figures/efficiency.png", dpi=200)
print(json.dumps(summary, indent=0))
