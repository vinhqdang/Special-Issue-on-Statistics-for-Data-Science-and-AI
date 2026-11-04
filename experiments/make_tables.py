import json, numpy as np
sim = json.load(open("results/simulation.json")); cs = json.load(open("results/casestudy.json"))
def yn(b): return "yes" if b else "no"
L = ["\\begin{tabular}{rrrrrrrr}\\toprule",
     "$n$ & $\\delta$ & $\\rho$ & \\multicolumn{2}{c}{Retention} & \\multicolumn{2}{c}{Exclusion} & TCS size \\\\",
     " & & & TCS & Naive & TCS & Naive & \\\\\\midrule"]
for r in sim:
    L.append(f"{r['n']} & {r['delta']:.2f} & {r['rho']:.1f} & {r['tcs_cover']:.3f} & {r['naive_cover']:.3f} & {r['tcs_excl']:.2f} & {r['naive_excl']:.2f} & {r['tcs_size']:.2f} \\\\")
L.append("\\bottomrule\\end{tabular}")
open("manuscript/tables/sim.tex", "w").write("\n".join(L))
for o in cs:
    L = ["\\begin{tabular}{lrrrrrcccr}\\toprule",
         "Model & Error & EO gap & ECE & Noise drop & $S_m$ & TCS & Naive & $\\hat\\pi_m$ & Top-1 \\\\\\midrule"]
    for i, m in enumerate(o["models"]):
        p = o["point"][i]
        L.append(f"{m.replace(chr(95), chr(92)+chr(95))} & {p[0]:.3f} & {p[1]:.3f} & {p[2]:.3f} & {p[3]:.3f} & {o['stat'][i]:.2f} & {yn(o['tcs'][i])} & {yn(o['pareto'][i])} & {o['pareto_prob'][i]:.2f} & {o['top1_freq'][i]:.2f} \\\\")
    L.append("\\bottomrule\\end{tabular}")
    open(f"manuscript/tables/case_{o['dataset']}.tex", "w").write("\n".join(L))
L = ["\\begin{tabular}{lrrrr}\\toprule", "Criteria subset & \\multicolumn{2}{c}{Adult} & \\multicolumn{2}{c}{German credit} \\\\",
     " & TCS & Naive & TCS & Naive \\\\\\midrule"]
ab = {"error": "err", "eo_gap": "EO", "ece": "ECE", "noise_drop": "noise"}
for a, b in zip(cs[0]["subsets"], cs[1]["subsets"]):
    L.append(f"{' + '.join(ab[c] for c in a['criteria'])} & {a['tcs_size']} & {a['pareto_size']} & {b['tcs_size']} & {b['pareto_size']} \\\\")
L.append("\\bottomrule\\end{tabular}")
open("manuscript/tables/subsets.tex", "w").write("\n".join(L))
for o in cs: print(o["dataset"], o["crit"], o["weight_win"], o["rank_lo"], o["rank_hi"])
print(max(r["tcs_size"] for r in sim), min(r["tcs_cover"] for r in sim), min(r["naive_cover"] for r in sim), max(r["naive_cover"] for r in sim))

# head-to-head with the marginal-box procedure
b5 = json.load(open("results/simulation_box.json")); b0 = json.load(open("results/simulation_box_mc0.0.json"))
L = ["\\begin{tabular}{rrrrrrrrr}\\toprule",
     "$n$ & $\\delta$ & $\\rho$ & \\multicolumn{2}{c}{Retention} & \\multicolumn{2}{c}{Exclusion, $r=0.5$} & \\multicolumn{2}{c}{Exclusion, $r=0$} \\\\",
     " & & & TCS & Box & TCS & Box & TCS & Box \\\\\\midrule"]
for a, c in zip(b5, b0):
    L.append(f"{a['n']} & {a['delta']:.2f} & {a['rho']:.1f} & {min(a['tcs_cover'], c['tcs_cover']):.3f} & {min(a['box_cover'], c['box_cover']):.3f} & {a['tcs_excl']:.2f} & {a['box_excl']:.2f} & {c['tcs_excl']:.2f} & {c['box_excl']:.2f} \\\\")
L.append("\\bottomrule\\end{tabular}")
open("manuscript/tables/box.tex", "w").write("\n".join(L))

# non-smooth criteria simulation
ns = json.load(open("results/simulation_nonsmooth.json"))
L = ["\\begin{tabular}{rrrrrrrr}\\toprule",
     "Design & $|\\mathcal{P}_0|$ & $n$ & \\multicolumn{3}{c}{Retention} & \\multicolumn{2}{c}{Exclusion} \\\\",
     " & & & TCS & Box & Naive & TCS & Box \\\\\\midrule"]
for r in ns:
    L.append(f"{r['design']} & {r['P0']} & {r['n']} & {r['tcs_retain']:.3f} & {r['box_retain']:.3f} & {r['naive_retain']:.3f} & {r['tcs_excl']:.2f} & {r['box_excl']:.2f} \\\\")
L.append("\\bottomrule\\end{tabular}")
open("manuscript/tables/nonsmooth.tex", "w").write("\n".join(L))

# real-data validation
rv = json.load(open("results/realdata_validation.json"))
lab = {"adult-sex": "Adult (sex)", "adult-race": "Adult (race)", "bank": "Bank (age)"}
L = ["\\begin{tabular}{llrrrrrrrr}\\toprule",
     "Data & $n$ & \\multicolumn{2}{c}{Retention} & \\multicolumn{3}{c}{Set size} & Winner & Excl. \\\\",
     " & & Naive & TCS & Naive & TCS & Ref. & correct & (TCS) \\\\\\midrule"]
last = None
for r in rv:
    name = lab[r["dataset"]] if r["dataset"] != last else ""; last = r["dataset"]
    L.append(f"{name} & {r['n']} & {r['naive_retain']:.2f} & {r['tcs_retain']:.2f} & {r['naive_size']:.1f} & {r['tcs_size']:.1f} & {r['P_ref']} & {r['winner_correct']:.2f} & {r['tcs_excl']:.2f} \\\\")
L.append("\\bottomrule\\end{tabular}")
open("manuscript/tables/realval.tex", "w").write("\n".join(L))
