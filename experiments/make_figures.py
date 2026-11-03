import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
sim = json.load(open("results/simulation.json"))
cs = json.load(open("results/casestudy.json"))
C1, C2 = "#1f4e79", "#c0504d"

# Fig 1: simulation
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for rho, ls in ((0.0, "-"), (0.5, "--")):
    for d, mk in ((0.02, "o"), (0.05, "s")):
        rows = [r for r in sim if r["rho"] == rho and r["delta"] == d]
        n = [r["n"] for r in rows]
        ax[0].plot(n, [r["tcs_cover"] for r in rows], ls, marker=mk, color=C1, ms=4)
        ax[0].plot(n, [r["naive_cover"] for r in rows], ls, marker=mk, color=C2, ms=4)
        ax[1].plot(n, [r["tcs_excl"] for r in rows], ls, marker=mk, color=C1, ms=4)
        ax[1].plot(n, [r["naive_excl"] for r in rows], ls, marker=mk, color=C2, ms=4)
ax[0].axhline(0.95, color="k", lw=.7, ls=":")
ax[0].set_ylabel("P(all non-dominated models retained)"); ax[1].set_ylabel("Share of dominated models excluded")
for a in ax: a.set_xscale("log"); a.set_xlabel("Test sample size n"); a.set_ylim(0, 1.03)
ax[0].plot([], [], color=C1, label="TCS"); ax[0].plot([], [], color=C2, label="Point-estimate Pareto front")
ax[0].plot([], [], "k-", label="rho=0"); ax[0].plot([], [], "k--", label="rho=0.5")
ax[0].legend(fontsize=7, loc="center right", frameon=False)
fig.tight_layout(); fig.savefig("figures/fig_simulation.pdf"); fig.savefig("figures/fig_simulation.png", dpi=200)

# Fig 2: case studies
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
for a, o in zip(ax, cs):
    m = o["models"]; y = np.arange(len(m))
    a.barh(y, o["pareto_prob"], color=C1, alpha=.85)
    a.set_yticks(y); a.set_yticklabels(m); a.invert_yaxis(); a.set_xlim(0, 1.45); a.set_xticks([0,.25,.5,.75,1])
    a.set_xlabel("Bootstrap probability of lying on the Pareto front")
    a.set_title(f"{'Adult (sex)' if o['dataset']=='adult' else 'German credit (age)'}, n_test={o['n_test']}", fontsize=9)
    for i in y:
        if not o["pareto"][i]: a.text(o["pareto_prob"][i] + .02, i, "off naive front", va="center", fontsize=7, color=C2)
fig.tight_layout(); fig.savefig("figures/fig_pareto_prob.pdf"); fig.savefig("figures/fig_pareto_prob.png", dpi=200)

# Fig 3: rank intervals of composite
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
for a, o in zip(ax, cs):
    m = o["models"]; y = np.arange(len(m))
    order = np.argsort(o["composite"])
    for r, i in enumerate(order):
        a.plot([o["rank_lo"][i], o["rank_hi"][i]], [r, r], color=C1, lw=2)
    a.set_yticks(range(len(m))); a.set_yticklabels([m[i] for i in order]); a.invert_yaxis()
    a.set_xlabel("95% bootstrap interval of composite-score rank"); a.set_title(o["dataset"], fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig_rank_intervals.pdf"); fig.savefig("figures/fig_rank_intervals.png", dpi=200)
