"""Coverage of POPULATION contrasts: pool-conditional Wald intervals versus two-phase intervals.

Population: generative model of experiments/simulate_nonsmooth.py (x ~ N(0, I_5), g ~ Bernoulli(0.3), y from a
logistic model), nine fixed scoring rules (design seed DSEED), 36 pairs x {dTPR, dFPR, dERR} = 108 contrasts.
Population values: totals over NPOP = 6e6 draws (chunked). For each replicate a NEW pool of N units is drawn, then
a PATTERN stratified label sample (L labels, 'plain' Neyman allocation with the mean-probability proxy), and
PATTERN-DIF is evaluated: (a) pool-conditional Wald interval (as in the main study), (b) two-phase interval
(tcs.labeling_two_phase). Coverage of the pool value is also recorded for (a) and (b).
Usage: python experiments/two_phase_check.py run R | report
"""
import sys, os, json, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from experiments import simulate_nonsmooth as sn
from tcs.labeling import total_columns, Estimands, make_strata, neyman_allocation, draw
from tcs.labeling_dif import DifContext
from tcs.labeling_two_phase import estimate_two_phase

sn.M = 9
M, DSEED, NPOP = 9, 1, 6_000_000
NS = (2000, 8000, 20000); LS = (250, 500, 1000, 2000)
OUT = "results/two_phase_check.json"
DES = sn.make_design(DSEED)
PAIRS = None


def totals(n, rng, chunk=300_000):
    T = 0; left = n; names = None
    while left > 0:
        c = min(chunk, left); left -= c
        x, g, y = sn.draw(c, rng); p, pred = sn.models(DES, x, g); U, V, names = total_columns(pred.astype(int), g)
        T = T + (y[:, None] * U + (1 - y[:, None]) * V).sum(0)
    return T, names


def pop_truth():
    f = "results/two_phase_truth.npz"
    if os.path.exists(f): return np.load(f)["theta"]
    T, names = totals(NPOP, np.random.default_rng(2024))
    est = Estimands(names, M, NPOP); th = est.value(T); np.savez(f, theta=th); return th


def make_pool(N, rng):
    x, g, y = sn.draw(N, rng); p, pred = sn.models(DES, x, g)
    pred = pred.astype(int); U, V, names = total_columns(pred, g)
    return dict(pred=pred, g=g, y=y.astype(float), p=p.mean(1), U=U, V=V, est=Estimands(names, M, N), N=N)


def rep(args):
    N, L, seed = args
    rng = np.random.default_rng(seed); d = make_pool(N, rng); U, V, y, p, est = d["U"], d["V"], d["y"], d["p"], d["est"]
    st = make_strata(d["pred"], d["g"], "pattern"); ids, Nh, nh = neyman_allocation(st, U, V, p, est, L)
    ctx = DifContext(st, ids, U, V, p, est)
    idx = draw(st, ids, nh, rng)
    o = estimate_two_phase(st, ids, Nh, nh, U, V, y[idx], idx, est, p, ctx=ctx)
    tpool = est.value((y[:, None] * U + (1 - y[:, None]) * V).sum(0))
    o["theta_pool"] = tpool
    return o


def summarise(N, L, R, res, truth, kinds):
    ok = np.array([np.isfinite(o["theta"]).all() and np.isfinite(o["se_total"]).all() for o in res]); res = [o for o, k in zip(res, ok) if k]
    A = {k: np.array([o[k] for o in res]) for k in res[0]}
    row = dict(N=N, L=L, R=R, fail=float(1 - ok.mean()))
    tol = 1e-9

    def cov(lo, hi, tgt): return ((lo <= tgt) & (tgt <= hi)).mean(0)
    for tag, (lo, hi) in {"design": (A["lo_design"], A["hi_design"]), "total": (A["lo_total"], A["hi_total"])}.items():
        cp = cov(lo, hi, truth[None, :]); cq = cov(lo, hi, A["theta_pool"])
        row[f"pop_{tag}_mean"] = float(cp.mean()); row[f"pop_{tag}_worst"] = float(cp.min()); row[f"pop_{tag}_p05"] = float(np.quantile(cp, 0.05))
        row[f"pool_{tag}_mean"] = float(cq.mean()); row[f"pool_{tag}_worst"] = float(cq.min())
        for k in ("dTPR", "dFPR", "dERR"): row[f"pop_{tag}_mean_{k}"] = float(cp[kinds == k].mean())
        row[f"width_{tag}_median"] = float(np.median((hi - lo).mean(0)))
    vr = (A["se_pool"] ** 2).mean(0) / (A["se_design"] ** 2).mean(0)                # estimated V_pool / V_design per target
    row["vratio_median"] = float(np.median(vr)); row["vratio_min"] = float(vr.min()); row["vratio_max"] = float(vr.max())
    mse_pop = ((A["theta"] - truth[None, :]) ** 2).mean(0); mse_pool = ((A["theta"] - A["theta_pool"]) ** 2).mean(0)
    er = mse_pop / np.maximum(mse_pool, 1e-30)                                       # empirical MSE(pop)/MSE(pool)
    row["mse_ratio_median"] = float(np.median(er)); row["mse_ratio_min"] = float(er.min()); row["mse_ratio_max"] = float(er.max())
    row["vtotal_over_mse_pop_median"] = float(np.median(((A["se_total"] ** 2).mean(0)) / mse_pop))
    row["pop_cover_by_vratio_bins"] = None
    # per-target pairs (vratio, coverage) for the figure
    row["per_target"] = dict(vratio=vr.tolist(), cov_design=cov(A["lo_design"], A["hi_design"], truth[None, :]).tolist(),
                             cov_total=cov(A["lo_total"], A["hi_total"], truth[None, :]).tolist())
    return row


def main_run(R):
    truth = pop_truth(); est0 = Estimands(total_columns(np.zeros((4, M), int), np.zeros(4, int))[2], M, 10)
    kinds = np.array([k for (k, _, _) in est0.labels]); rows = []
    for N in NS:
        for L in LS:
            if L > N / 4: continue
            with Pool(3) as pool:
                res = pool.map(rep, [(N, L, zlib.crc32(f"tp{N}_{L}".encode()) % 100003 * 1000 + r) for r in range(R)], chunksize=4)
            row = summarise(N, L, R, res, truth, kinds); rows.append(row)
            print(N, L, {k: round(row[k], 3) for k in ("pop_design_mean", "pop_design_worst", "pop_total_mean", "pop_total_worst", "pool_design_mean", "vratio_median", "fail")}, flush=True)
            json.dump(dict(R=R, dseed=DSEED, npop=NPOP, rows=rows), open(OUT, "w"))
    return rows


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    J = json.load(open(OUT)); rows = J["rows"]
    out = [r"\begin{tabular}{rr r rr rr r}\toprule",
           r" & & & \multicolumn{2}{c}{Pool-conditional} & \multicolumn{2}{c}{Two-phase} & Pool value \\",
           r"$N$ & $L$ & $V_{\mathrm{pool}}/V_{\mathrm{des}}$ & mean & worst & mean & worst & (pool-cond.) mean \\\midrule"]
    lastN = None
    for r in rows:
        if lastN is not None and r["N"] != lastN: out.append(r"\addlinespace")
        out.append((f"{r['N']:,}".replace(",", "{,}") if r["N"] != lastN else "") + f" & {r['L']} & {r['vratio_median']:.2f} & {r['pop_design_mean']:.3f} & {r['pop_design_worst']:.2f} & "
                   f"{r['pop_total_mean']:.3f} & {r['pop_total_worst']:.2f} & {r['pool_design_mean']:.3f}" + r" \\")
        lastN = r["N"]
    out.append(r"\bottomrule\end{tabular}")
    os.makedirs("manuscript_v2/tables", exist_ok=True); os.makedirs("manuscript_v2/figures", exist_ok=True)
    open("manuscript_v2/tables/two_phase_check.tex", "w").write("\n".join(out) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cN = {2000: "#d1495b", 8000: "#1b4f9c", 20000: "#00798c"}
    ax = axes[0]
    for r in rows:
        ax.scatter(r["vratio_median"], r["pop_design_mean"], marker="o", s=45, color=cN[r["N"]], label=f"pool-conditional, N={r['N']}" if r["L"] == min(x["L"] for x in rows if x["N"] == r["N"]) else None)
        ax.scatter(r["vratio_median"], r["pop_total_mean"], marker="^", s=45, facecolor="none", edgecolor=cN[r["N"]], label=f"two-phase, N={r['N']}" if r["L"] == min(x["L"] for x in rows if x["N"] == r["N"]) else None)
    ax.axhline(0.95, color="grey", ls="--", lw=0.8); ax.set_xscale("log"); ax.set_xlabel(r"median $\hat V_{\rm pool}/\hat V_{\rm design}$"); ax.set_ylabel("mean coverage of population contrasts")
    ax.set_ylim(0.5, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax = axes[1]
    for r in rows:
        ax.scatter(r["vratio_median"], r["pop_design_worst"], marker="o", s=45, color=cN[r["N"]])
        ax.scatter(r["vratio_median"], r["pop_total_worst"], marker="^", s=45, facecolor="none", edgecolor=cN[r["N"]])
    ax.axhline(0.95, color="grey", ls="--", lw=0.8); ax.set_xscale("log"); ax.set_xlabel(r"median $\hat V_{\rm pool}/\hat V_{\rm design}$"); ax.set_ylabel("worst-contrast coverage (of 108)")
    ax.set_ylim(0.0, 1.0); ax.grid(alpha=0.3)
    ax.set_title("circles: pool-conditional, triangles: two-phase", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"manuscript_v2/figures/two_phase_check.{ext}", dpi=200)
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("per_target", "pop_cover_by_vratio_bins")})


if __name__ == "__main__":
    if sys.argv[1] == "run": main_run(int(sys.argv[2]))
    else: report()
