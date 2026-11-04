"""Certified selection on twelve real data settings, validated against large held-out references,
plus a test-sample-size analysis per criterion.

Main criteria (natural units): error, ECE, equalised-odds gap, noise drop. W(r) = box of half-width r
around equal weights within the simplex. For each test sample: plug-in minimax-regret choice, the
equal-weight argmin ("naive"), and certified upper bounds (one-stage, valid; selective, heuristic).
Truth: regrets computed on the held-out pool.

Sample-size analysis: each criterion alone (including a smooth "soft" equalised-odds gap built from
predicted probabilities), one-stage certificate of the plug-in choice at two test sizes; the
certificate is extrapolated as U(n) ~ U(n0) sqrt(n0/n) and the prediction is checked at the other size.
"""
import sys, os, json, zlib, warnings
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from experiments.preference_validation import get_pool, SETTINGS
from experiments import realdata_validation as rv
from tcs.certify import certify
from tcs.preference import box_simplex_vertices, worst_regret
warnings.filterwarnings("ignore")
RADII = [0.0, 0.1, 0.25]
W0 = np.full(4, 0.25)
VERT = {r: box_simplex_vertices(W0, r) for r in RADII}
CRIT5 = ["error", "ece", "eo_gap", "noise_drop", "soft_eo_gap"]
POOLS = {}


def crit5(W, y, g, p, pred, predn):
    c = rv.crit4(W, y, g, p, pred, predn)
    soft = np.zeros(c.shape[:2])
    for lab in (1, 0):
        m0 = ((y == lab) & (g == 0)).astype(float)[:, None]; m1 = ((y == lab) & (g == 1)).astype(float)[:, None]
        a = (W @ (p * m0)) / np.maximum(W @ m0, 1e-12); b = (W @ (p * m1)) / np.maximum(W @ m1, 1e-12)
        soft = np.maximum(soft, np.abs(a - b))
    return np.concatenate([c, soft[:, :, None]], axis=2)


def one(args):
    key, n, seed, B, alpha = args
    d = POOLS[key]; rng = np.random.default_rng(seed)
    idx = rng.choice(len(d["y"]), n, replace=False)
    a = [d[k][idx] for k in ("y", "g", "p", "pred", "predn")]
    pt = crit5(np.ones((1, n)), *a)[0]
    bt = crit5(rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float), *a)
    out = {"naive": int((pt[:, :4] @ W0).argmin())}
    for r in RADII:
        c = certify(pt[:, :4], bt[:, :, :4], VERT[r], alpha)
        out[r] = (c["choice"], float(c["U_one"][c["choice"]]), c["U_sel"])
    for k in range(5):
        c = certify(pt[:, [k]], bt[:, :, [k]], np.ones((1, 1)), alpha)
        out[f"k{k}"] = (c["choice"], float(c["U_one"][c["choice"]]))
    return out


if __name__ == "__main__":
    R, B, alpha = int(sys.argv[1]), 300, 0.05
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else SETTINGS
    rows, ss = [], []
    for key in keys:
        d = get_pool(key); POOLS[key] = d; N = len(d["y"])
        ref = crit5(np.ones((1, N)), d["y"], d["g"], d["p"], d["pred"], d["predn"])[0]
        sizes = [n for n in (500, 2000, 5000) if n <= N / 3] or [500]
        res_by_n = {}
        with Pool(4) as pool:
            for n in sizes:
                res_by_n[n] = pool.map(one, [(key, n, zlib.crc32(key.encode()) % 99991 + 11 * n + i, B, alpha) for i in range(R)])
        for n, res in res_by_n.items():
            for r in RADII:
                Rt = worst_regret(ref[:, :4], VERT[r])
                A = np.array([[Rt[x[r][0]], x[r][1], x[r][2], Rt[x["naive"]]] for x in res])
                row = dict(dataset=key, pool=N, n=n, r=r, R=R, best_regret=float(Rt.min()),
                           regret_choice=float(A[:, 0].mean()), regret_choice_q90=float(np.quantile(A[:, 0], .9)),
                           regret_naive=float(A[:, 3].mean()), regret_naive_q90=float(np.quantile(A[:, 3], .9)),
                           U_one=float(A[:, 1].mean()), cover_one=float((A[:, 0] <= A[:, 1] + 1e-12).mean()),
                           U_sel=float(A[:, 2].mean()), cover_sel=float((A[:, 0] <= A[:, 2] + 1e-12).mean()))
                rows.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("pool", "R")}, flush=True)
        # sample-size analysis: per criterion
        n_hi, n_lo = max(sizes), min(sizes)
        for k in range(5):
            Rk = ref[:, k] - ref[:, k].min()
            Uh = np.mean([x[f"k{k}"][1] for x in res_by_n[n_hi]]); Ul = np.mean([x[f"k{k}"][1] for x in res_by_n[n_lo]])
            cov = np.mean([Rk[x[f"k{k}"][0]] <= x[f"k{k}"][1] + 1e-12 for x in res_by_n[n_hi]])
            ss.append(dict(dataset=key, criterion=CRIT5[k], n_hi=n_hi, U_hi=float(Uh), n_lo=n_lo, U_lo=float(Ul),
                           U_lo_predicted=float(Uh * np.sqrt(n_hi / n_lo)), cover_hi=float(cov),
                           n_for_001=float(n_hi * (Uh / 0.01) ** 2), n_for_002=float(n_hi * (Uh / 0.02) ** 2)))
            print(ss[-1], flush=True)
        json.dump(dict(rows=rows, sample_size=ss), open("results/certify_realdata.json", "w"), indent=1)
