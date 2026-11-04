"""Label efficiency of stratified designs for comparing classifiers on fairness and error.

For each data setting (cached pool: predictions of nine fixed models, group, labels used only as
truth and to answer label queries) and budget L, repeat R times:
  srs      simple random sampling of L labels
  group    strata = group, proxy-Neyman allocation (oversamples the minority group)
  pattern  strata = group x number of models predicting 1, proxy-Neyman allocation (proposed)
  oracle   as pattern, but allocation computed with the true labels (upper bound, not usable)
and record squared errors and 95% CI coverage for every pairwise comparison estimand (dTPR, dFPR, dERR).
Relative efficiency = MSE(srs) / MSE(method), i.e. how many times more labels SRS needs.
"""
import sys, json, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, make_strata, neyman_allocation, proportional_allocation, estimate, draw
from experiments.preference_validation import SETTINGS

METHODS = ["srs", "group", "pattern", "oracle"]
S = {}


def setup(key):
    z = np.load(f"results/pool_cache/{key}.npz", allow_pickle=True)
    pred, g, y, p = z["pred"], z["g"].astype(int), z["y"].astype(float), z["p"].mean(1)
    U, V, names = total_columns(pred, g); N, M = pred.shape
    est = Estimands(names, M, N)
    truth = est.value((y[:, None] * U + (1 - y[:, None]) * V).sum(0))
    err = (pred != y[:, None]).mean(0); top = np.argsort(err)[:3]
    top_pairs = [k for k, (m, j) in enumerate(est.pairs) if m in top and j in top]
    return dict(pred=pred, g=g, y=y, p=p, U=U, V=V, est=est, truth=truth, N=N,
                top_targets=[3 * k + r for k in top_pairs for r in range(3)],
                Pg=[int(((g == gg) & (y == 1)).sum()) for gg in (0, 1)], Qg=[int(((g == gg) & (y == 0)).sum()) for gg in (0, 1)],
                disagree_top=float(np.mean([(pred[:, a] != pred[:, b]).mean() for a in top for b in top if a < b])))


def designs(d, L):
    out = {}
    one = np.zeros(d["N"], int)
    out["srs"] = (one,) + proportional_allocation(one, L)
    sg = make_strata(d["pred"], d["g"], "group"); out["group"] = (sg,) + neyman_allocation(sg, d["U"], d["V"], d["p"], d["est"], L)
    sp = make_strata(d["pred"], d["g"], "pattern"); out["pattern"] = (sp,) + neyman_allocation(sp, d["U"], d["V"], d["p"], d["est"], L)
    out["oracle"] = (sp,) + neyman_allocation(sp, d["U"], d["V"], d["y"], d["est"], L)
    return out


def run(args):
    key, L, seed = args
    d = S[key]; rng = np.random.default_rng(seed); res = {}
    for name, (st, ids, Nh, nh) in S[(key, L)].items():
        idx = draw(st, ids, nh, rng)
        th, se, lo, hi = estimate(st, ids, Nh, nh, d["U"], d["V"], d["y"][idx], idx, d["est"])
        res[name] = (th, lo, hi)
    return res


if __name__ == "__main__":
    R = int(sys.argv[1]); keys = sys.argv[2].split(",") if len(sys.argv) > 2 else SETTINGS
    rows = []
    for key in keys:
        d = setup(key)
        if min(d["Pg"] + d["Qg"]) < 30:
            print(key, "skipped: a group has fewer than 30 positives or negatives", d["Pg"], d["Qg"], flush=True); continue
        S[key] = d
        for L in (250, 500, 1000, 2000):
            if L > d["N"] / 4: continue
            S[(key, L)] = designs(d, L)
            with Pool(4) as pool:
                res = pool.map(run, [(key, L, zlib.crc32(key.encode()) % 9973 + 17 * L + r) for r in range(R)])
            truth = d["truth"]
            mse = {m: np.mean([(x[m][0] - truth) ** 2 for x in res], axis=0) for m in METHODS}
            cov = {m: np.mean([(x[m][1] <= truth) & (truth <= x[m][2]) for x in res], axis=0) for m in METHODS}
            kinds = np.array([k for (k, _, _) in d["est"].labels])
            row = dict(dataset=key, N=d["N"], L=L, R=R, Pg=d["Pg"], disagree_top=d["disagree_top"])
            for m in METHODS[1:]:
                eff = mse["srs"] / np.maximum(mse[m], 1e-30)
                row[f"eff_{m}_all"] = float(np.median(eff))
                row[f"eff_{m}_top"] = float(np.median(eff[d["top_targets"]]))
                for k in ("dTPR", "dFPR", "dERR"):
                    row[f"eff_{m}_{k}"] = float(np.median(eff[kinds == k]))
            for m in METHODS:
                row[f"cover_{m}"] = float(cov[m].mean()); row[f"cover_{m}_min"] = float(cov[m].min())
                big = np.abs(truth) > 0.02
                row[f"sign_{m}"] = float(np.mean([np.mean(np.sign(x[m][0][big]) == np.sign(truth[big])) for x in res])) if big.any() else None
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k.startswith(("dataset", "L", "eff_pattern", "eff_group_all", "eff_oracle_all", "cover_srs", "cover_pattern", "sign_srs", "sign_pattern"))}, flush=True)
        json.dump(rows, open("results/label_efficiency.json", "w"), indent=1)
