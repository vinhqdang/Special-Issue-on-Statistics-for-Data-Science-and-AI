"""Real-data validation with a large held-out pool as the population proxy.

For each dataset, nine models are trained once. The large held-out pool (about 18-20k rows) defines
reference criteria and the reference nondominated set P_ref. Small test samples (n rows drawn
without replacement from the pool) are then analysed as a practitioner would: naive Pareto front,
composite-score winner, bootstrap TCS and a Bonferroni box procedure. We record how often each
recovers the reference conclusions.
"""
import sys, json, warnings, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tcs import trust_confidence_set, pareto_front
from experiments.casestudy import models
from experiments.simulate_nonsmooth import crit_w, dominated_mask, box_keep
warnings.filterwarnings("ignore")
CRIT = ["error", "ece", "eo_gap", "noise_drop"]


def load(name):
    if name.startswith("adult"):
        d = fetch_openml("adult", version=2, as_frame=True); X, y = d.data.copy(), (d.target == ">50K").astype(int).values
        if name == "adult-sex": g = (X["sex"] == "Female").astype(int).values; X = X.drop(columns=["sex"])
        else: g = (X["race"] != "White").astype(int).values; X = X.drop(columns=["race"])
    else:
        d = fetch_openml("bank-marketing", version=1, as_frame=True); X, y = d.data.copy(), (d.target == "2").astype(int).values
        g = (X["V1"] < 30).astype(int).values; X = X.drop(columns=["V1"])
    return X, y, g


def crit4(W, y, g, p, pred, predn):
    base = crit_w(W, y, g, p, pred)                                   # error, ECE, EO gap
    sw = W.sum(1, keepdims=True)
    nd = (W @ (predn != y[:, None])) / sw - base[:, :, 0]
    return np.concatenate([base, nd[:, :, None]], axis=2)


def prepare(name, seed=0):
    X, y, g = load(name)
    num = X.select_dtypes("number").columns.tolist(); cat = [c for c in X.columns if c not in num]
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, y, g, test_size=0.4, random_state=seed, stratify=y)
    ms = models(num, cat); rng = np.random.default_rng(seed)
    Xn = Xte.copy()
    for c in num: Xn[c] = Xn[c] + rng.normal(0, 0.5 * Xtr[c].std(), len(Xn))
    P, Q, PN = [], [], []
    for m in ms.values():
        m.fit(Xtr, ytr); P.append(m.predict(Xte)); Q.append(m.predict_proba(Xte)[:, 1]); PN.append(m.predict(Xn))
    return dict(names=list(ms), y=yte, g=gte, pred=np.column_stack(P).astype(float), p=np.column_stack(Q), predn=np.column_stack(PN).astype(float))


POOL = {}


def one(args):
    name, n, seed, B, alpha = args
    d = POOL[name]; rng = np.random.default_rng(seed)
    idx = rng.choice(len(d["y"]), n, replace=False)
    y, g, p, pred, predn = d["y"][idx], d["g"][idx], d["p"][idx], d["pred"][idx], d["predn"][idx]
    point = crit4(np.ones((1, n)), y, g, p, pred, predn)[0]
    boot = crit4(rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float), y, g, p, pred, predn)
    r = trust_confidence_set(point, boot, alpha)
    sd = d["ref"].std(0); win = int(((point / sd).sum(1)).argmin())
    return dict(tcs=r.keep, box=box_keep(point, boot, alpha), naive=r.pareto, win=win)


if __name__ == "__main__":
    R, B, alpha = int(sys.argv[1]), 300, 0.05
    names = sys.argv[2].split(",")
    out = []
    for name in names:
        d = prepare(name); N = len(d["y"])
        d["ref"] = crit4(np.ones((1, N)), d["y"], d["g"], d["p"], d["pred"], d["predn"])[0]
        POOL[name] = d
        dom = dominated_mask(d["ref"]); P0 = ~dom; M = len(P0)
        refwin = int(((d["ref"] / d["ref"].std(0)).sum(1)).argmin())
        print(name, "pool", N, "|P_ref|", int(P0.sum()), "ref winner", d["names"][refwin], flush=True)
        with Pool(4) as pool:
            for n in (500, 1000, 2000, 5000):
                res = pool.map(one, [(name, n, zlib.crc32(name.encode()) % 999983 + 31 * n + r, B, alpha) for r in range(R)])
                row = dict(dataset=name, pool=N, n=n, R=R, P_ref=int(P0.sum()), models=M, ref_winner=d["names"][refwin],
                           ref=d["ref"].tolist(), names=d["names"], ref_dominated=dom.tolist())
                for k in ("tcs", "box", "naive"):
                    keep = np.array([x[k] for x in res])
                    row[k + "_retain"] = float(keep[:, P0].all(1).mean())
                    row[k + "_excl"] = float((~keep[:, dom]).mean()) if dom.any() else None
                    row[k + "_size"] = float(keep.sum(1).mean())
                    row[k + "_retain_each"] = keep[:, P0].mean(0).tolist() if k != "naive" else None
                wins = np.array([x["win"] for x in res])
                row["winner_correct"] = float((wins == refwin).mean()); row["n_distinct_winners"] = int(len(set(wins)))
                out.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in ("n", "tcs_retain", "tcs_excl", "tcs_size", "box_excl", "naive_retain", "naive_excl", "naive_size", "winner_correct", "n_distinct_winners")}, flush=True)
    json.dump(out, open(f"results/realdata_validation.json", "w"), indent=1)
