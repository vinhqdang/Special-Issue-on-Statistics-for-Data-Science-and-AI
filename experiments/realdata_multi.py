"""Extend the held-out-pool validation to more binary-classification data sets (OpenML).

Group attributes: credit-card default (sex), law-school (race) are natural; for the other data sets
the group is an arbitrary split (first numeric feature above its median) and is removed from the
features. Those groups only exercise the equalised-odds coordinate and have no fairness meaning.
"""
import sys, json, warnings, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, ".")
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from experiments import realdata_validation as rv
from experiments.casestudy import models
from experiments.simulate_nonsmooth import dominated_mask
warnings.filterwarnings("ignore")

REG = {  # name: (openml name, group kind)
    "credit-card": ("default-of-credit-card-clients", "sex"),
    "law-school": ("law-school-admission-bianry", "race"),
    "nomao": ("nomao", "arbitrary"), "electricity": ("electricity", "arbitrary"),
    "magic": ("MagicTelescope", "arbitrary"), "eeg": ("eeg-eye-state", "arbitrary"),
    "phoneme": ("phoneme", "arbitrary"), "spambase": ("spambase", "arbitrary"), "churn": ("churn", "arbitrary"),
}
CAP = 30000


def load(key):
    oname, kind = REG[key]
    d = fetch_openml(oname, version=1, as_frame=True, parser="auto"); X = d.data.copy(); t = d.target.astype(str)
    y = (t == t.value_counts().index[-1]).astype(int).values            # minority class = positive
    if key == "credit-card": g = (X["x2"] == 2).astype(int).values; X = X.drop(columns=["x2"])
    elif key == "law-school": g = (X["race1"] != "white").astype(int).values; X = X.drop(columns=["race1"])
    else:
        c = X.select_dtypes("number").columns[0]; g = (X[c] > X[c].median()).astype(int).values; X = X.drop(columns=[c])
    if len(y) > CAP:
        keep = np.random.default_rng(0).choice(len(y), CAP, replace=False)
        X, y, g = X.iloc[keep].reset_index(drop=True), y[keep], g[keep]
    return X, y, g


def prepare(key, seed=0):
    X, y, g = load(key)
    num = X.select_dtypes("number").columns.tolist(); cat = [c for c in X.columns if c not in num]
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, y, g, test_size=0.4, random_state=seed, stratify=y)
    ms = models(num, cat); rng = np.random.default_rng(seed); Xn = Xte.copy()
    for c in num: Xn[c] = Xn[c] + rng.normal(0, 0.5 * Xtr[c].std(), len(Xn))
    P, Q, PN = [], [], []
    for m in ms.values():
        m.fit(Xtr, ytr); P.append(m.predict(Xte)); Q.append(m.predict_proba(Xte)[:, 1]); PN.append(m.predict(Xn))
    return dict(names=list(ms), y=yte, g=gte, pred=np.column_stack(P).astype(float), p=np.column_stack(Q), predn=np.column_stack(PN).astype(float))


if __name__ == "__main__":
    R, B, alpha = int(sys.argv[1]), 300, 0.05
    keys = sys.argv[2].split(","); out = []
    for key in keys:
        try:
            d = prepare(key)
        except Exception as e:
            print(key, "FAILED", repr(e)[:120], flush=True); continue
        N = len(d["y"]); d["ref"] = rv.crit4(np.ones((1, N)), d["y"], d["g"], d["p"], d["pred"], d["predn"])[0]
        rv.POOL[key] = d; dom = dominated_mask(d["ref"]); P0 = ~dom; M = len(P0)
        refwin = int(((d["ref"] / d["ref"].std(0)).sum(1)).argmin())
        print(key, "pool", N, "|P_ref|", int(P0.sum()), "winner", d["names"][refwin], flush=True)
        with Pool(4) as pool:
            for n in [n for n in (200, 500, 1000, 2000, 5000) if n <= N / 3]:
                res = pool.map(rv.one, [(key, n, zlib.crc32(key.encode()) % 999983 + 31 * n + r, B, alpha) for r in range(R)])
                row = dict(dataset=key, group=REG[key][1], pool=N, n=n, R=R, P_ref=int(P0.sum()), models=M, ref_winner=d["names"][refwin],
                           ref=d["ref"].tolist(), names=d["names"], ref_dominated=dom.tolist())
                for k in ("tcs", "box", "naive"):
                    keep = np.array([x[k] for x in res])
                    row[k + "_retain"] = float(keep[:, P0].all(1).mean()); row[k + "_excl"] = float((~keep[:, dom]).mean()) if dom.any() else None
                    row[k + "_size"] = float(keep.sum(1).mean())
                wins = np.array([x["win"] for x in res])
                row["winner_correct"] = float((wins == refwin).mean()); row["n_distinct_winners"] = int(len(set(wins)))
                out.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in ("n", "tcs_retain", "tcs_excl", "tcs_size", "naive_retain", "naive_size", "winner_correct")}, flush=True)
        json.dump(out, open("results/realdata_multi.json", "w"), indent=1)
