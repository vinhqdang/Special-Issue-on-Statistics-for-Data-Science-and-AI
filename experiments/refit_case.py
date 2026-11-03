"""Refit-bootstrap TCS vs fixed-model TCS on German credit and (subsampled) Adult.

usage: python experiments/refit_case.py credit-g [B] [R]
       python experiments/refit_case.py adult N_SUB [B] [R]
Writes results/refit_case.json (merging by dataset key).
"""
import os, sys, json, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, "."); sys.path.insert(0, "experiments")
import numpy as np
from sklearn.model_selection import train_test_split
import casestudy as cs
from tcs import trust_confidence_set
from tcs import refit as rf
warnings.filterwarnings("ignore")


def fixed_model(X, y, g, tr, te, make_models, num, B, alpha, seed=0):
    """Fixed-model TCS exactly as experiments/casestudy.py (same split, noise, bootstrap)."""
    rng = np.random.default_rng(seed)
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    Xn = Xte.copy()
    for c in num:
        Xn[c] = Xn[c] + rng.normal(0, 0.5 * Xtr[c].std(), len(Xn))
    pred, prob, predn = [], [], []
    for m in make_models().values():
        m.fit(Xtr, y[tr]); pred.append(m.predict(Xte)); prob.append(m.predict_proba(Xte)[:, 1]); predn.append(m.predict(Xn))
    pred, prob, predn = map(np.column_stack, (pred, prob, predn))
    n = len(te)
    point = cs.stats(y[te], g[te], pred, prob, predn, np.arange(n))
    boot = np.stack([cs.stats(y[te], g[te], pred, prob, predn, rng.integers(0, n, n)) for _ in range(B)])
    return point, boot


def main():
    name = sys.argv[1]
    a = sys.argv[2:]
    n_sub = None
    if name == "adult":
        n_sub = int(a.pop(0))
    B = int(a[0]) if len(a) > 0 else 200
    R = int(a[1]) if len(a) > 1 else B
    alpha, seed = 0.05, 0
    X, y, g = cs.load(name)
    if n_sub:
        sub = np.random.default_rng(seed).choice(len(y), n_sub, replace=False)
        X, y, g = X.iloc[sub].reset_index(drop=True), y[sub], g[sub]
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    mk = lambda: cs.models(num, cat)
    names = list(mk())
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.4, random_state=seed, stratify=y)
    T = {}
    t = time.time()
    point, bootF = fixed_model(X, y, g, tr, te, mk, num, 1000, alpha, seed)
    T["fixed"] = time.time() - t
    t = time.time()
    bootP = rf.paired_bootstrap(mk, cs.stats, X, y, g, tr, te, B, seed=1, num=num)
    T["paired"] = time.time() - t; print("paired", T["paired"], flush=True)
    t = time.time()
    bootS = rf.repeated_splits(mk, cs.stats, X, y, g, R, 0.4, seed=2, num=num)
    T["splits"] = time.time() - t; print("splits", T["splits"], flush=True)
    t = time.time()
    bootO = rf.oob_bootstrap(mk, cs.stats, X, y, g, B, seed=3, num=num)
    T["oob"] = time.time() - t; print("oob", T["oob"], flush=True)

    res = dict(dataset=name, n=int(len(y)), n_train=int(len(tr)), n_test=int(len(te)), B_fixed=1000, B_refit=B, R_splits=R,
               alpha=alpha, models=names, criteria=cs.CRIT, seconds=T, point_fixed=point.tolist())
    res["fixed"] = rf.summarize(point, bootF, trust_confidence_set, alpha)
    res["paired_refit"] = rf.summarize(point, bootP, trust_confidence_set, alpha)
    res["repeated_splits"] = rf.summarize(bootS.mean(0), bootS, trust_confidence_set, alpha)
    res["oob_refit"] = rf.summarize(point, bootO, trust_confidence_set, alpha)
    res["sd_ratio_paired_over_fixed"] = (bootP.std(0) / bootF.std(0)).tolist()
    res["sd_ratio_splits_over_fixed"] = (bootS.std(0) / bootF.std(0)).tolist()
    res["sd_ratio_oob_over_fixed"] = (bootO.std(0) / bootF.std(0)).tolist()
    res["membership_changes_vs_fixed"] = {k: [names[i] for i in range(len(names)) if res[k]["tcs"][i] != res["fixed"]["tcs"][i]]
                                          for k in ("paired_refit", "repeated_splits", "oob_refit")}
    path = "results/refit_case.json"
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr[name] = res
    json.dump(allr, open(path, "w"), indent=1)
    for k in ("fixed", "paired_refit", "repeated_splits", "oob_refit"):
        print(name, k, "size", res[k]["size"], "crit", round(res[k]["crit"], 2), "tcs", [int(v) for v in res[k]["tcs"]])
    print("changes", res["membership_changes_vs_fixed"])


if __name__ == "__main__":
    main()
