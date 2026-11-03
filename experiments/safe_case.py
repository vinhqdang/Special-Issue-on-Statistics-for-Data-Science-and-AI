"""TCS with SAFE-family (rank graduation) coordinates on Adult (sex) and German credit (age).

Run from the repository root:  python experiments/safe_case.py
Reuses the data loading, model zoo, split and perturbation of experiments/casestudy.py.

Criterion sets (all lower is better):
  conv4 : error, eo_gap, ece, noise_drop           (casestudy.py)
  conv3 : error, eo_gap, noise_drop                 (dimension-matched to the SAFE set)
  safe3 : 1-RGA(y, p), 1-RGR(p, p_noisy), |RGA_g0 - RGA_g1|
Probabilities p are predict_proba[:, 1]; p_noisy is the same model evaluated on the
feature-perturbed test set of casestudy.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import sys, json, time, warnings
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
import casestudy as cs
from sklearn.model_selection import train_test_split
from tcs import trust_confidence_set, pareto_front
from tcs.safe_metrics import rga, rgr, rga_parity_gap
warnings.filterwarnings("ignore")

CONV4 = ["error", "eo_gap", "ece", "noise_drop"]
SAFE3 = ["1-RGA", "1-RGR", "RGA_parity"]


def safe_stats(y, g, prob, prob_n, idx):
    yy, gg = y[idx].astype(float), g[idx]
    P, PN = prob[idx], prob_n[idx]
    a = 1.0 - rga(yy, P)
    r = 1.0 - rgr(P, PN)
    par = rga_parity_gap(yy, P, gg)
    return np.stack([a, r, par], axis=1)


def summarize(name, crit_names, point, boot, alpha):
    r = trust_confidence_set(point, boot, alpha)
    pp = np.mean([pareto_front(b) for b in boot], axis=0)
    return dict(criteria=crit_names, tcs=r.keep.tolist(), tcs_size=int(r.keep.sum()),
                pareto=r.pareto.tolist(), pareto_size=int(r.pareto.sum()),
                pareto_prob=pp.tolist(), stat=r.stat.tolist(), crit=float(r.crit),
                point=point.tolist(), boot_sd=boot.std(0).tolist())


def run(name, B=1000, alpha=0.05, seed=0):
    t0 = time.time()
    X, y, g = cs.load(name)
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, y, g, test_size=0.4, random_state=seed, stratify=y)
    ms = cs.models(num, cat)
    rng = np.random.default_rng(seed)
    Xn = Xte.copy()
    for c in num:
        Xn[c] = Xn[c] + rng.normal(0, 0.5 * Xtr[c].std(), len(Xn))
    pred, prob, predn, probn = [], [], [], []
    for k, m in ms.items():
        if k == "RandomForest":
            m.set_params(randomforestclassifier__n_jobs=2)
        m.fit(Xtr, ytr)
        pred.append(m.predict(Xte)); prob.append(m.predict_proba(Xte)[:, 1])
        predn.append(m.predict(Xn)); probn.append(m.predict_proba(Xn)[:, 1])
    pred, prob, predn, probn = map(np.column_stack, (pred, prob, predn, probn))
    n = len(yte)
    full = np.arange(n)
    conv_pt = cs.stats(yte, gte, pred, prob, predn, full)
    safe_pt = safe_stats(yte, gte, prob, probn, full)
    cb, sb = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        cb.append(cs.stats(yte, gte, pred, prob, predn, idx))
        sb.append(safe_stats(yte, gte, prob, probn, idx))
    cb, sb = np.stack(cb), np.stack(sb)
    sets = {
        "conv4": (CONV4, conv_pt, cb),
        "conv3": ([CONV4[i] for i in (0, 1, 3)], conv_pt[:, [0, 1, 3]], cb[:, :, [0, 1, 3]]),
        "safe3": (SAFE3, safe_pt, sb),
    }
    # one-at-a-time swaps of a conventional coordinate for its SAFE counterpart
    pairs = {"swap_acc": (0, 0), "swap_rob": (3, 1), "swap_fair": (1, 2)}
    for key, (ci, si) in pairs.items():
        pt, bt = conv_pt[:, [0, 1, 3]].copy(), cb[:, :, [0, 1, 3]].copy()
        pos = {0: 0, 1: 1, 3: 2}[ci]
        pt[:, pos] = safe_pt[:, si]; bt[:, :, pos] = sb[:, :, si]
        nm = [CONV4[i] for i in (0, 1, 3)]; nm[pos] = SAFE3[si]
        sets[key] = (nm, pt, bt)
    res = {k: summarize(k, *v, alpha) for k, v in sets.items()}
    # criteria agreement across models (Spearman) between conventional and SAFE counterparts
    from scipy.stats import spearmanr
    agree = {f"{CONV4[ci]}~{SAFE3[si]}": float(spearmanr(conv_pt[:, ci], safe_pt[:, si])[0])
             for ci, si in [(0, 0), (3, 1), (1, 2)]}
    return dict(dataset=name, n_test=int(n), n_train=int(len(ytr)), B=B, alpha=alpha,
                models=list(ms), sets=res, spearman_conv_vs_safe=agree,
                seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out = [run("adult"), run("credit-g")]
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/safe_case.json", "w"), indent=1)
    for o in out:
        print(o["dataset"], o["n_test"], o["seconds"], "s", o["spearman_conv_vs_safe"])
        for k, v in o["sets"].items():
            print(f"  {k:10s} TCS={v['tcs_size']} Pareto={v['pareto_size']} "
                  f"pP={np.round(v['pareto_prob'], 2).tolist()}")
