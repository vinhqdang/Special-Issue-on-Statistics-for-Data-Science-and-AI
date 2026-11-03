"""Case studies with step-down / margin variants (reuses experiments/casestudy.py unchanged)."""
import sys, json, itertools
import numpy as np
sys.path.insert(0, ".")
from sklearn.model_selection import train_test_split
from experiments.casestudy import load, models, stats, CRIT
from tcs import trust_confidence_set
from tcs.stepdown import trust_confidence_set_sd

EPS = {"small": [0.005, 0.01, 0.005, 0.005], "large": [0.01, 0.02, 0.01, 0.01]}   # error, eo_gap, ece, noise_drop


def point_boot(name, B=1000, seed=0):
    # identical data pipeline and random-number stream as casestudy.run
    X, y, g = load(name)
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, y, g, test_size=0.4, random_state=seed, stratify=y)
    ms = models(num, cat)
    rng = np.random.default_rng(seed)
    Xn = Xte.copy()
    for c in num:
        Xn[c] = Xn[c] + rng.normal(0, 0.5 * Xtr[c].std(), len(Xn))
    pred, prob, predn = [], [], []
    for k, m in ms.items():
        m.fit(Xtr, ytr)
        pred.append(m.predict(Xte)); prob.append(m.predict_proba(Xte)[:, 1]); predn.append(m.predict(Xn))
    pred, prob, predn = map(np.column_stack, (pred, prob, predn))
    n = len(yte)
    point = stats(yte, gte, pred, prob, predn, np.arange(n))
    boot = np.stack([stats(yte, gte, pred, prob, predn, rng.integers(0, n, n)) for _ in range(B)])
    return list(ms), point, boot, n


def variants(point, boot, alpha, names=None):
    K = point.shape[1]
    out = {}
    r0 = trust_confidence_set(point, boot, alpha)
    out["orig"] = (r0.keep, r0.crit, 1)
    for nm, kw in {"stepdown": {}, "sd_pre": dict(preselect=0.01)}.items():
        r = trust_confidence_set_sd(point, boot, alpha, **kw)
        out[nm] = (r.keep, r.crit, r.n_steps)
    for lab, e in EPS.items():
        r = trust_confidence_set_sd(point, boot, alpha, eps=np.array(e)[:K] if K == 4 else 0.0)
        out[f"sd_eps_{lab}"] = (r.keep, r.crit, r.n_steps)
    return out, r0


if __name__ == "__main__":
    alpha = 0.05
    res = []
    for name in ("adult", "credit-g"):
        ms, point, boot, n = point_boot(name)
        v, r0 = variants(point, boot, alpha)
        row = dict(dataset=name, n_test=int(n), alpha=alpha, models=ms, criteria=CRIT, S=r0.stat.tolist(),
                   pareto=r0.pareto.tolist(),
                   variants={k: dict(keep=a.tolist(), n_excluded=int((~a).sum()), crit=c, steps=s) for k, (a, c, s) in v.items()},
                   S_minus_orig_crit_max=float(r0.stat.max() - r0.crit))
        subs = []
        for kk in (2, 3):
            for c in itertools.combinations(range(4), kk):
                sv, _ = variants(point[:, c], boot[:, :, c], alpha)
                subs.append(dict(criteria=[CRIT[i] for i in c],
                                 n_excluded={k: int((~a).sum()) for k, (a, _, _) in sv.items() if not k.startswith("sd_eps")}))
        row["subsets"] = subs
        res.append(row)
        print(name, {k: (x["n_excluded"], round(x["crit"], 2)) for k, x in row["variants"].items()},
              "max S", round(float(r0.stat.max()), 2), "subsets with exclusions:",
              {k: sum(s["n_excluded"][k] > 0 for s in subs) for k in subs[0]["n_excluded"]}, flush=True)
    json.dump(res, open("results/casestudy_power.json", "w"), indent=1)
