"""Simulation: retention of population-nondominated ALGORITHMS by fixed-model TCS,
paired-refit TCS and repeated-split TCS, on synthetic data with known generative model.

usage: python experiments/refit_sim.py [reps] [B] [n]
Writes results/refit_sim.json.
"""
import os, sys, json, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, "."); sys.path.insert(0, "experiments")
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
import casestudy as cs
from tcs import trust_confidence_set, pareto_front
from tcs import refit as rf
warnings.filterwarnings("ignore")
P = 6
NUM = list(range(P))


CRIT3 = ["error", "ece", "noisy_error"]


def stats3(*a):
    """error, ECE, error under feature noise.  (With the equalised-odds gap and the noise drop every model
    trades off against the others, so almost nothing is population-dominated and retention is vacuous.)"""
    r = cs.stats(*a)
    return np.stack([r[:, 0], r[:, 2], r[:, 0] + r[:, 3]], axis=1)


def make_models():
    sc = lambda: StandardScaler()
    return {
        "LR": make_pipeline(sc(), LogisticRegression(max_iter=500)),
        "LR-strong-reg": make_pipeline(sc(), LogisticRegression(C=0.003, max_iter=500)),
        "Tree(d=2)": DecisionTreeClassifier(max_depth=2, random_state=0),
        "Tree(d=5)": DecisionTreeClassifier(max_depth=5, random_state=0),
        "Tree(d=12)": DecisionTreeClassifier(max_depth=12, random_state=0),
        "kNN(1)": make_pipeline(sc(), KNeighborsClassifier(1)),
        "kNN(5)": make_pipeline(sc(), KNeighborsClassifier(5)),
        "kNN(40)": make_pipeline(sc(), KNeighborsClassifier(40)),
        "NaiveBayes": GaussianNB(),
        "Prior(const)": DummyClassifier(strategy="prior"),
        "RF(20)": RandomForestClassifier(20, min_samples_leaf=5, random_state=0, n_jobs=1),
    }


def gen(n, rng):
    x = rng.normal(size=(n, P))
    g = (rng.random(n) < 1 / (1 + np.exp(-0.8 * x[:, 0]))).astype(int)
    eta = 0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.8 * x[:, 2] * x[:, 3] + 0.6 * x[:, 4] ** 2 - 0.6 + 0.7 * g
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return x, y, g


def one_training(n_tr, big, seed):
    """criteria of each algorithm trained on a fresh training set, evaluated on the big test set."""
    rng = np.random.default_rng(10_000 + seed)
    x, y, g = gen(n_tr, rng)
    Xb, yb, gb = big
    pr, pb, pn = rf.fit_predict_all(make_models, x, y, Xb, rng, seed, NUM)
    return stats3(yb, gb, pr, pb, pn, np.arange(len(yb)))


def one_rep(rep, n, B, big, mu_alg, p0_alg, alpha):
    rng = np.random.default_rng(rep)
    X, y, g = gen(n, rng)
    tr, te = train_test_split(np.arange(n), test_size=0.4, random_state=rep, stratify=y)
    Xb, yb, gb = big
    Xev = np.vstack([X[te], Xb])
    pr, pb, pn = rf.fit_predict_all(make_models, X[tr], y[tr], Xev, rng, 0, NUM)
    nte = len(te)
    s = lambda a, b: slice(a, b)
    point = stats3(y[te], g[te], pr[:nte], pb[:nte], pn[:nte], np.arange(nte))
    cond = stats3(yb, gb, pr[nte:], pb[nte:], pn[nte:], np.arange(len(yb)))   # trained models, population
    p0_cond = ~pareto_strict(cond)
    bootF = np.stack([stats3(y[te], g[te], pr[:nte], pb[:nte], pn[:nte], rng.integers(0, nte, nte)) for _ in range(500)])
    bootP = rf.paired_bootstrap(make_models, stats3, X, y, g, tr, te, B, seed=rep + 1, num=NUM, n_jobs=1)
    bootS = rf.repeated_splits(make_models, stats3, X, y, g, B, 0.4, seed=rep + 2, num=NUM, n_jobs=1)
    out = {}
    for k, (pt, bt) in dict(fixed=(point, bootF), paired_refit=(point, bootP), repeated_splits=(bootS.mean(0), bootS)).items():
        r = trust_confidence_set(pt, bt, alpha)
        out[k] = dict(keep=r.keep.tolist(), ret_alg=bool(r.keep[p0_alg].all()), ret_cond=bool(r.keep[p0_cond].all()),
                      size=int(r.keep.sum()), per_model_keep_p0=r.keep[p0_alg].tolist())
    out["pareto_point"] = dict(ret_alg=bool(pareto_front(point)[p0_alg].all()), ret_cond=bool(pareto_front(point)[p0_cond].all()),
                               size=int(pareto_front(point).sum()))
    out["p0_cond_size"] = int(p0_cond.sum())
    out["sd_ratio_paired_over_fixed"] = (bootP.std(0) / bootF.std(0)).mean(0).tolist()
    out["sd_ratio_splits_over_fixed"] = (bootS.std(0) / bootF.std(0)).mean(0).tolist()
    return out


def pareto_strict(c):
    """strictly dominated mask (population definition of P0: dominated iff some j is better on ALL criteria)."""
    M = len(c)
    return np.array([np.any(np.all(c < c[m], axis=1)) for m in range(M)])


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    alpha = 0.05
    n_tr = n - int(round(0.4 * n))
    t0 = time.time()
    big = gen(10000, np.random.default_rng(999))
    K = 100
    mus = np.stack(Parallel(n_jobs=2)(delayed(one_training)(n_tr, big, s) for s in range(K)))
    mu_alg = mus.mean(0)
    se_mu = mus.std(0, ddof=1) / np.sqrt(K)
    dominated = pareto_strict(mu_alg)
    p0_alg = ~dominated
    names = list(make_models())
    print("truth done", time.time() - t0, "P0:", [names[i] for i in np.where(p0_alg)[0]], flush=True)
    # margin of the closest dominance: for each dominated model, min over rivals of max_k (mu_j - mu_m)... report smallest
    # slack (min over criteria of mu_m-mu_j, best rival) to show how near the strictness boundary the truth is
    slack = []
    for m in np.where(dominated)[0]:
        slack.append(float(max(np.min(mu_alg[m] - mu_alg[j]) for j in range(len(names)) if np.all(mu_alg[j] < mu_alg[m]))))
    res = Parallel(n_jobs=2)(delayed(one_rep)(r, n, B, big, mu_alg, p0_alg, alpha) for r in range(reps))
    summ = {}
    for k in ("fixed", "paired_refit", "repeated_splits", "pareto_point"):
        ra = np.array([r[k]["ret_alg"] for r in res]); rc = np.array([r[k]["ret_cond"] for r in res])
        summ[k] = dict(retention_algorithm_P0=float(ra.mean()), retention_trained_model_P0=float(rc.mean()),
                       mc_se_alg=float(np.sqrt(ra.mean() * (1 - ra.mean()) / reps)),
                       mean_size=float(np.mean([r[k]["size"] for r in res])))
        if k != "pareto_point":
            pm = np.mean([r[k]["per_model_keep_p0"] for r in res], axis=0)
            summ[k]["per_P0_algorithm_retention"] = {names[i]: float(v) for i, v in zip(np.where(p0_alg)[0], pm)}
            summ[k]["fraction_of_models_retained"] = float(np.mean([np.mean(r[k]["keep"]) for r in res]))
    margin = {names[m]: float(max(np.min(mu_alg[m] - mu_alg[j]) for j in range(len(names)) if j != m)) for m in np.where(p0_alg)[0]}
    out = dict(P0_margin=margin, reps=reps, B_refit=B, B_fixed=500, n=n, n_train=n_tr, n_test=n - n_tr, alpha=alpha, n_big_test=10000,
               n_training_draws_for_truth=K, models=names, criteria=CRIT3,
               mu_alg=mu_alg.tolist(), mu_alg_mc_se=se_mu.tolist(), P0_algorithm=[names[i] for i in np.where(p0_alg)[0]],
               dominated_slack=slack, mean_P0_trained_size=float(np.mean([r["p0_cond_size"] for r in res])),
               summary=summ,
               sd_ratio_paired_over_fixed_mean_over_models=np.mean([r["sd_ratio_paired_over_fixed"] for r in res], axis=0).tolist(),
               sd_ratio_splits_over_fixed_mean_over_models=np.mean([r["sd_ratio_splits_over_fixed"] for r in res], axis=0).tolist(),
               seconds=time.time() - t0)
    json.dump(out, open("results/refit_sim.json", "w"), indent=1)
    print(json.dumps(summ, indent=1)); print("P0", out["P0_algorithm"], "slack", slack, "secs", out["seconds"])


if __name__ == "__main__":
    main()
