"""Case studies: TCS on Adult (sex) and German credit (age group)."""
import sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tcs import trust_confidence_set, pareto_front
warnings.filterwarnings("ignore")
CRIT = ["error", "eo_gap", "ece", "noise_drop"]


def load(name):
    if name == "adult":
        d = fetch_openml("adult", version=2, as_frame=True)
        X, y = d.data.copy(), (d.target == ">50K").astype(int).values
        g = (X["sex"] == "Female").astype(int).values
        X = X.drop(columns=["sex"])
    else:
        d = fetch_openml("credit-g", version=1, as_frame=True)
        X, y = d.data.copy(), (d.target == "bad").astype(int).values
        g = (X["age"] < 26).astype(int).values
        X = X.drop(columns=["age"])
    return X, y, g


def models(num, cat):
    def prep(scale=True):
        return ColumnTransformer([
            ("n", make_pipeline(SimpleImputer(), StandardScaler()), num),
            ("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat)])
    return {
        "LR": make_pipeline(prep(), LogisticRegression(max_iter=1000)),
        "LR-strong-reg": make_pipeline(prep(), LogisticRegression(C=0.01, max_iter=1000)),
        "Tree(d=3)": make_pipeline(prep(), DecisionTreeClassifier(max_depth=3, random_state=0)),
        "Tree(d=12)": make_pipeline(prep(), DecisionTreeClassifier(max_depth=12, random_state=0)),
        "RandomForest": make_pipeline(prep(), RandomForestClassifier(200, min_samples_leaf=3, random_state=0, n_jobs=-1)),
        "HistGB": make_pipeline(prep(), HistGradientBoostingClassifier(random_state=0)),
        "kNN(25)": make_pipeline(prep(), KNeighborsClassifier(25)),
        "NaiveBayes": make_pipeline(prep(), GaussianNB()),
        "MLP": make_pipeline(prep(), MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=0)),
    }


def stats(y, g, pred, prob, pred_noisy, idx):
    """criteria for each model on resample idx. pred/prob/pred_noisy: (n, M)."""
    yy, gg = y[idx], g[idx]
    P, Q, PN = pred[idx], prob[idx], pred_noisy[idx]
    M = P.shape[1]
    err = (P != yy[:, None]).mean(0)
    errn = (PN != yy[:, None]).mean(0)
    eo = np.zeros(M)
    for lab in (1, 0):
        m0, m1 = (yy == lab) & (gg == 0), (yy == lab) & (gg == 1)
        eo = np.maximum(eo, np.abs(P[m0].mean(0) - P[m1].mean(0)))
    bins = np.minimum((Q * 10).astype(int), 9)
    ece = np.zeros(M)
    for b in range(10):
        mask = bins == b
        cnt = mask.sum(0)
        conf = np.where(cnt > 0, (Q * mask).sum(0) / np.maximum(cnt, 1), 0)
        acc = np.where(cnt > 0, (yy[:, None] * mask).sum(0) / np.maximum(cnt, 1), 0)
        ece += cnt / len(idx) * np.abs(conf - acc)
    return np.stack([err, eo, ece, errn - err], axis=1)


def run(name, B=1000, alpha=0.05, seed=0):
    X, y, g = load(name)
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, y, g, test_size=0.4, random_state=seed, stratify=y)
    ms = models(num, cat)
    rng = np.random.default_rng(seed)
    Xn = Xte.copy()
    for c in num:   # perturb numeric features by 0.5 training-sd Gaussian noise
        Xn[c] = Xn[c] + rng.normal(0, 0.5 * Xtr[c].std(), len(Xn))
    pred, prob, predn = [], [], []
    for k, m in ms.items():
        m.fit(Xtr, ytr)
        pred.append(m.predict(Xte)); prob.append(m.predict_proba(Xte)[:, 1]); predn.append(m.predict(Xn))
    pred, prob, predn = map(np.column_stack, (pred, prob, predn))
    n = len(yte)
    point = stats(yte, gte, pred, prob, predn, np.arange(n))
    boot = np.stack([stats(yte, gte, pred, prob, predn, rng.integers(0, n, n)) for _ in range(B)])
    r = trust_confidence_set(point, boot, alpha)
    # composite score: equal-weight sum of criteria standardised by their cross-model sd at the point estimate
    sd = point.std(0)
    comp0 = (point / sd).sum(1)
    compb = (boot / sd).sum(2)
    top1 = np.bincount(compb.argmin(1), minlength=len(ms)) / B
    ranks = compb.argsort(1).argsort(1) + 1
    from tcs.core import pareto_front as pf
    ppro = np.mean([pf(b) for b in boot], axis=0)
    W = rng.dirichlet(np.ones(point.shape[1]), 5000)
    wwin = np.bincount(((point / sd) @ W.T).argmin(0), minlength=len(ms)) / len(W)
    import itertools
    subs = []
    for kk in (2, 3, 4):
        for c in itertools.combinations(range(4), kk):
            rr = trust_confidence_set(point[:, c], boot[:, :, c], alpha)
            subs.append(dict(criteria=[CRIT[i] for i in c], tcs_size=int(rr.keep.sum()),
                             pareto_size=int(rr.pareto.sum()), tcs=rr.keep.tolist()))
    out = dict(subsets=subs, pareto_prob=ppro.tolist(), weight_win=wwin.tolist(), dataset=name, n_test=int(n), n_train=int(len(ytr)), B=B, alpha=alpha, crit=float(r.crit),
               models=list(ms), criteria=CRIT, point=point.tolist(), stat=r.stat.tolist(),
               tcs=r.keep.tolist(), pareto=r.pareto.tolist(), composite=comp0.tolist(),
               top1_freq=top1.tolist(), rank_lo=np.quantile(ranks, .025, axis=0).tolist(),
               rank_hi=np.quantile(ranks, .975, axis=0).tolist(),
               boot_sd=boot.std(0).tolist())
    return out


if __name__ == "__main__":
    res = [run("adult"), run("credit-g")]
    json.dump(res, open("results/casestudy.json", "w"), indent=1)
    for o in res:
        print(o["dataset"], o["n_test"], "crit", round(o["crit"], 2))
        df = pd.DataFrame(o["point"], index=o["models"], columns=o["criteria"]).round(3)
        df["S"] = np.round(o["stat"], 2); df["TCS"] = o["tcs"]; df["Pareto"] = o["pareto"]
        df["pP"]=np.round(o["pareto_prob"],2); df["wwin"]=np.round(o["weight_win"],2); df["comp"] = np.round(o["composite"], 2); df["top1"] = o["top1_freq"]
        print(df)
