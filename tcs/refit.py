"""Refit-bootstrap and repeated-split inputs for trust confidence sets.

The fixed-model TCS in tcs.core resamples only test instances and keeps the trained
models fixed.  The functions here produce (B, M, K) arrays in which *every model is
refitted* on each replicate, so training-set and seed variability enter the spread that
trust_confidence_set turns into critical values.  See docs/refit_extension.md for the
estimand and caveats.

All schemes return an array of shape (R, M, K) of criteria (lower is better), where
replicate r uses one common training set, one common seed and one common evaluation
sample for all M models (pairing across models is what makes pairwise differences
informative).

Schemes
  paired_bootstrap   fixed original train/test split; resample train rows (with
                     replacement, size n_train) and test rows (with replacement, size n_test)
                     independently; refit all models on the train resample, evaluate on
                     the test resample.  Primary scheme.
  oob_bootstrap      bootstrap the whole dataset (size n), refit, evaluate on the
                     out-of-bag rows.  (~63.2% unique rows in training, ~36.8% OOB test.)
  repeated_splits    R random stratified train/test splits (no resampling with
                     replacement), same split and seed for every model.  Cheapest.
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed


def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _num_cols(X, num):
    if num is not None:
        return list(num)
    return list(X.columns) if hasattr(X, "columns") else list(range(X.shape[1]))


def _perturb(X, num, sd, rng):
    """Add N(0, (0.5 sd)^2) noise to numeric columns (as in experiments/casestudy.py)."""
    if hasattr(X, "iloc"):
        Xn = X.copy()
        for c in num:
            Xn[c] = Xn[c] + rng.normal(0, 0.5 * sd[c], len(Xn))
        return Xn
    Xn = X.astype(float).copy()
    Xn[:, num] = Xn[:, num] + rng.normal(0, 1, (len(Xn), len(num))) * 0.5 * sd[num]
    return Xn


def _train_sd(Xtr, num):
    if hasattr(Xtr, "iloc"):
        return {c: float(Xtr[c].std()) for c in num}
    return Xtr.std(0)


def reseed(model, seed):
    """Set every *random_state parameter to seed and every *n_jobs to 1."""
    ps = {k: seed for k in model.get_params() if k.endswith("random_state")}
    ps.update({k: 1 for k in model.get_params() if k.endswith("n_jobs")})
    return model.set_params(**ps)


def fit_predict_all(make_models, Xtr, ytr, Xev, rng, seed, num=None):
    """Fit fresh copies of all models on (Xtr, ytr); return pred, prob, pred_noisy on Xev
    as (n_ev, M) arrays.  Noise sd is estimated from the training rows."""
    ms = make_models()
    num = _num_cols(Xtr, num)
    Xn = _perturb(Xev, num, _train_sd(Xtr, num), rng)
    pred, prob, predn = [], [], []
    for m in ms.values():
        m = reseed(clone(m), seed)
        m.fit(Xtr, ytr)
        pred.append(m.predict(Xev)); prob.append(m.predict_proba(Xev)[:, 1]); predn.append(m.predict(Xn))
    return tuple(np.column_stack(a) for a in (pred, prob, predn))


def _one_paired(make_models, stats_fn, X, y, g, tr, te, seed, num):
    rng = np.random.default_rng(seed)
    itr = rng.choice(tr, len(tr), replace=True)
    ite = rng.integers(0, len(te), len(te))          # positions into te
    Xte, yte, gte = _take(X, te), y[te], g[te]
    pred, prob, predn = fit_predict_all(make_models, _take(X, itr), y[itr], Xte, rng, seed, num)
    return stats_fn(yte, gte, pred, prob, predn, ite)


def paired_bootstrap(make_models, stats_fn, X, y, g, tr, te, B, seed=0, num=None, n_jobs=2):
    ss = np.random.SeedSequence(seed).generate_state(B)
    out = Parallel(n_jobs=n_jobs)(delayed(_one_paired)(make_models, stats_fn, X, y, g, tr, te, int(s), num) for s in ss)
    return np.stack(out)


def _one_oob(make_models, stats_fn, X, y, g, seed, num):
    rng = np.random.default_rng(seed)
    n = len(y)
    itr = rng.integers(0, n, n)
    oob = np.setdiff1d(np.arange(n), itr)
    pred, prob, predn = fit_predict_all(make_models, _take(X, itr), y[itr], _take(X, oob), rng, seed, num)
    return stats_fn(y[oob], g[oob], pred, prob, predn, np.arange(len(oob)))


def oob_bootstrap(make_models, stats_fn, X, y, g, B, seed=0, num=None, n_jobs=2):
    ss = np.random.SeedSequence(seed).generate_state(B)
    out = Parallel(n_jobs=n_jobs)(delayed(_one_oob)(make_models, stats_fn, X, y, g, int(s), num) for s in ss)
    return np.stack(out)


def _one_split(make_models, stats_fn, X, y, g, test_size, seed, num):
    rng = np.random.default_rng(seed)
    itr, ite = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)
    pred, prob, predn = fit_predict_all(make_models, _take(X, itr), y[itr], _take(X, ite), rng, seed, num)
    return stats_fn(y[ite], g[ite], pred, prob, predn, np.arange(len(ite)))


def repeated_splits(make_models, stats_fn, X, y, g, R, test_size=0.4, seed=0, num=None, n_jobs=2):
    ss = np.random.SeedSequence(seed).generate_state(R)
    out = Parallel(n_jobs=n_jobs)(delayed(_one_split)(make_models, stats_fn, X, y, g, test_size, int(s), num) for s in ss)
    return np.stack(out)


def summarize(point, boot, tcs_fn, alpha=0.05, names=None, seed=0):
    """TCS plus descriptive companions from a (R, M, K) array: Pareto-membership
    probabilities and composite-rank intervals (composite = sum of criteria standardised by
    their cross-model sd at the point estimate, as in experiments/casestudy.py)."""
    from .core import pareto_front
    point = np.asarray(point, float); boot = np.asarray(boot, float)
    r = tcs_fn(point, boot, alpha)
    sd = point.std(0)
    compb = (boot / sd).sum(2)
    ranks = compb.argsort(1).argsort(1) + 1
    return dict(
        tcs=r.keep.tolist(), size=int(r.keep.sum()), stat=r.stat.tolist(), crit=float(r.crit),
        pareto=r.pareto.tolist(),
        pareto_prob=np.mean([pareto_front(b) for b in boot], axis=0).tolist(),
        rank_lo=np.quantile(ranks, .025, axis=0).tolist(), rank_hi=np.quantile(ranks, .975, axis=0).tolist(),
        top1_freq=(np.bincount(compb.argmin(1), minlength=point.shape[0]) / len(boot)).tolist(),
        boot_sd=boot.std(0).tolist(), boot_mean_minus_point=(boot.mean(0) - point).tolist(),
        point=point.tolist(),
    )
