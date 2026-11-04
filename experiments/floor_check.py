"""Independent check of the proxy variance floor on synthetic pools that were not used to design it.

Pools come from the generative model of simulate_nonsmooth.py (covariates, a binary group, logistic
labels, fixed scoring rules). Nine models per pool, pool size N, budgets L. Compares Wald intervals with
the default linearisation SE and with the floor SE (labeling_variants.estimate_full with sigma).
"""
import sys, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from experiments import simulate_nonsmooth as sn
from tcs.labeling import total_columns, Estimands, make_strata, neyman_allocation, draw
from tcs.labeling_variants import estimate_full, proxy_cov
P = {}


def build(design, N, seed):
    des = sn.make_design(design); rng = np.random.default_rng(seed)
    x, g, y = sn.draw(N, rng); p, pred = sn.models(des, x, g)
    p, pred = p[:, :9], pred[:, :9]
    U, V, names = total_columns(pred, g); est = Estimands(names, 9, N)
    truth = est.value((y[:, None] * U + (1 - y[:, None]) * V).sum(0))
    st = make_strata(pred, g); pi = p.mean(1)
    return dict(y=y.astype(float), U=U, V=V, est=est, truth=truth, st=st, pi=pi, pred=pred, g=g)


def one(args):
    key, L, seed = args
    d = P[key]; ids, Nh, nh = P[(key, L)]; rng = np.random.default_rng(seed)
    idx = draw(d["st"], ids, nh, rng)
    th, se, df, T, sef = estimate_full(d["st"], ids, Nh, nh, d["U"], d["V"], d["y"][idx], idx, d["est"], sigma=P[(key, "sig")])
    return th, se, sef


if __name__ == "__main__":
    R = int(sys.argv[1]); rows = []
    for design in (3, 4, 8, 10):
        for N in (2000, 8000):
            key = (design, N); d = build(design, N, 1000 + design + N); P[key] = d
            ids0 = np.unique(d["st"])
            P[(key, "sig")] = proxy_cov(d["st"], ids0, d["U"], d["V"], d["pi"])
            for L in (250, 500):
                if L > N / 4: continue
                P[(key, L)] = neyman_allocation(d["st"], d["U"], d["V"], d["pi"], d["est"], L)
                with Pool(4) as pool:
                    res = pool.map(one, [(key, L, 7 * design + N + 13 * L + r) for r in range(R)])
                th = np.array([r[0] for r in res]); se = np.array([r[1] for r in res]); sef = np.array([r[2] for r in res])
                ok = np.isfinite(th).all(1); tr = d["truth"]; nz = np.abs(tr) > 1e-12
                cw = (np.abs(th[ok] - tr) <= 1.96 * se[ok]).mean(0)[nz]; cf = (np.abs(th[ok] - tr) <= 1.96 * sef[ok]).mean(0)[nz]
                dis = float(np.mean([(d["pred"][:, a] != d["pred"][:, b]).mean() for a in range(9) for b in range(a + 1, 9)]))
                row = dict(design=design, N=N, L=L, R=R, disagree=dis, cover_wald_mean=float(cw.mean()), cover_wald_worst=float(cw.min()),
                           cover_floor_mean=float(cf.mean()), cover_floor_worst=float(cf.min()),
                           width_ratio_median=float(np.median((sef[ok] / np.where(se[ok] > 0, se[ok], np.nan))[:, nz][np.isfinite((sef[ok] / np.where(se[ok] > 0, se[ok], np.nan))[:, nz])])),
                           zero_se_share=float(np.mean(se[ok][:, nz] == 0)), fail=float(1 - ok.mean()))
                rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    json.dump(rows, open("results/floor_check.json", "w"), indent=1)
