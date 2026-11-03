"""Power / validity comparison: original TCS, step-down, two-stage preselection, margin, naive Pareto."""
import numpy as np, json, sys
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs import trust_confidence_set
from tcs.stepdown import trust_confidence_set_sd, eps_nondominated
from experiments.simulate import pop as pop_A, K

EPS_LIST = (0.01, 0.02)


def pop_B(delta):
    """Tie design: 6 models tied on criterion 1 and strictly ordered on 2-3 (all in P0, every pair is
    tied on one criterion: the least-favourable configuration), plus 2 models strictly dominated by model 0."""
    base = np.array([[0.2, 0.10 + 0.10 * i, 0.10 + 0.10 * i] for i in range(6)])
    d = np.array([[0.2 + delta, 0.10 + delta, 0.10 + delta], [0.2 + 2 * delta, 0.10 + 2 * delta, 0.10 + 2 * delta]])
    return np.vstack([base, d])


POPS = {"A": pop_A, "B": pop_B}


def variants(est, boot, alpha):
    r0 = trust_confidence_set(est, boot, alpha)
    out = {"orig": r0.keep, "naive": r0.pareto}
    out["stepdown"] = trust_confidence_set_sd(est, boot, alpha).keep
    out["sd_pre"] = trust_confidence_set_sd(est, boot, alpha, preselect=0.01).keep
    for e in EPS_LIST:
        out[f"sd_eps{e}"] = trust_confidence_set_sd(est, boot, alpha, eps=e).keep
    return out


def one(args):
    scen, n, delta, rho, seed, B, alpha = args
    rng = np.random.default_rng(seed)
    mu = POPS[scen](delta)
    M = mu.shape[0]
    Sig_k = rho * np.ones((K, K)) + (1 - rho) * np.eye(K)
    Sig_m = 0.5 * np.ones((M, M)) + 0.5 * np.eye(M)
    L = np.linalg.cholesky(np.kron(Sig_m, Sig_k))
    x = mu.reshape(1, M, K) + 0.15 * (rng.standard_normal((n, M * K)) @ L.T).reshape(n, M, K)
    est = x.mean(0)
    idx = rng.integers(0, n, size=(B, n))
    boot = np.stack([x[i].mean(0) for i in idx])
    v = variants(est, boot, alpha)
    P0 = eps_nondominated(mu, 0.0)
    res = {}
    for name, keep in v.items():
        e = float(name.split("eps")[1]) if "eps" in name else 0.0
        Pe = eps_nondominated(mu, e)
        res[name] = dict(cover_P0=bool(keep[P0].all()), cover_Peps=bool(keep[Pe].all()),
                         excl_P0c=float((~keep[~P0]).mean()), excl_Pepsc=float((~keep[~Pe]).mean()) if (~Pe).any() else None,
                         size=int(keep.sum()))
    res["sd_subset_orig"] = bool(np.all(~v["stepdown"] | v["orig"]))
    return res


if __name__ == "__main__":
    R, B, alpha = 300, 300, 0.05
    grid = [("A", n, d, rho) for n in (200, 500, 2000) for d in (0.02, 0.05) for rho in (0.0, 0.5)]
    grid += [("B", n, 0.03, rho) for n in (200, 500, 2000) for rho in (0.0, 0.5)]
    out = []
    with Pool(2) as p:
        for gi, (s, n, d, rho) in enumerate(grid):
            res = p.map(one, [(s, n, d, rho, 7000 + 1000 * gi + r, B, alpha) for r in range(R)])
            row = dict(scenario=s, n=n, delta=d, rho=rho, R=R, B=B, alpha=alpha,
                       P0=eps_nondominated(POPS[s](d)).tolist(),
                       Peps={str(e): eps_nondominated(POPS[s](d), e).tolist() for e in EPS_LIST},
                       sd_subset_orig=float(np.mean([r["sd_subset_orig"] for r in res])))
            for name in res[0]:
                if name == "sd_subset_orig":
                    continue
                row[name] = {k: (None if res[0][name][k] is None else float(np.mean([r[name][k] for r in res])))
                             for k in res[0][name]}
            out.append(row)
            print(s, n, d, rho, {k: (round(v["cover_Peps"], 3), round(v["excl_P0c"], 2)) for k, v in row.items() if isinstance(v, dict) and "cover_Peps" in v}, flush=True)
    json.dump(out, open("results/simulation_power.json", "w"), indent=1)
