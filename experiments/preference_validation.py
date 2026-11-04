"""Validate the preference-robust TCS against large held-out references on twelve data settings.

W(r) = {w in simplex : |w_k - 1/4| <= r}; r = 0 is a single equal-weight score, r = 0.75 the full
simplex (ordinary Pareto TCS). Criteria are in their natural units (all probabilities).
Pools are cached in CACHE so that models are trained once per setting.
"""
import sys, os, json, zlib, warnings
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from tcs import pareto_front
from tcs.preference import box_simplex_vertices, preference_confidence_set, w_dominated
from experiments import realdata_validation as rv
from experiments import realdata_multi as rm
warnings.filterwarnings("ignore")
CACHE = os.environ.get("POOL_CACHE", "results/pool_cache")
SETTINGS = ["adult-sex", "adult-race", "bank", "credit-card", "law-school", "magic", "eeg", "phoneme",
            "spambase", "churn", "electricity", "nomao"]
RADII = [0.0, 0.05, 0.10, 0.15, 0.25, 0.75]
W0 = np.full(4, 0.25)
VERT = {r: box_simplex_vertices(W0, r) for r in RADII}
POOLS = {}


def get_pool(key):
    os.makedirs(CACHE, exist_ok=True); f = f"{CACHE}/{key}.npz"
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); return {k: z[k] for k in z.files}
    d = rv.prepare(key) if key in ("adult-sex", "adult-race", "bank") else rm.prepare(key)
    d = {k: np.asarray(v) for k, v in d.items()}
    np.savez_compressed(f, **d); return d


def one(args):
    key, n, seed, B, alpha = args
    d = POOLS[key]; rng = np.random.default_rng(seed)
    idx = rng.choice(len(d["y"]), n, replace=False)
    y, g, p, pred, predn = d["y"][idx], d["g"][idx], d["p"][idx], d["pred"][idx], d["predn"][idx]
    point = rv.crit4(np.ones((1, n)), y, g, p, pred, predn)[0]
    boot = rv.crit4(rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float), y, g, p, pred, predn)
    out = {}
    for r in RADII:
        V = VERT[r]
        res = preference_confidence_set(point, boot, V, alpha)
        out[r] = dict(tcs=res.keep, naive=~w_dominated(point, V))
    out["win"] = int((point @ W0).argmin())
    return out


if __name__ == "__main__":
    R, B, alpha = int(sys.argv[1]), 300, 0.05
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else SETTINGS
    rows = []
    for key in keys:
        d = get_pool(key); POOLS[key] = d; N = len(d["y"])
        ref = rv.crit4(np.ones((1, N)), d["y"], d["g"], d["p"], d["pred"], d["predn"])[0]
        refwin = int((ref @ W0).argmin())
        with Pool(4) as pool:
            for n in (500, 2000):
                if n > N / 3: continue
                res = pool.map(one, [(key, n, zlib.crc32(key.encode()) % 99991 + 7 * n + i, B, alpha) for i in range(R)])
                for r in RADII:
                    P0 = ~w_dominated(ref, VERT[r]); dom = ~P0
                    tk = np.array([x[r]["tcs"] for x in res]); nk = np.array([x[r]["naive"] for x in res])
                    row = dict(dataset=key, n=n, r=r, R=R, pool=int(N), M=int(len(P0)), P_ref=int(P0.sum()),
                               tcs_retain=float(tk[:, P0].all(1).mean()), tcs_size=float(tk.sum(1).mean()),
                               tcs_excl=float((~tk[:, dom]).mean()) if dom.any() else None,
                               naive_retain=float(nk[:, P0].all(1).mean()), naive_size=float(nk.sum(1).mean()),
                               tcs_contains_refwin=float(tk[:, refwin].mean()))
                    if r == 0: row["winner_correct"] = float(np.mean([x["win"] == refwin for x in res]))
                    rows.append(row)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in ("dataset", "n", "r", "P_ref", "tcs_retain", "tcs_size", "naive_retain", "naive_size", "tcs_excl", "winner_correct")}, flush=True)
        json.dump(rows, open("results/preference_validation.json", "w"), indent=1)
