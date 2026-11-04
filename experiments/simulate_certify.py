"""Monte-Carlo check of certified selection with known population criteria (non-smooth design).

Uses the 12-model design of simulate_nonsmooth.py (error, ECE, equalised-odds gap). W(r) is the box of
half-width r around equal weights intersected with the simplex. With DUP=1 the last model is an exact
copy of model 0, which creates ties (including ties at the best model when model 0 is best).
"""
import sys, os, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from experiments import simulate_nonsmooth as sn
from tcs.certify import certify
from tcs.preference import box_simplex_vertices, worst_regret
RADII = [0.0, 0.1, 0.25]
W0 = np.full(3, 1 / 3)
VERT = {r: box_simplex_vertices(W0, r) for r in RADII}


def one(args):
    ds, n, seed, B, alpha = args
    des = sn.make_design(ds); rng = np.random.default_rng(seed)
    x, g, y = sn.draw(n, rng); p, pred = sn.models(des, x, g)
    point = sn.crit_w(np.ones((1, n)), y, g, p, pred)[0]
    boot = sn.crit_w(rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float), y, g, p, pred)
    out = {}
    for r in RADII:
        c = certify(point, boot, VERT[r], alpha)
        out[r] = (c["choice"], float(c["U_one"][c["choice"]]), float(c["U_two"][c["choice"]]), c["U_sel"])
    return out


if __name__ == "__main__":
    R, B, alpha = int(sys.argv[1]), 300, 0.05
    designs = [int(a) for a in sys.argv[2].split(",")]
    rows = []
    with Pool(4) as pool:
        for ds in designs:
            mu = sn.population(sn.make_design(ds))
            for n in (500, 2000, 10000):
                res = pool.map(one, [(ds, n, 500000 + 1000 * ds + 13 * n + i, B, alpha) for i in range(R)])
                for r in RADII:
                    Rt = worst_regret(mu, VERT[r])
                    a = np.array([[Rt[x[r][0]], x[r][1], x[r][2], x[r][3]] for x in res])
                    row = dict(design=ds, dup=sn.DUP, n=n, r=r, R=R, best_regret=float(Rt.min()),
                               regret_mean=float(a[:, 0].mean()), regret_max=float(a[:, 0].max()))
                    for i, k in enumerate(("one", "two", "sel"), start=1):
                        row[f"cover_{k}"] = float((a[:, 0] <= a[:, i] + 1e-12).mean()); row[f"U_{k}"] = float(a[:, i].mean())
                    rows.append(row)
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    json.dump(rows, open("results/simulation_certify%s.json" % ("_dup" if sn.DUP else ""), "w"), indent=1)
