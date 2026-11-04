"""Calibration and discrimination of the ensemble-mean proxy pi_i = mean_m p_mi on each cached pool.

Writes results/proxy_calibration.json: prevalence, 10-bin calibration error, largest gap between mean proxy and
positive rate over proxy quintiles, and the area under the ROC curve of the proxy. Usage: python experiments/proxy_calibration.py
"""
import glob, json, os
import numpy as np
from scipy.stats import rankdata

rows = []
for f in sorted(glob.glob("results/pool_cache/*.npz")):
    k = os.path.basename(f)[:-4]
    if k == "nomao":
        continue
    z = np.load(f, allow_pickle=True)
    pi, y = z["p"].mean(1), z["y"].astype(float)
    bins = np.minimum((pi * 10).astype(int), 9)
    ece = sum((bins == b).mean() * abs(pi[bins == b].mean() - y[bins == b].mean()) for b in range(10) if (bins == b).any())
    qb = np.digitize(pi, np.quantile(pi, [.2, .4, .6, .8]))
    worst = max(abs(pi[qb == i].mean() - y[qb == i].mean()) for i in range(5))
    r = rankdata(pi); n1 = y.sum(); n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    rows.append(dict(dataset=k, N=int(len(y)), prevalence=round(float(y.mean()), 3), ece10=round(float(ece), 3),
                     worst_quintile_gap=round(float(worst), 3), auc=round(float(auc), 3)))
json.dump(rows, open("results/proxy_calibration.json", "w"), indent=1)
for r in rows:
    print(r)
