"""Ablation and robustness studies with PATTERN-DIF as the default (results/label_ablations_dif.json).

Default method (as in tcs.labeling_dif): strata = group x K (K = number of models predicting 1, strata < 50
merged within group), 'plain' (generalised Neyman, ensemble-mean proxy) allocation with a floor of 10 labels per
stratum, stratified difference estimator with lambda = 1 and the ensemble-mean proxy f_i = pi_i U_i + (1-pi_i) V_i,
linearised Wald intervals. Reference estimator: plug-in expansion estimator of simple random sampling ('srs|ht').

For each of the 11 settings (cached pools) and L in {250, 500, 1000} with L <= N/4, R replications:
  A1 stratifier   srs, group, pattern (default), binary (group x agree/disagree), joint3 (group x pattern of the 3 models
                  agreeing most with the ensemble), kp2 / kp3 (group x K x proxy bins), pattern_resid (residual allocation);
                  each sample is analysed with the plug-in (HT, lambda = 0) and the difference estimator (lambda = 1).
  A2 proxy        allocation proxy ens | single0 | single7 | const | shuffle | noise | flip | oracle (true y, allocation
                  only) | prop (no proxy); estimators on the same sample: HT (proxy in allocation only), DIF with the same
                  proxy in the estimator ('dif', proxy in both), DIF with the ensemble proxy in the estimator ('difens',
                  alternative proxy in the allocation only) and tuned lambda (per stratum and target, lam in [0, 1]).
  A3 models M     random subsets of size 2, 3, 5 (3 subsets each, the seed of label_ablations.py) and all 9; SRS vs PATTERN,
                  HT and DIF.
  A4 inference    pattern|dif and srs|dif: Wald | Wald + floor 'zero' | Wald + floor 'max' | t (Satterthwaite) |
                  t with the zero floor | stratified bootstrap percentile; plug-in HT as reference.
  A5 EO ranking   plug-in EO_m = max(|dTPR_m|, |dFPR_m|) from DIF totals vs SRS (HT) vs PATTERN-HT, also gap-targeted allocation.
Targets whose numerator vanishes identically in the pool (exact for every design) are excluded from every summary
(as in label_efficiency_dif.py); only law school is affected.
Usage: python experiments/label_ablations_dif.py R [keys] [B] [out]     |     python experiments/label_ablations_dif.py report
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, zlib, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import t as tdist
sys.path.insert(0, ".")
from tcs.labeling import total_columns, Estimands, estimate, draw
from tcs.labeling_baselines import VecEstimands, ppi_estimate
from tcs.labeling_dif import DifContext
from tcs.labeling_variants import (strata_kcount, strata_binary, strata_joint, strata_kproxy, top_agreement_models, proxies,
                                   eo_gap, GapEstimands, kendall_tau)
from tcs.labeling_variants_dif import allocate_dif, ht_totals, dif_inference, dif_bootstrap
from experiments.label_ablations import load, config, SETTINGS, BUDGETS, SUBSET_SIZES, N_SUBSETS

OUT = "results/label_ablations_dif.json"
Z = 1.959964
CLIP = 1e4
ALT = ["single0", "single7", "const", "shuffle", "noise", "flip"]
EO_ENTRIES = ["srs|ht", "srs|dif", "group|ht", "group|dif", "pattern|ht", "pattern|dif", "pattern_gap|ht", "pattern_gap|dif",
              "oracle|ht", "oracle|difens"]
FULL = ["pattern|dif", "srs|dif"]            # entries with all interval variants


def live_targets(pred, y, pairs):
    live = []
    for (m, j) in pairs:
        dif = pred[:, m] != pred[:, j]
        live += [bool((dif & (y == 1)).any()), bool((dif & (y == 0)).any()), bool(dif.any())]
    return np.array(live)


def build(key, L):
    pred, p, g, y = load(key); N, M = pred.shape
    base = config(pred, p, g, y); base["live"] = live_targets(pred, y, base["est"].pairs)
    base["kinds"] = np.array([k for (k, _, _) in base["est"].labels])
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    px = proxies(p, g, y, rng); pi = base["pi"]; px["ens"] = pi
    U, V, est = base["U"], base["V"], base["est"]
    one = np.zeros(N, int); sk = strata_kcount(pred, g); top3 = top_agreement_models(pred, p, 3)
    S = {"srs": one, "group": g.copy(), "pattern": sk, "binary": strata_binary(pred, g),
         "joint3": strata_joint(pred, g, top3), "kp2": strata_kproxy(pred, g, pi, 2), "kp3": strata_kproxy(pred, g, pi, 3)}
    D = {}      # design name -> dict(sname, strata, ids, Nh, nh, cfg, ...)
    E = {}      # entry label -> (design, kind, estimator-proxy key, cfg)

    def add(name, sname, aprox="ens", kind="plain", est_=est, cfg="main", strata=None, entries=(), Ucfg=None, Vcfg=None, pcfg=None):
        st = S[sname] if strata is None else strata
        Uc = U if Ucfg is None else Ucfg; Vc = V if Vcfg is None else Vcfg
        pr = (px[aprox] if pcfg is None else pcfg)
        ids, Nh, nh, mn = allocate_dif(st, Uc, Vc, pr, est_, L, 10, kind)
        D[name] = dict(sname=sname, strata=st, ids=ids, Nh=Nh, nh=nh, min_n=mn, H=len(ids), cfg=cfg, n=int(nh.sum()))
        for lab, k, ep in entries:
            E[f"{name}|{lab}"] = (name, k, ep, cfg)

    add("srs", "srs", kind="prop", entries=[("ht", "ht", "ens"), ("dif", "dif", "ens"), ("tuned", "tuned", "ens")])
    add("group", "group", entries=[("ht", "ht", "ens"), ("dif", "dif", "ens")])
    add("pattern", "pattern", entries=[("ht", "ht", "ens"), ("dif", "dif", "ens"), ("tuned", "tuned", "ens")])
    for nm in ("binary", "joint3", "kp2", "kp3"):
        add(nm, nm, entries=[("ht", "ht", "ens"), ("dif", "dif", "ens")])
    add("pattern_resid", "pattern", kind="resid", entries=[("ht", "ht", "ens"), ("dif", "dif", "ens")])
    for nm in ALT:
        add(nm, "pattern", aprox=nm, entries=[("ht", "ht", "ens"), ("dif", "dif", nm), ("difens", "dif", "ens"), ("tuned", "tuned", nm)])
    add("oracle", "pattern", aprox="oracle", entries=[("ht", "ht", "ens"), ("difens", "dif", "ens")])
    add("prop", "pattern", kind="prop", entries=[("ht", "ht", "ens"), ("dif", "dif", "ens")])
    gap_est = GapEstimands(base["names"], M)
    add("pattern_gap", "pattern", est_=gap_est, entries=[("ht", "ht", "ens"), ("dif", "dif", "ens")])
    # A3 subsets (same seed and procedure as label_ablations.py)
    srng = np.random.default_rng(20261104); subsets = []
    for size in SUBSET_SIZES:
        seen = set()
        while len([s for s in subsets if len(s) == size]) < N_SUBSETS:
            s = tuple(sorted(srng.choice(M, size, replace=False).tolist()))
            if s not in seen: seen.add(s); subsets.append(s)
    SUB = {}
    for s in subsets:
        cname = "M" + "-".join(map(str, s)); ms = list(s)
        c = config(pred[:, ms], p[:, ms], g, y); c["models"] = ms
        c["live"] = live_targets(pred[:, ms], y, c["est"].pairs); c["kinds"] = np.array([k for (k, _, _) in c["est"].labels])
        SUB[cname] = c
        px[cname] = c["pi"]; S[f"{cname}|srs"] = one; S[f"{cname}|pattern"] = strata_kcount(pred[:, ms], g)
        for sn, kd in (("srs", "prop"), ("pattern", "plain")):
            add(f"{cname}|{sn}", f"{cname}|{sn}", aprox=cname, kind=kd, est_=c["est"], cfg=cname, Ucfg=c["U"], Vcfg=c["V"],
                entries=[("ht", "ht", cname), ("dif", "dif", cname)])
    return dict(base=base, sub=SUB, D=D, E=E, px=px, y=y, g=g, top3=top3, M=M, N=N, S=S)


def cfg_of(S, cfg):
    return S["base"] if cfg == "main" else S["sub"][cfg]


def run_entry(kind, d, cfgd, ctx, y, idx, ep, ve_p, rng_b=None, full=False):
    """Returns dict(theta, se, T [, extras])."""
    U, V, est = cfgd["U"], cfgd["V"], cfgd["est"]
    if kind == "ht":
        th, se, lo, hi = estimate(d["strata"], d["ids"], d["Nh"], d["nh"], U, V, y[idx], idx, est)
        return dict(theta=th, se=se)
    if kind == "tuned":
        th, se, lo, hi = ppi_estimate(d["strata"], d["ids"], d["Nh"], d["nh"], idx, y, U, V, ep, ve_p, tune=True)
        return dict(theta=th, se=se)
    r = dif_inference(ctx, d["Nh"], d["nh"], U, V, y[idx], idx, est, ep)
    if r is None:
        x = np.full(len(cfgd["truth"]), np.nan); return dict(theta=x, se=x, T=None)
    out = dict(theta=r["theta"], se=r["se_none"], T=r["T"])
    if full:
        for k in ("se_zero", "se_max", "df_none", "df_zero"): out[k] = r[k]
        out["lo_b"], out["hi_b"] = dif_bootstrap(ctx, d["Nh"], U, V, y[idx], idx, est, rng_b, B=full)
    return out


def mse_eff(a, b, truth, live):
    """Per live target MSE(b)/MSE(a) (= efficiency of a relative to reference b) on replications where both are finite."""
    both = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if both.sum() < 10: return None
    ma = ((a[both] - truth) ** 2).mean(0)[live]; mb = ((b[both] - truth) ** 2).mean(0)[live]
    r = np.clip(mb / np.maximum(ma, 1e-300), 1 / CLIP, CLIP)
    return np.where((ma == 0) & (mb == 0), np.nan, r)


def cover(lo, hi, truth, live, ok):
    tol = 1e-9 * (1 + np.abs(truth))
    c = ((lo[ok] - tol <= truth) & (truth <= hi[ok] + tol)).mean(0)[live]
    return c


def summarise(rec, truth, live, kinds, ref="srs|ht", ref_def=None, full=()):
    """rec: label -> dict of arrays (R, nt). Returns label -> summary dict."""
    out = {}
    for lab, r in rec.items():
        A, Sd = r["theta"], r["se"]
        if not live.any(): out[lab] = dict(fail=0.0, no_live=True); continue
        ok = np.isfinite(A).all(1) & np.isfinite(Sd).all(1)
        s = dict(fail=float(1 - ok.mean()))
        if ok.sum() < 10: out[lab] = s; continue
        c = cover(A - Z * Sd, A + Z * Sd, truth, live, ok)
        s.update(cover=float(c.mean()), cover_min=float(c.min()), cover_p05=float(np.quantile(c, 0.05)),
                 width=float(np.median((2 * Z * Sd[ok]).mean(0)[live])),
                 degenerate=float(np.mean(Sd[ok][:, live] == 0)),
                 bias2share=float(np.median(((A[ok].mean(0) - truth) ** 2 / np.maximum(((A[ok] - truth) ** 2).mean(0), 1e-30))[live])))
        if lab != ref and ref in rec:
            e = mse_eff(A, rec[ref]["theta"], truth, live)
            if e is not None:
                s["eff_vs_srs"] = float(np.nanmedian(e))
                lv = np.flatnonzero(live)
                for k in ("dTPR", "dFPR", "dERR"):
                    s[f"eff_vs_srs_{k}"] = float(np.nanmedian(e[kinds[lv] == k]))
        if ref_def and lab != ref_def and ref_def in rec:
            e = mse_eff(A, rec[ref_def]["theta"], truth, live)
            if e is not None:
                s["eff_vs_def"] = float(np.nanmedian(e)); s["eff_vs_def_gmean"] = float(np.exp(np.nanmean(np.log(e))))
                s["win_share_targets"] = float(np.nanmean(e > 1))
        if lab in full:
            w0 = float(np.mean((2 * Z * Sd[ok])[:, live]))
            v = {}
            for nm, se_ in (("wald", Sd), ("zero", r["se_zero"]), ("max", r["se_max"])):
                cc = cover(A - Z * se_, A + Z * se_, truth, live, ok)
                v[nm] = (float(cc.mean()), float(cc.min()), float(np.mean((2 * Z * se_[ok])[:, live])) / w0)
            for nm, se_, df in (("t", Sd, r["df_none"]), ("t_zero", r["se_zero"], r["df_zero"])):
                q = np.where(np.isfinite(df), tdist.ppf(0.975, np.where(np.isfinite(df), df, 1e6)), Z)
                cc = cover(A - q * se_, A + q * se_, truth, live, ok)
                v[nm] = (float(cc.mean()), float(cc.min()), float(np.mean((2 * q * se_)[ok][:, live])) / w0)
            cb = cover(r["lo_b"], r["hi_b"], truth, live, ok)
            v["boot"] = (float(cb.mean()), float(cb.min()), float(np.mean((r["hi_b"] - r["lo_b"])[ok][:, live])) / w0)
            s["inf"] = {k: dict(cover=a, cover_min=b, width=c_) for k, (a, b, c_) in v.items()}
            w = int(np.flatnonzero(live)[int(np.argmin(c))])
            sd = A[ok].std(0, ddof=1)
            s["worst_target"] = str(kinds[w]); s["worst_truth"] = float(truth[w]); s["worst_bias_over_sd"] = float((A[ok][:, w].mean() - truth[w]) / max(sd[w], 1e-300))
            s["worst_se_over_sd"] = float(Sd[ok][:, w].mean() / max(sd[w], 1e-300))
            s["se_over_sd_median"] = float(np.median((Sd[ok].mean(0) / np.maximum(sd, 1e-300))[live]))
        out[lab] = s
    return out


def run_cell(args):
    key, L, R, B = args
    t0 = time.time()
    S = build(key, L); base, D, E, y = S["base"], S["D"], S["E"], S["y"]
    crc = zlib.crc32(key.encode()) % 9973
    names = list(D)
    ctxs = {}; ves = {}

    def get_ctx(d, ep_key, cfg):
        k = (d["sname"], ep_key, cfg)
        if k not in ctxs:
            c = cfg_of(S, cfg)
            ctxs[k] = DifContext(d["strata"], d["ids"], c["U"], c["V"], S["px"][ep_key], c["est"])
        return ctxs[k]

    for cfg in ["main"] + list(S["sub"]):
        c = cfg_of(S, cfg); ves[cfg] = VecEstimands(c["est"])
    rec = {lab: {} for lab in E}; eo = {lab: [] for lab in EO_ENTRIES}
    by_design = {}
    for lab, (dn, kd, ep, cfg) in E.items(): by_design.setdefault(dn, []).append((lab, kd, ep, cfg))
    for r in range(R):
        for di, n in enumerate(names):
            d = D[n]; rng = np.random.default_rng([crc, L, r, di])
            idx = draw(d["strata"], d["ids"], d["nh"], rng)
            for (lab, kd, ep, cfg) in by_design[n]:
                cfgd = cfg_of(S, cfg); full = (B if lab in FULL else False)
                ctx = get_ctx(d, ep, cfg) if kd == "dif" else None
                o = run_entry(kd, d, cfgd, ctx, y, idx, S["px"][ep], ves[cfg], np.random.default_rng([crc, L, r, di, 1]), full)
                for k, v in o.items():
                    if k == "T": continue
                    rec[lab].setdefault(k, []).append(v)
                if lab in eo:
                    if kd == "ht": T = ht_totals(get_ctx(d, "ens", "main"), d["Nh"], idx, y[idx], base["U"], base["V"])
                    else: T = o.get("T")
                    eo[lab].append(np.full(S["M"], np.nan) if T is None else eo_gap(T, base["est"].ix, base["M"]))
    rec = {lab: {k: np.array(v) for k, v in dd.items()} for lab, dd in rec.items()}
    out = dict(dataset=key, L=L, R=R, B=B, N=S["N"], top3=S["top3"], nonunan=base["nonunan"], pairdis=base["pairdis"],
               n_live=int(base["live"].sum()), n_targets=int(len(base["live"])),
               design={n: dict(H=D[n]["H"], min_n=D[n]["min_n"], n=D[n]["n"]) for n in names if D[n]["cfg"] == "main"})
    mainrec = {lab: v for lab, v in rec.items() if E[lab][3] == "main"}
    out["ent"] = summarise(mainrec, base["truth"], base["live"], base["kinds"], "srs|ht", "pattern|dif", FULL)
    out["subsets"] = {}
    for cfg, c in S["sub"].items():
        sr = {lab.split("|", 1)[1]: v for lab, v in rec.items() if E[lab][3] == cfg}
        sm = summarise(sr, c["truth"], c["live"], c["kinds"], "srs|ht", "pattern|dif")
        out["subsets"][cfg] = dict(models=c["models"], M=len(c["models"]), nonunan=c["nonunan"], pairdis=c["pairdis"],
                                   n_live=int(c["live"].sum()), ent=sm)
    # A5
    eo_true = eo_gap(base["Ttrue"], base["est"].ix, base["M"])
    srt = np.sort(eo_true); bestset = np.flatnonzero(eo_true <= srt[0] + 1e-12)
    top2 = np.flatnonzero(eo_true <= srt[min(1, len(srt) - 1)] + 1e-12)
    out["eo_truth"] = eo_true.tolist(); out["eo_n_best"] = int(len(bestset))
    out["eo_margin"] = float(srt[len(bestset)] - srt[0]) if len(bestset) < len(srt) else 0.0
    out["eo"] = {}
    for n in EO_ENTRIES:
        Ee = np.array(eo[n]); ok = np.isfinite(Ee).all(1); Ee = Ee[ok]
        out["eo"][n] = dict(fail=float(1 - ok.mean()), rmse=float(np.sqrt(((Ee - eo_true) ** 2).mean(0)).mean()),
                            bias=float((Ee - eo_true).mean()), p_best=float(np.mean(np.isin(Ee.argmin(1), bestset))),
                            p_best_top2=float(np.mean(np.isin(Ee.argmin(1), top2))),
                            tau=float(np.mean([kendall_tau(e, eo_true) for e in Ee])))
    Es = np.array(eo["srs|ht"])
    for n in EO_ENTRIES:
        if n == "srs|ht": continue
        Ed = np.array(eo[n]); both = np.isfinite(Es).all(1) & np.isfinite(Ed).all(1)
        ms = ((Es[both] - eo_true) ** 2).mean(0); md = ((Ed[both] - eo_true) ** 2).mean(0)
        z0 = (ms < 1e-20) & (md < 1e-20)                              # both exact (up to rounding): no information
        rr_ = np.clip(ms / np.maximum(md, 1e-300), 1 / CLIP, CLIP)
        out["eo"][n]["eff_vs_srs"] = float(np.nanmedian(np.where(z0, np.nan, rr_))) if (~z0).any() else None
    nl = int(base["live"].sum()); mc = np.random.default_rng(1).binomial(R, 0.95, size=(2000, nl)).min(1) / R
    out["mc_min_ref"] = dict(mean=float(mc.mean()), q05=float(np.quantile(mc, 0.05)), n_live=nl)
    out["seconds"] = time.time() - t0
    print(key, L, f"{out['seconds']:.0f}s", "pattern|dif cover_min", round(out["ent"]["pattern|dif"]["cover_min"], 3), flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "report":
        from experiments.label_ablations_dif_report import report
        report(*sys.argv[2:]); sys.exit()
    R = int(sys.argv[1]); keys = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else SETTINGS
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 999
    out_path = sys.argv[4] if len(sys.argv) > 4 else OUT
    tasks = []
    for key in keys:
        N = len(np.load(f"results/pool_cache/{key}.npz")["y"])
        tasks += [(key, L, R, B) for L in BUDGETS if L <= N / 4]
    tasks.sort(key=lambda a: -a[1])
    with Pool(3) as pool:
        rows = pool.map(run_cell, tasks, chunksize=1)
    order = {k: i for i, k in enumerate(SETTINGS)}
    rows.sort(key=lambda r: (order.get(r["dataset"], 99), r["L"]))
    json.dump(rows, open(out_path, "w"), indent=1)
