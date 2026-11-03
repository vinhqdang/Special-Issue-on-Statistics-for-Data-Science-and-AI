# Step-down trust confidence sets and a practical-equivalence margin

Code: `tcs/stepdown.py`, `experiments/simulate_power.py`, `experiments/casestudy_power.py`.
Results: `results/simulation_power.json`, `results/casestudy_power.json`.

## Targets

With `mu[m,k]` the population criterion values (smaller is better) and a margin vector `eps_k >= 0`:
model `j` eps-dominates `m` if `mu[m,k] - mu[j,k] > eps_k` for every `k`.
`P0(eps)` is the set of models eps-dominated by nobody. `P0(0)` is the strict-dominance set `P0` of the
manuscript, and `P0 ⊆ P0(eps)`. A positive margin therefore changes the target to a larger set.
It does not give more power for the target `P0`.

## Procedures

Statistic: `t[m,j,k] = (d[m,j,k] - eps_k)/se`, `S_m = max_{j≠m} min_k t[m,j,k]`.
Centred bootstrap statistics `t*[m,j,k] = (d* - d)/se` do not involve `eps`.

* **Step-down.** Set `A_1` = all models. At step `s`, `q_s` is the `(1-alpha)` quantile of
  `max over {m in A_s, j≠m, k} t*`. Set `A_{s+1} = {m in A_s : S_m <= q_s}` and stop when nothing is removed.
  Rivals `j` always range over all models, because an excluded model is still a valid population dominator.
* **Two-stage preselection (`sd_pre`, "sharper" critical value).** Triples with `t0 > c_pre` are dropped from the
  critical-value family, where `c_pre` is the `(1-beta)` quantile of the full max (`beta = 0.01`). The main
  quantile uses level `1-(alpha-beta)`.

## Proposition (asymptotic)

Under the conditions of Proposition 1 (joint bootstrap consistency of the studentised differences,
continuity of the limit at its quantile), for either variant and any `eps`:
`limsup Pr(P0(eps) ⊄ TCS) <= alpha`. Every excluded model is worse than some rival by more than `eps_k` on every `k`.

**Proof sketch.** Let `m ∈ P0(eps)`. For each rival `j` some `k(j)` has `delta_{mjk(j)} <= eps_{k(j)}`, so
`t[m,j,k(j)] <= z[m,j,k(j)]` (the studentised centred error), hence `S_m <= Z := max over {m' ∈ P0(eps), j, k: delta <= eps} z`.
*Step-down:* consider the first step `s` at which some `m ∈ P0(eps)` is excluded. Then `P0(eps) ⊆ A_s`, so the family
at step `s` contains the family `{m' ∈ P0(eps)}`, and `q_s >= q_{P0(eps)}`, the `(1-alpha)` quantile of the max of the
bootstrap analogue of `Z`. Exclusion requires `Z > q_s >= q_{P0(eps)}`, which has limiting probability `alpha`.
Models excluded only after a first false exclusion are irrelevant, since the event of interest has already happened. This is the
Romano–Wolf argument with hypotheses indexed by `m`. The one-step TCS is the case `s = 1`.
*Preselection:* on `{Z <= c_pre}` (probability `>= 1-beta`) every triple with `delta <= eps` survives the filter,
so the filtered family contains the relevant triples, and the level-`(alpha-beta)` quantile gives
error `<= (alpha-beta) + beta = alpha` by a union bound. This is Bonferroni-type, in the spirit of
Romano–Shaikh–Wolf two-stage methods. I argued it, I did not write a rigorous proof; treat it as argued, not proved.

## Caveats

* Asymptotic only. The simulation uses plain mean criteria. Non-smooth criteria (ECE binning, max in the EO gap) are not tested.
* Step-down and preselection do not remove the main source of conservativeness. The max over criteria in the critical
  value is retained, and the critical value is dominated by the many null/tied triples. The two variants only trim the family a little.
* The margin shrinks `S_m` and so excludes less. It is a statement about a different target and costs power relative to `eps=0`.
* The margin levels (0.01, 0.02 in the simulation; two preset vectors in the case studies) are arbitrary illustrations, not recommendations.
* No tuning was done. All variants were run once as specified.

## Simulation (R = 300 replications per cell, B = 300, alpha = 0.05, 8 models, K = 3)

Scenario A is the design of `experiments/simulate.py` (12 cells). Scenario B is a new least-favourable tie design: six models tied on
criterion 1 and strictly ordered on criteria 2-3 (all in `P0`, every pair tied on one criterion), plus two models strictly dominated
by model 0 (`delta = 0.03`; 6 cells). The Monte-Carlo standard error of a retention proportion near 0.99 is about 0.6 percentage points.
The step-down set was a subset of the original TCS in 100% of replications.

Retention of the relevant population set (`P0(eps)` for margin variants):

| variant | scenario A | scenario B |
|---|---|---|
| original TCS | 0.997-1.000 | 0.990-1.000 |
| step-down | 0.997-1.000 | 0.987-1.000 |
| step-down + preselection | 0.997-1.000 | 0.987-0.997 |
| step-down, eps = 0.01 / 0.02 | 1.000 | 0.997-1.000 |
| naive Pareto front | 0.32-0.40 | 0.000-0.003 |

All TCS variants retained at or above the nominal 0.95. None fell below. The margin variants retain `P0(eps)` at 1.000 or 0.997 even in the boundary cell
(`delta = 0.02 = eps`, where model 4 is in `P0(eps)`). All TCS variants remain very conservative even in the tie design.

Exclusion power (share of strictly dominated models excluded), selected cells:

| cell | original | step-down | + preselect | eps=0.01 | eps=0.02 |
|---|---|---|---|---|---|
| A n=200 d=.02 rho=0 | 0.158 | 0.158 | 0.170 | 0.022 | 0.002 |
| A n=200 d=.05 rho=0.5 | 0.910 | 0.918 | 0.922 | 0.715 | 0.557 |
| A n=500 d=.02 rho=0.5 | 0.570 | 0.573 | 0.575 | 0.353 | 0.078 |
| B n=200 rho=0 | 0.503 | 0.505 | 0.507 | 0.400 | 0.137 |
| B n=500 rho=0 | 0.823 | 0.830 | 0.845 | 0.523 | 0.490 |
| B n=500 rho=0.5 | 0.835 | 0.843 | 0.858 | 0.578 | 0.495 |

Naive Pareto front excludes 0.96-1.00 in every cell, with retention of 0-40%. The step-down gain is at most about 1 percentage point and
the preselection gain at most about 2.2 points (B, n=500). Both vanish where power is already 1.
For margin variants, power is measured for strictly dominated models, so the drop is expected. Against the
margin target `P0(eps)` the exclusion share is higher (for example A, n=2000, delta=0.02, eps=0.02: 0.983 of the models outside `P0(eps)`).

## Case studies (alpha = 0.05, B = 1000, same pipeline and seed as `experiments/casestudy.py`)

| data | critical value: orig / step-down / preselect | models excluded (all variants, incl. both margins) | max S_m |
|---|---|---|---|
| Adult (n_test 19,537) | 3.58 / 3.58 / 3.60 | 0 | 0.59 |
| German credit (n_test 400) | 3.71 / 3.71 / 3.78 | 0 | 0.51 |

No model is excluded by any variant. The step-down has nothing to iterate on, because no model is removed at step 1.
The largest statistic is about 3 critical-value units below the cutoff, so no reasonable sharpening of the critical
value changes the result. On criterion subsets (2-3 of 4 criteria, 10 subsets per dataset), the number of subsets with at least one
exclusion is the same for the original, step-down and preselection variants (Adult 7 of 10, German 0 of 10).

## Conclusion

Step-down and preselection keep validity in these simulations (retention at or above 0.987 against a nominal 0.95) and give a real
but small power gain (about 0 to 2 percentage points of exclusion share, no gain on real data). The margin changes the target and
reduces exclusions. The conservativeness reported in the manuscript is inherited from the max-over-criteria critical value and the
intersection-union structure, and these extensions do not remove it.
