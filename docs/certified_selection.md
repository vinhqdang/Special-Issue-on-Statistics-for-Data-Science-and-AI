# Certified model selection under several trust criteria

## Problem
M trained models, K criteria (smaller is better, all in probability units), one test sample. The
decision maker states a polytope W of acceptable weight vectors (here: equal weights plus or minus r,
intersected with the simplex). The output is one model and an upper bound on how far it can be from
the best model for the least favourable acceptable weighting.

## Target
For model m and weight w, regret is w'mu_m - min_j w'mu_j. The worst case over W is
R_m = max_{w in W} max_j w'(mu_m - mu_j). The inner function is linear in w, so the maximum is at a
vertex v of W: R_m = max_{v, j} d_mj(v) with d_mj(v) = v'(mu_m - mu_j). R_m >= 0 (take j = m).

## Procedure (one-stage, the main method)
1. Paired bootstrap of the test instances, B resamples, same resample for all models.
2. For every ordered pair (m, j), m != j, and vertex v: estimate d^, bootstrap standard error se,
   centred studentised replicate z = (d* - d^)/se.
3. q = (1 - alpha) quantile over b of max_{m != j, v} z.
4. U_m = max(0, max_{j != m, v} d^_mj(v) + q se_mj(v)).
5. Choose m^ = argmin_m max_{v, j} d^_mj(v) (plug-in minimax regret) and report U_{m^}.

## Validity
Let T = max_{m != j, v} (d_mj(v) - d^_mj(v)) / se_mj(v). If T <= q then for every m,
R_m = d_{m j*(m) v*(m)} <= d^_{m j* v*} + q se_{m j* v*} <= U_m (and R_m = 0 <= U_m when the maximiser is j = m).
Because d_mj = -d_jm, the family of all ordered pairs is closed under sign change, so the bootstrap
law of max z approximates the law of T. Under joint consistency of the bootstrap for the studentised
differences and continuity of the limit law at its quantile, P(R_m <= U_m for all m) >= 1 - alpha + o(1).
The bound holds simultaneously for all models, hence for the data-selected m^ and for any other choice rule.

Assumptions and caveats: finite M and K; non-degenerate standard errors (pairs with zero standard error
are treated as exact); criteria such as binned ECE and the equalised-odds gap are not smooth
functionals, so consistency is not guaranteed and is checked by simulation; inference is conditional
on the trained models.

## Variants tried
- Two-stage (screen plausible best models per vertex at alpha/2, then bound at alpha/2): valid by a
  union bound, but in all checks it was slightly wider than one-stage, because the screen removes few
  rivals and the per-stage level is halved. Not used.
- Selective (quantile of the bootstrap maximum over the comparisons of the bootstrap choice only):
  narrower by roughly 10-20 percent, no proof of validity; reported as a heuristic with its empirical coverage.

## Sample-size planning
For one criterion the certificate behaves like q * se, and se scales as n^(-1/2). A pilot certificate
U(n0) therefore predicts the test size needed for a target bound delta: n(delta) = n0 (U(n0)/delta)^2.
