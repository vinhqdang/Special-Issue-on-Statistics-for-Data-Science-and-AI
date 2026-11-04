# Shared notation and conventions for manuscript_v2 (all sections must follow this)

Working title: "Labelling where models disagree: label-efficient comparison of classifiers on group fairness"

- Pool: U = {1,...,N} unlabeled instances. Models m = 1..M (fixed, trained elsewhere), binary predictions
  \hat y_{mi} in {0,1}, predicted probabilities \hat p_{mi}. Group G_i in {0,1} (known for all i).
  Label Y_i in {0,1} (unknown until queried). Budget: L labels.
- Totals (linear in Y): P_g = sum_{i: G_i=g} Y_i ; Q_g = sum_{i: G_i=g} (1-Y_i);
  TP_{mg} = sum_{G_i=g} Y_i \hat y_{mi} ; FP_{mg} = sum_{G_i=g} (1-Y_i) \hat y_{mi} ; E_m = sum_i |\hat y_{mi} - Y_i|.
  Generic: t_{ic}(Y_i) = Y_i u_{ic} + (1-Y_i) v_{ic}, T_c = sum_i t_{ic}(Y_i), vector T.
- Group rates: TPR_{mg} = TP_{mg}/P_g, FPR_{mg} = FP_{mg}/Q_g. Gaps: \Delta^{TPR}_m = TPR_{m1} - TPR_{m0}, \Delta^{FPR}_m likewise;
  equalised-odds gap EO_m = max(|\Delta^{TPR}_m|, |\Delta^{FPR}_m|).
- Comparison estimands for a pair (m,j): \theta^{TPR}_{mj} = \Delta^{TPR}_m - \Delta^{TPR}_j ; \theta^{FPR}_{mj} likewise ;
  \theta^{ERR}_{mj} = (E_m - E_j)/N. Vector \theta = f(T), f smooth.
- Key identity: \theta^{TPR}_{mj} = sum_g s_g (1/P_g) sum_{G_i=g} Y_i (\hat y_{mi}-\hat y_{ji}), s_1=+1, s_0=-1;
  numerators vanish on instances where m and j agree; denominators P_g, Q_g involve all instances.
- Design: strata h = 1..H (default: (G_i, K_i) with K_i = sum_m \hat y_{mi}, the number of models predicting 1;
  strata smaller than 50 merged with a neighbour of the same group). N_h stratum size, n_h labels,
  stratified simple random sampling without replacement (STSI). Estimator \hat T_c = sum_h N_h \bar t_{hc}.
  Plug-in \hat\theta = f(\hat T). Variance: linearisation z_i = \nabla f(\hat T)^T t_i(Y_i),
  \hat V = sum_h N_h^2 (1 - n_h/N_h) s^2_{z,h} / n_h ; 95% Wald intervals.
- Allocation: n_h \propto N_h \sqrt{ \sum_t S^2_{ht} / S^2_t } (weighted generalised Neyman), S^2 computed with Y_i
  replaced by Bernoulli(\pi_i), proxy \pi_i = (1/M) sum_m \hat p_{mi}; minimum 10 labels per stratum.
- Designs compared: SRS (simple random sampling), GROUP (strata = group, same allocation rule),
  PATTERN (proposed), ORACLE-ALLOC (PATTERN strata, allocation computed with true labels; a reference, not a bound).
- Efficiency: MSE_SRS / MSE_design (equivalently: how many times more labels SRS needs).
- Data: 11 settings (Adult-sex, Adult-race, Bank, Credit-card default, Law school, MAGIC, EEG, Phoneme, Spambase,
  Churn, Electricity), nine models (logistic regression x2, trees depth 3 and 12, random forest, histogram GB,
  25-NN, Gaussian NB, MLP), pools = 40% held-out split (cached in results/pool_cache/<key>.npz: arrays pred (N,M),
  p (N,M), g, y, predn). Groups in 6 settings are arbitrary median splits (no fairness meaning). Nomao excluded.
- Code: tcs/labeling.py (design, estimators), experiments/label_efficiency.py (main study), results/label_efficiency.json.
- Style: British spelling ("equalised", "labelling"), no bold-heavy prose, no em-dash runs, numbers with source file.
  Never invent results or references; every number must come from a results file produced by code in the repo.
  No mention of any AI tool or assistant anywhere.
