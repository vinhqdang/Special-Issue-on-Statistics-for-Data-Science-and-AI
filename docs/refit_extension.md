# Refit-bootstrap extension of the trust confidence set

Code: `tcs/refit.py` (schemes), `experiments/refit_case.py` (German credit, Adult), `experiments/refit_sim.py` (simulation).
Results: `results/refit_case.json`, `results/refit_sim.json`. `tcs/core.py` is unchanged; every scheme only produces a (R, M, K) array that is passed to `trust_confidence_set`.

## 1. What the fixed-model TCS targets, and what the refit version targets

Fixed-model TCS (current): the models f_1..f_M are trained once on a training set D_tr. Resampling test rows gives the law of the criterion estimate for the **trained models**, so the target is mu_cond[m,k] = criterion k of the fixed function f_m under the data distribution P. Training randomness (which rows, which seed) is conditioned on.

Refit TCS: the target is the criterion of the **learning algorithm** A_m at training size n_tr, averaged over training sets and algorithm seeds,

    mu_alg[m,k] = E_{D ~ P^{n_tr}, seed} [ criterion_k( A_m(D, seed) ; P ) ].

The non-dominated set P0 is defined from mu_alg exactly as before (m is strictly dominated iff some j is strictly smaller on every criterion). This is the right target when the question is "which modelling recipes are worth carrying forward", and it is not the same set as P0 of one trained model. The two targets differ most for unstable learners (deep trees, kNN with small k, MLP, forests with few trees), whose trained instances vary a lot around mu_alg. The size n_tr matters: mu_alg is a function of the training size, so conclusions refer to n_tr, not to the full n.

Why one common refit per replicate. In replicate r all M models see the same training resample, the same evaluation sample and the same seed. Pairwise differences d[m,j,k] then carry the positive covariance between models that is needed for the studentised statistics (the same pairing logic as in the original test-set bootstrap).

## 2. Schemes

1. **paired_bootstrap (primary).** Keep the original split (60/40, stratified, seed 0). In each replicate draw n_tr rows of the training set and n_te rows of the test set, independently, with replacement; refit all models on the training resample (with every `random_state` set to the replicate seed); evaluate on the test resample. The point estimate is the original-split estimate, and the replicates are centred on it by `trust_confidence_set`. Training and evaluation data are independent, so there is no out-of-bag optimism and no leakage; the test-side variance is the same as in the fixed-model TCS and the training-side variance is added.
2. **oob_bootstrap.** Resample all n rows with replacement, refit, evaluate on the out-of-bag rows (about 36.8% of n, a different number each replicate). Included for comparison.
3. **repeated_splits (cheap alternative).** R random stratified 60/40 splits, the same split and seed for every model; point estimate = mean over splits; replicates = the R split-level estimates. No resampling with replacement, so no duplicates.

Why paired_bootstrap is the primary scheme: (a) evaluation rows are independent of training rows, which avoids the OOB scheme's variable evaluation size and its dependence between the training and evaluation draws; (b) it retains the original-split point estimate, which is an approximately unbiased draw of mu_alg at size n_tr (no duplicates), whereas the bootstrap-replicate mean is a biased version of it; (c) it adds exactly one thing to the existing procedure, the training-side variability.

## 3. Caveats (real, and visible in the numbers)

- **Duplicates in training.** A bootstrap training set contains about 63.2% distinct rows, so each refit learns from fewer effective observations than n_tr. This degrades learners that are sensitive to effective sample size and also changes the behaviour of others: kNN counts duplicates as extra neighbours, trees with a leaf-size rule and MLP with an internal early-stopping split can see the same row on both sides, forests are affected through the leaf-size rule. The replicate-level criteria are therefore biased relative to mu_alg (typically pessimistic for error). Measured mean(replicate) minus point, averaged over models, German: error +0.027, equalised-odds gap +0.038, ECE +0.049, noise drop +0.014 (paired scheme); Adult (n=4000): about 0 to +0.01, with the OOB scheme at -0.05 for the equalised-odds gap. Because the replicates are re-centred at the point estimate, only the spread (and its shape) is used, but the spread itself is also distorted: unstable learners get inflated training variance from the duplicates, stable ones (LR, strong regularisation) do not.
- **Out-of-bag evaluation (oob_bootstrap).** OOB sets have random size (about 0.368n) and models are trained on about 0.632n distinct rows, so the replicate target is mu_alg at a smaller, random effective size and is pessimistic. Criteria that are not smooth in the sample (ECE binning, group-conditional gaps on small groups) are noisier on a 37% subsample; this is visible as the largest critical values (5.1 German, 5.3 Adult).
- **Optimism if train rows leak into evaluation.** A refit evaluated on the full resampled data (instead of OOB / an independent test resample) would be strongly optimistic. None of the schemes here do this.
- **Variance is not the sampling variance of a single split.** The bootstrap distribution mimics variation of (training set, test set) around one observed pair. It does not account for the variation of the original split itself beyond what the resamples reproduce, and the bootstrap of non-smooth, unstable learners (trees, kNN, nearest-neighbour-type interpolators) is known not to be consistent. Asymptotic control of the TCS is therefore not established for this extension; the claim is only empirical.
- **Repeated splits are dependent and under-dispersed.** Splits overlap, so their spread understates the variance of an independent-draw estimate, as in Nadeau and Bengio (2003) for CV variance. In the case studies the split-based sd ranged from 0.84 to 1.86 times the fixed-model sd (mostly above 1, because the splits also vary the training data), but in the simulation it was 0.74 to 0.91 times, i.e. below the fixed-model sd. It also has no duplicate bias.
- **Cost.** B' x M refits instead of M. German (n=1000, 9 models): 205 s for B'=100 paired, 181 s for 100 splits, 159 s for 100 OOB replicates (2 workers), against 3.5 s for the fixed-model TCS with B=1000. Adult (n=4000): 111 s for B'=60.
- The seed is varied across replicates for every model, so seed variance (not only data variance) is included. Model hyper-parameters are fixed; selection by search is not captured.

## 4. Case studies

Fixed-model TCS reproduces the settings of `experiments/casestudy.py` (same split, noise draw, B=1000). Refit schemes: German B'=100 (R=100 splits); Adult subsampled to n=4000 (2400 train, 1600 test; seed 0), B'=60, R=60. Reduced B' and subsampling are for runtime. alpha=0.05.

TCS size (out of 9 models) and critical value:

| scheme | German size / crit | Adult size / crit |
|---|---|---|
| fixed model | 9 / 3.71 | 9 / 3.69 |
| paired refit | 9 / 4.63 | 9 / 4.61 |
| repeated splits | 9 / 3.57 | 9 / 3.55 |
| OOB refit | 9 / 5.10 | 9 / 5.27 |

No set membership changes: every scheme keeps all 9 models, on both datasets. The refit versions raise the critical value by about 25% (paired) to 40% (OOB). Ratio of replicate sd to fixed-model sd, averaged over models, for (error, equalised-odds gap, ECE, noise drop): German paired 1.71, 1.35, 1.97, 1.64; German splits 1.37, 0.93, 1.86, 1.15; Adult paired 1.30, 1.14, 1.70, 2.31; Adult splits 0.91, 0.84, 1.21, 1.34. So training variance roughly adds 30 to 100% to the standard errors for most criteria, but the sets were already maximal, so the sets do not change.

Pareto-membership probabilities (fixed / paired / splits / OOB), selected rows. German: Tree(d=12) 0.87 / 0.19 / 0.08 / 0.20 and NaiveBayes 0.13 / 0.22 / 0.25 / 0.27 (the fixed-model value for Tree(d=12) is high because the fixed test-bootstrap cannot see its training instability); RandomForest 0.99 / 0.95 / 0.96 / 0.97. Adult: Tree(d=12) 0.16 / 0.27 / 0.22 / 0.13, LR-strong-reg 0.26 / 0.78 / 0.50 / 0.88, LR 1.00 / 1.00 / 0.95 / 0.95. Refits can move a probability by more than 0.5, so this descriptive companion is much more sensitive than the TCS.

Composite-rank 95% intervals (German, fixed vs paired refit): Tree(d=12) [2, 9] -> [5.5, 9]; RandomForest [1, 5] -> [1, 6]; NaiveBayes [6, 9] -> [5.5, 9]; LR [2, 8] -> [1, 8]. Adult: LR [1, 4] -> [1, 7]; kNN(25) [1, 3] -> [1, 5.5]; MLP [1, 5] -> [1, 5.5]; NaiveBayes [8, 9] -> [6.5, 9]. Intervals widen for the stable models; the unstable deep tree moves toward a worse and tighter rank band in the German data. Full tables in `results/refit_case.json` (keys `fixed`, `paired_refit`, `repeated_splits`, `oob_refit`).

## 5. Simulation with known truth

Generative model (`experiments/refit_sim.py`): 6 Gaussian features, a binary group correlated with x1, y from a logistic model with an interaction and a quadratic term. 11 algorithms (LR, strongly regularised LR, trees of depth 2/5/12, kNN with k=1/5/40, Gaussian naive Bayes, a prior-only constant classifier, a 20-tree forest). Criteria: error, ECE, error under feature noise (the equalised-odds gap and the noise drop were dropped because they trade off against every model, making almost nothing dominated and the retention test vacuous). Truth: mu_alg from 100 independent training sets of size n_tr, each evaluated on 10,000 fresh test points, so mu_alg has Monte Carlo error; P0 uses strict dominance on these values. Per replicate: a fresh dataset, split 60/40, then fixed-model TCS (B=500), paired-refit TCS and repeated-split TCS.

| setting | P0 (algorithms) | scheme | retention of P0(algorithm) | retention of P0(trained models) | mean TCS size (of 11) |
|---|---|---|---|---|---|
| n=500 (300/200), 60 reps, B'=40 | NaiveBayes, Prior(const) | fixed | 1.00 | 1.00 | 10.95 |
| | | paired refit | 1.00 | 1.00 | 11.00 |
| | | repeated splits | 1.00 | 1.00 | 11.00 |
| | | Pareto of point estimates | 0.85 | 0.47 | 4.2 |
| n=3000 (1800/1200), 40 reps, B'=30 | LR, kNN(40), NaiveBayes, Prior(const), RF(20) | fixed | 1.00 | 1.00 | 9.18 |
| | | paired refit | 1.00 | 1.00 | 11.00 |
| | | repeated splits | 1.00 | 1.00 | 8.98 |
| | | Pareto of point estimates | 0.25 | 0.38 | 4.5 |

Honest reading. All three TCS variants retain the population-nondominated algorithms in every replicate, so retention is at or above nominal, but this is largely because the procedure retains almost everything: at n=500 neither version ever excludes an algorithm, and at n=3000 the paired refit still retains all 11 (it removes nothing at all), whereas the fixed-model TCS and the split version drop about 2 algorithms. With 40-60 replicates, 100% retention is only consistent with true retention above about 0.93 (rule of three), so the simulation does not test whether the refit procedure is exactly calibrated at the 95% level; it shows it is not anti-conservative here and has less power than the fixed-model TCS. The naive Pareto front of point estimates fails badly (retention 0.25 to 0.85 of the population front; 0.38 or lower for trained-model targets at n=3000, as in the manuscript's claim). Mean ratio of replicate sd to fixed-model sd (error, ECE, noisy error): paired 1.28, 1.33, 1.33 at n=500 and 1.20, 1.30, 1.27 at n=3000; splits 0.74, 0.84, 0.80 and 0.75, 0.91, 0.81. The splits scheme is therefore the least conservative, and its sd below the fixed-model sd, together with the Nadeau-Bengio argument, suggests it may undercover in settings where the paired scheme does not retain everything; the present simulation was not powered to show that. A better-powered check (more replicates, more near-ties, n between 500 and 3000) was not run within the time budget. P0 sizes (trained models): 3.2 at n=500 and 4.2 at n=3000.

## 6. Recommendation

For reporting TCS as a statement about the algorithm rather than a fixed trained model, use the paired refit bootstrap (original split, resample train and test independently, refit with varying seeds) and report the fixed-model TCS alongside. Treat its sets as conservative (critical values about 25% higher, standard errors up to 2x higher for ECE and noise drop). The repeated-split version costs about the same here (4 to 12% less time), has no duplicate bias and is a reasonable screening tool, but its spread can be too small to be relied on for coverage (simulation). In these data the change of scheme did not alter membership, so the conclusion "all 9 models survive" is robust to ignoring training variance; it moves Pareto-membership probabilities and rank intervals instead.
