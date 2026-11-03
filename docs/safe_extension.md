# SAFE-family (rank graduation) metrics as TCS coordinates

Files: `tcs/safe_metrics.py` (metrics, with a `_selfcheck()`), `experiments/safe_case.py` (driver), `results/safe_case.json` (output). Run `python experiments/safe_case.py` from the repository root (B = 1000, alpha = 0.05, seed 0, same split, model zoo and feature perturbation as `experiments/casestudy.py`).

## 1. Verified definitions and sources

All formulas below were read in the sources listed; none are reconstructed from memory.

**Rank graduation metric RGX (the common form).** For vectors Y and Z of length N, the concordance curve L_{Y,Z} interpolates the points (k/N, sum_{i<=k} Y_{r*_i}), with r* the ascending ordering induced by Z. L_Y is the Lorenz curve of Y (ascending ordering of Y), L^c_Y the dual Lorenz curve. Then

RGX(Y, Z) = 1 - int |L_{Y,Z}(t) - L_Y(t)| dt / int |L^c_Y(t) - L_Y(t)| dt = 1 - CvM(Y, Z) / G(Y),

with 0 <= RGX <= 1 because L_Y <= L_{Y,Z} <= L^c_Y.
Source: G. Auricchio, A. E. Bernardelli, P. Giudici, G. Toscani, "On Rank Graduation Metrics for High Dimensional Ordinal Data", arXiv:2511.23100 (v1), Sec. 2.2 (eq. 2.3, 2.4) and Sec. 4 (eq. 4.28, 4.29). The slide deck of P. Giudici, "SAFE Artificial Intelligence in Finance" (Bank of Finland event, https://www.suomenpankki.fi/globalassets/bof/en/financial-stability/events/6_giudici_slides.pdf) gives the same quantity as the ratio of the area between the dual Lorenz and concordance curves to the area between the dual Lorenz and Lorenz curves, and states RGX(Y,Z) = 1 - CvM(Y,Z)/G(Y). Note: the first expression of eq. (2.4) in the arXiv text is printed with L^c_Y - L_{Y,Z}; the second expression (and the slide formula) is the one that gives RGA = 1 for a perfect ranking, and it is the one implemented.

**RGA (accuracy).** RGA = RGX(Y, Y-hat), Y the ground truth, Y-hat the prediction (arXiv:2511.23100, Remark 2). Properties stated on the slides: 0 <= RGA <= 1; 1 for a perfectly concordant model, 0 for a perfectly discordant one, 0.5 for random predictions; for a binary response RGA equals AUC; defined for ordinal, continuous and multivariate responses. The arXiv paper repeats the AUC statement (Sec. 1.1).

**RGR (robustness).** RGR = RGX(Y-hat, Y-hat^(p)): first argument the prediction on the original data, second the prediction on perturbed data (arXiv:2511.23100, Remark 2; slides). The perturbation itself is not fixed by the definition: the arXiv application adds Gaussian noise N(0, (0.5 sd(y-hat))^2) to the predictions (eq. 5.53); the CEUR-WS Vol-4259 paper 15 (Kolesnikov et al., HMDA application) uses Gaussian noise injection at several levels; the `safeaipackage` function `compute_rgr_values` perturbs selected variables by a `perturbation_percentage` (default 0.05) with the mechanism not documented on the page I could read. We therefore use the perturbation already in `casestudy.py` (feature noise, 0.5 training sd on numeric features), and state that it is our choice.

**RGE (explainability).** RGE = RGX(Y-hat, Y-hat^(-j)), prediction on the full data against the prediction after removing feature j (arXiv:2511.23100, Remark 2; slides). Defined, not implemented: it needs one model refit per feature and per bootstrap draw (or a fixed-model variant), which does not fit the paired-resample design of the TCS bootstrap used here.

**Fairness.** The slides define RGF as RGX(Y-hat_A, Y-hat_B), "fairness between groups A and B", without further detail on how two vectors of different group sizes are paired. The authors' Python package (`GolnooshBabaei/safeaipackage`, README as summarised by the page; also `safeai.readthedocs.io`) provides `check_fairness.compute_rga_parity(xtrain, xtest, ytest, yhat, model, protectedvariable)`, documented as computing RGA separately in the two groups of a binary protected variable and returning the difference. That documented procedure is what we implement: RGA parity gap = |RGA(group 0) - RGA(group 1)|. The absolute value is our choice (the package returns the signed difference).

## 2. What could not be verified

- Raffinetti (2023), Quality & Quantity 57:131-150: only the abstract and library descriptions were reached; the exact finite-sample formula, tie handling and any inference results come from the slides and the arXiv paper, not from the journal text.
- Babaei, Giudici, Raffinetti (2025), Expert Systems with Applications 259:125239; Giudici, Raffinetti, Riani (2024), IJDSA 20:1043-1050; Wu (2026), IJDSA, DOI 10.1007/s41060-025-01002-8: the publisher pages were not readable (access wall or not returned). The fairness metric in Wu (2026), including its binarisation of continuous variables, is therefore NOT verified and NOT implemented. Whether the RGA parity gap coincides with the fairness measure of that paper is unknown.
- No source consulted specifies tie handling. We use midranks for ties in the scores and in y; with a binary response this gives RGA = AUC with half credit for ties (checked numerically). This is a convention, not a published one.
- The slides' exact exposition of RGF (how Y-hat_A and Y-hat_B are aligned) is unverified.

## 3. Implementation

With ascending midranks ry, rz and n observations, the area ratio reduces exactly to RGX(y, z) = sum_i y_i (ry_i + rz_i - n - 1) / sum_i y_i (2 ry_i - n - 1), which is vectorised over models (columns) and needs only two rank computations per resample. Higher RGA/RGR is better; the TCS coordinates are the lower-is-better transforms `rga_loss = 1 - RGA`, `rgr_loss = 1 - RGR`, `rga_parity_gap`. Unit checks (`python tcs/safe_metrics.py`): RGA(y, y) = 1; RGA(y, -y) = 0 (the minimum); invariance to monotone transforms; RGA = sklearn AUC on binary y; constant score gives 0.5; a 3-point example computed by hand from the Lorenz, dual and concordance curves (0.75); matrix input; RGR(s, s) = 1.

## 4. Results

Setups as in `casestudy.py`: Adult (n_test = 19,537, sex) and German credit (n_test = 400, age < 26), nine models, B = 1000 paired bootstrap resamples (the same resample indices feed every criterion set), alpha = 0.05. The predicted probability is the score for RGA/RGR. TCS size is out of 9.

| Dataset | Criteria | TCS size | Naive Pareto size |
|---|---|---|---|
| Adult | conv4: error, EO gap, ECE, noise drop | 9 | 8 |
| Adult | conv3: error, EO gap, noise drop | 9 | 7 |
| Adult | safe3: 1-RGA, 1-RGR, RGA parity gap | 8 | 6 |
| Credit-g | conv4 | 9 | 7 |
| Credit-g | conv3 | 9 | 4 |
| Credit-g | safe3 | 9 | 5 |

Adult, safe3: LR is the only model excluded (S = 3.44 > c = 3.30); it is significantly dominated on all three SAFE coordinates (e.g. by RandomForest, whose point estimates are 0.086 / 0.045 / 0.043 against 0.099 / 0.052 / 0.052). No model is excluded with any conventional set, so the SAFE set is sharper here. The likely reason is that 1-RGA (= 1 - AUC for this binary response) is estimated more precisely than 0/1 error and separates models that are tied on accuracy at a threshold. In the one-at-a-time swaps (conv3 with one coordinate replaced), swapping the fairness coordinate also yields 8.
Pareto-membership probabilities, Adult, model order LR, LR-strong-reg, Tree(d=3), Tree(d=12), RF, HistGB, kNN, NB, MLP: conv4 = 1.00, 1.00, 1.00, 0.84, 1.00, 1.00, 1.00, 0.27, 1.00; safe3 = 0.00, 1.00, 0.05, 1.00, 1.00, 1.00, 1.00, 1.00, 0.02. Under the conventional criteria almost every model is on the front with probability near 1 (NaiveBayes, with ECE of 0.40, is the exception); under the SAFE criteria LR, Tree(d=3) and MLP are almost surely off the front while NaiveBayes (very robust, 1-RGR = 0.006) and Tree(d=12) (smallest parity gap) are on it. The rankings differ, so the choice of coordinates changes the trust conclusion, which is what the TCS is designed to expose.
Credit-g, safe3: all nine models stay in the TCS (S from -4.85 to 0.63, c = 3.46) although the naive front has five; Pareto probabilities are 1.00, 0.92, 0.50, 0.30, 0.74, 0.32, 0.20, 1.00, 0.28 (safe3) against 0.64, 0.80, 0.94, 0.87, 0.99, 0.86, 0.78, 0.13, 0.75 (conv4). With 400 test points no set separates models, so the naive front is mostly sampling noise; the safe3 probabilities are less uniform than the conventional ones but mostly between 0.2 and 0.9.
Spearman correlation across the nine models between a conventional coordinate and its SAFE counterpart: error vs 1-RGA 0.80 (Adult), 0.74 (Credit-g); noise drop vs 1-RGR 0.87 (Adult), -0.67 (Credit-g; noise drop is negative for several models there); EO gap vs RGA parity gap 0.75 (Adult), 0.50 (Credit-g).

Caveats: single split and seed; the bootstrap is over the test set with fixed fitted models (as in the paper); the Adult RGR uses a feature perturbation not prescribed by the cited definition.

## 5. Paragraph for the manuscript

Rank-based SAFE measures plug directly into the criterion vector. Writing RGX(Y, Z) = 1 - CvM(Y, Z)/G(Y) for the Rank Graduation metric, we use 1 - RGA (RGX of the outcome against the predicted probability, equal to 1 - AUC for a binary outcome), 1 - RGR (RGX of the predictions on the original and on the perturbed test set), and the absolute difference of RGA between the two levels of the protected attribute, all oriented so that smaller is better. These statistics are computed on the same bootstrap resamples as the conventional criteria, so the TCS applies unchanged. On Adult (n_test = 19,537) the three SAFE criteria give a TCS of 8 of 9 models, excluding logistic regression, whereas the conventional error, equalised-odds gap, calibration and noise-sensitivity criteria retain all nine; the Pareto-membership probabilities also change markedly (logistic regression 1.00 against 0.00). On German credit (n_test = 400) both criterion sets retain all nine models. The fairness coordinate is the RGA parity gap documented in the authors' software; we did not implement the binarisation-based fairness measure of Wu (2026), and the explainability measure RGE, which requires refitting without each feature, is left to future work.
