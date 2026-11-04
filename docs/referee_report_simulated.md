# Referee report (simulated)

**Manuscript:** "PATTERN-DIF: fixed-budget stratified labelling with a difference estimator for comparing several classifiers on group-fairness contrasts" (manuscript_v2)
**Venue:** International Journal of Data Science and Analytics, special issue "Statistics for Data Science and AI"
**Note on references:** proposition, table and section numbers below follow the order of appearance in the compiled manuscript and may be off by one; I give section names where precision matters.
**Material examined:** manuscript_v2/main.tex and all section files (including the appendix proofs), docs/label_efficient_lit.md, the code in tcs/ and experiments/, the cached pools in results/pool_cache and the JSON results. I recomputed several quantities from the repository (marked "checked" below); where I checked something and found it sound I say so.

---

## 1. Summary of claims

The paper considers a finite pool of N instances with known group G and nine already-trained binary classifiers with known predictions and probabilities. A budget of L labels is to be spent once to compare the models on 3·C(9,2) = 108 contrasts: differences between models of the between-group TPR gap, the FPR gap, and the error rate (eq. 3).

The proposal, PATTERN-DIF, has four components:

- **Strata.** Strata are (group, K), where K is the number of models predicting 1 (at most 2(M+1) = 20 strata, with strata smaller than 50 merged).
- **Allocation.** Labels are allocated by a weighted generalised Neyman rule (eq. 6), with anticipated variances computed under a Bernoulli(π_i) working model, π being the mean of the models' probabilities. Weights are 1/S_k², with a floor of 10 labels per stratum.
- **Estimator.** The stratified difference (generalised-regression-type, "PPI with λ = 1") estimator (eq. 5) is used, with the pool-wide proxy total treated as a known constant. Contrasts are obtained by plug-in, with linearisation Wald intervals.
- **Variance floor.** A "variance floor" is added for strata with zero sample variance.

The theory covers unbiasedness of totals, consistency, asymptotic normality and variance-estimator consistency (Prop. 1), influence values (Props. 2–3), conditions for the difference estimator to beat the plain one (Props. 4–7), weighted Neyman allocation and its robustness to proxy error (Props. 8–11), and design-effect formulae in terms of the disagreement rate ρ (Cor. 1).

The empirical claims are these:

- Median efficiency relative to SRS is 5.8 / 6.3 / 6.9 / 15.5 at L = 250 / 500 / 1000 / 2000, against 3.7–4.4 for the plain estimator on the same design and about 1.0 for group stratification alone.
- Most of the gain comes from the disagreement strata.
- Coverage is near nominal on average (0.94–0.95), with poor worst-case contrasts at small budgets.
- The linearised design effect predicts the observed efficiency to within Monte Carlo error.
- The method beats importance-sampling, score-strata, disagreement-only and PPI/PPI++/StratPPI baselines.

I reproduced the headline medians from results/label_efficiency_dif.json (5.76, 6.29, 6.87, 15.45; 5.53, 5.79, 5.98, 9.42 without law school). The numbers in the text are faithful to the stored results.

## 2. Novelty and positioning

**Overall.** The authors state the contribution honestly and narrowly (abstract; Sec. 2 "Positioning"; Sec. 4 opening). I agree with that narrow statement, and it has consequences that the paper does not fully draw. Every ingredient is classical or published:

- Stratified sampling with Neyman and multivariate-compromise allocation.
- The difference/GREG estimator (Cassel–Särndal–Wretman 1976), which is exactly what PPI with λ = 1 is in a finite pool.
- Linearisation variance.
- Disagreement-focused sampling (Sawade et al. 2012; Musgrove et al. 2023; the diagnostic-test literature).
- Score-based stratification for classifier evaluation (Bennett and Carvalho; Kumar and Raj).
- Stratified PPI (Fisch et al. 2024).

What is left is the choice of the stratifier (group × count of positive predictions, K), a standard multi-target Neyman rule, and the application to between-model contrasts of group-rate gaps. This is a sensible and useful application paper. It is not a methodological advance in survey sampling or in PPI.

**The theory is mostly textbook.**

- Unbiasedness (Prop. 1a) is one line.
- The weighted Neyman rule (Prop. 8) is Cauchy–Schwarz.
- The proxy-misallocation loss (Prop. 11b–c) is Kantorovich.
- The sufficient condition β_h > 1/2 (Prop. 4a) is an algebraic identity.
- The closed-form design effects (Cor. 1) are exact only for a two-model idealisation. The paper's own check (Table 3, Sec. 5.1) shows they overstate the nine-model gain by a median factor of about 2, so they are an illustration rather than a result.
- The genuinely design-specific content is Prop. 2 (influence values and the structure of the agreement strata). It is correct (I re-derived eqs. 11–15 and the TPR stratum variance; they are right) and is the best part of the theory.

**Positioning against cited work.** The related-work section is thorough and fair, and the novelty sweep (docs/label_efficient_lit.md) was done carefully and the paper respects its "avoid claiming" list. Two gaps:

1. The baselines section lists, as "not run", the methods that are conceptually closest: active statistical inference with proxy-driven inclusion probabilities (Zrnic and Candès 2024; Li et al. 2025 robust sampling) and active testing with LURE. The justification, that B1 is a one-shot analogue, is weak. B1 is Sawade's importance sampling with replacement, which is a different estimator and a different design. A one-shot, fixed-size, proxy-optimal Poisson/conditional-Poisson design with the same difference estimator is the obvious competitor to "stratify on K". The paper cites the fixed-size variant (Skalnes and Oberst 2026) but does not run it. SSME and Gao–Gronsbell's semi-supervised estimator are also not run. They are model-based, but they are the natural "what would a practitioner with these labels do" comparators for the estimator half.

2. Missing references that I verified on Crossref:
   - Zrnic and Candès, "Cross-prediction-powered inference", PNAS 2024, doi:10.1073/pnas.2322083121. Not cited. It matters because the proxy models here are trained on a separate split and a cross-fitting variant is the obvious way to handle proxy fitting on labelled data (the paper insists the proxy is never refitted; this is a design choice, and the citation would clarify the trade-off).
   - Zhang, Brown and Cai, "Semi-supervised inference: general theory and estimation of means", Annals of Statistics 2019, doi:10.1214/18-aos1756. This is the standard semi-supervised-efficiency reference that the "proxy as control variate" argument rests on.
   - Black, Raghavan and Barocas, "Model Multiplicity: Opportunities, Concerns, and Solutions", FAccT 2022, doi:10.1145/3531146.3533149. The motivating fact that near-equally accurate models disagree on individual predictions is currently supported by Marx et al. only.
   - Särndal and Swensson, "A general view of estimation for two phases of selection…", International Statistical Review 1987, doi:10.2307/1403406. The paper itself says (Sec. 2.7) that its design is two-phase with a free first phase. The two-phase variance decomposition is exactly what is missing from the inference (see M1).

I tried to verify the classical multivariate-allocation references (Bethel; Chromy) on Crossref and could not, so I do not name them as missing.

**Citations.** The 2026 references in the bibliography (Kuboyama arXiv:2609.18622, PPI++ in AoAS 2026, doi:10.1214/26-aoas2215, and others) resolve on Crossref/arXiv as far as I could check (the AoAS record resolves; the arXiv abs page for 2609.18622 is reachable). Several arXiv items are abstract-only according to the sweep's own "Basis" column, and some positioning sentences (e.g. "reporting large savings over stratified sampling", Hartmann et al.) rest on abstracts. This is acceptable but should be flagged in the text.

## 3. Technical soundness

### 3.1 What I checked and found sound

- **Unbiasedness and covariance of the difference estimator (Prop. 1a, appendix).** Correct. Because ψ_i is label-free and the pool-wide proxy total is a known constant, T̂_ψ − T = Σ N_h(ξ̄_{s_h} − ξ̄_h), and the stratified SRS covariance formula and its unbiased sample estimator apply directly. The design-based framing is legitimate for the finite pool with a label-free proxy. The authors' exhaustive enumeration check (14-unit pool, 840 samples) is a good verification.
- **Linearisation and asymptotic normality proof (Prop. 1b–d).** I read the proof of (c) and (d) line by line. The subsequence argument, the splitting of strata into the two alternatives of (A4), the Hájek-condition verification via boundedness of x_i, the Slutsky steps and the Lemma on Σ̂_h are all correct given (A1)–(A4). The bound |V̂ − V| ≤ r² max|·| is correct. I found no error in the proof as stated. The problems are with the assumptions (3.2 below), not the algebra.
- **Influence values and stratum variances (Props. 2–3, eqs. 11–15).** Correct, including the identity e_i = (Y_i − π_i)d_i and the fact that the TPR influence values sum to zero within group. The "agreement stratum" statements follow.
- **Identity V_0 − V_ψ = Σ c_h S²_z̃ (2β_h − 1) (Prop. 4a), the λ-decomposition (4b), the minimum-variance property of the clipped coefficient, and Minkowski bound (7c).** Correct. The paper's own clarification that "worse only if anti-correlated" is too strong is also right and is a point in its favour.
- **Prop. 8 (a)–(d), Prop. 9 closed forms, Prop. 10 (allocation loss), eq. (deff-sandwich).** I re-derived 8(a), (c), the Kantorovich step and the sandwich 1/ρ_any ≤ Eff ≤ 1/ρ. All correct. The Prop. 11(d) counterexample (25-fold loss) is also correct.
- **Implementation vs. text.** tcs/labeling_dif.py implements eq. (5), the residual influence values and eq. (4) as described, and the code handles fully enumerated strata and undefined denominators as stated. Coverage is computed with a 1e-9 tolerance only; this is not inflating coverage.
- **Efficiency and coverage numbers** in the text match results/label_efficiency_dif.json (my recomputation; see M3 for a composition issue).

### 3.2 Gaps and issues in the theory

**(T1) Asymptotic regime does not match the design or the claims.**

- (A1) requires a fixed H and min n_h → ∞ with f_h ≤ 1 − ε. In the experiments the per-stratum sample sizes are 10–22 at L = 250 (checked, e.g. Adult-sex: 10,11,…,22 over H = 19 strata; Credit-card and MAGIC similar), with the 10-label floor binding in several strata. Fully enumerated strata and f_h close to 1 are routine (Churn, Law school), not exceptions.
- (A3), V_ψ ≥ c·r² with r² = ΣW_h²/n_h, is the key condition for both the Taylor remainder and the variance-ratio consistency. It is an assumption about a regime in which the disagreement-weighted variance is of the same order as the "generic" variance. This is exactly the regime that the design is meant to escape: when the disagreement rate ρ is small, V_ψ/r² is of order ρ (the agreement strata carry W_h ≈ 1 − ρ of the weight but almost no variance; for the error contrast they carry none). The constants B and 1/c^{1/2} in the remainder bound then scale badly. The theory is therefore pointwise in the pool sequence and not uniform in ρ, P_g/N or n_h, and it cannot support the practical statements "reliable from 250 labels on" (Sec. 4.3) or the guidance in Sec. 6.
- The authors acknowledge that (A3)/(A4) exclude exact proxies and tiny strata, and that the floor is "a finite-sample safeguard, not an asymptotic correction". The honest summary is that coverage in the regime of interest is established only empirically, and that evidence is weak at small budgets (see E6).

**(T2) Finite-population versus superpopulation: the extension is not "usual".**

- Sec. 2.1 says inference refers to the finite pool and "extends to the superpopulation in the usual way when the pool is a random sample from it (Särndal et al. 1992, Ch. 13)". With a pool-wide known proxy total and a second-phase sample whose variance is reduced by the factor Eff, this extension is not innocuous.
- If the target is the value of the contrast in the deployment population, the total variance is the design variance plus the first-phase (pool-sampling) variance, approximately S²/N for each influence value. The ratio of the second to the first is L·Eff/N.
- I computed this ratio from results/label_efficiency_dif.json (checked). For PATTERN-DIF: Adult-sex 0.17/0.40/0.96, Adult-race 0.32/0.75/2.2, Bank 0.33/0.83/2.5 and Credit-card 0.22/0.50/1.3 at L = 500/1000/2000, and above 1 in Phoneme, Spambase and Churn already at L = 250–500. Under the naive intervals the coverage of a deployment-population contrast would then be approximately 0.86 (ratio 0.75), 0.83 (ratio 1) and 0.73 (ratio 2.2) in those cells. These are my approximate back-of-envelope figures under independence, not experimental results, but they show that the claim is not a technicality.
- The abstract's and Sec. 4's "5–16 times fewer labels" are all relative to SRS on the same finite pool. If the pool is itself a sample, the practical gain in precision about the deployment quantity is bounded by the phase-1 variance, and the headline gains at L = 1000–2000 are partly unusable.
- The paper treats the first phase as "free". Free in cost, but not free in variance. The two-phase variance estimator is standard (Särndal–Swensson) and should be derived and evaluated.

**(T3) Denominators bounded away from zero.** (A2) requires P_g/N ≥ c and Q_g/N ≥ c. This is fine as a regularity condition, and Sec. 6 Step 2 gives sensible advice. The issue is that the condition is on the *pool* totals, whereas what matters for the interval is the *sampled* count of group positives per stratum. The paper reports that undefined estimates "never occurred under the stratified designs" (checked: fail rates are 0.0 for PATTERN and PATTERN-DIF), which is plausible because each of the ~19 strata gets at least 10 labels. But the difference estimator can in principle give P̂_g ≤ 0 (the stratum corrections need not be positive), and the code silently returns NaN in that case (den_ok). The claim "never occurred" should be restated as an empirical observation for these pools, not a property of the design, and the NaN handling should be described in the paper.

**(T4) Wald intervals when strata have zero sample variance.** Rem. 1 and Sec. 4.3 treat this seriously and correctly identify the three mechanisms (structural zeros, sampling zeros, exact proxies). The statement that the floor "does not affect Prop. 1(c)–(d)" is correct given the stated conditions. Two cautions:

- (i) The "zero" floor was designed after seeing failures on these pools (the authors say so, repeatedly), and the largest improvement is in Law school, a degenerate pool. Its in-sample benefit is not evidence of generalisation.
- (ii) For ratio estimands with few sampled minority positives, the dominant failure at n_h ≈ 10 is not a zero variance but a skewed sampling distribution of a ratio, which neither a floor nor a Wald interval repairs. The authors' own diagnostic (SE/MC-SD = 0.90 in the worst target, bias < 0.16 SD) shows intervals "too short rather than off-centre" and they "did not test this further". A skewness-aware interval (e.g. a studentised or Fieller-type interval for the ratio contrasts) is the natural remedy and has not been tried; only t quantiles, a stratified percentile bootstrap and floors were.

**(T5) Calibration is assumed, but fails in the data.**

- Sec. 3.2 says "Because the predictions here are calibrated and informative". Props. 5–6 give the guarantee G ≥ 1 and the label-free forecast ANT under calibration.
- My check on the cached pools (π = mean of the nine probabilities) shows marked miscalibration. In EEG the lowest π-quintile has mean π = 0.136 but positive rate 0.003. In Churn the disagreement strata have mean π = 0.43 against a positive rate 0.64. In Adult-sex group 1 the mean π is 0.155 against a positive rate 0.109 (similar for Adult-race). In Bank, Credit-card and MAGIC the low quintiles are overestimated by a factor 2–30.
- The estimator remains valid (the theory does not need calibration), but statements such as "the gain is guaranteed and can be computed without labels" (after Prop. 4), and the ANT forecast, are not supported on these data. The paper reports exactly this failure (ANT fails in Law school: forecast 1.3–1.4 vs observed 5–11; correlation 0.62 across estimands without Law school), but the Discussion still says the benefit "can be forecast … before any label is collected".

**(T6) Allocation is nearly flat at small budgets.** For Adult-sex at L = 250 the allocation is (sorted) 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 14, 14, 14, 15, 19, 22 over 19 strata (checked from tcs.labeling.neyman_allocation). The quantity H·Σn_h²/L² (the variance of equal allocation relative to this one, if the floors were not binding) is about 1.05 for Adult-sex and 1.07 for Credit-card at L = 250. So at L = 250–500 the "weighted generalised Neyman rule with proxy variances" is, to within a few per cent, equal allocation over (G,K) strata, and 190 of the 250 labels (76%) are consumed by the floor in a 19-stratum pool. This explains the central finding that "the allocation is insensitive to the proxy", including reversed and shuffled proxies (proxy ablation, "Ablations and robustness"): there is little allocation to be insensitive about. It also means that the claimed contribution "proxy-based allocation for many contrasts at once" is not what is being tested at the budgets where most cells sit. At L = 1000 the allocation is more variable (e.g. 18 to 168), and equal allocation would lose about 46% there (1.46 in the same calculation); this is where the Neyman rule actually earns its keep and where a proxy ablation would be informative.

## 4. Experimental design

**E1. One split, one model set, one seed.** Each data set is split once (seed 0) into 60% training and 40% pool, and nine fixed models are trained once (Appendix A). All efficiencies and coverages are conditional on this single pool/model realisation per setting; Monte Carlo replication (R = 200, 150 in the ablations) is only over label draws. The disagreement structure — the thing everything depends on — is therefore a single draw per setting. The paper says this in Limitations, but the abstract presents medians over eleven "settings" as if they were eleven independent replications. Five of them are four data sets (Adult twice).

**E2. Six of eleven settings have no fairness content.** The "group" in six settings is a median split of the first numeric feature (Sec. 4.1). Arguably this tests the statistics, but it also means the paper's central application (fairness) is evaluated on five settings from four data sets, one of which (Law school) is degenerate (the three best models disagree on < 0.01% of the pool; 51 of 108 contrasts are exact). That leaves four informative fairness settings: Adult-sex, Adult-race, Bank (age < 30 vs. older), Credit-card (sex). No COMPAS, German credit, ACS/Folktables income, MIMIC or other recognised fairness benchmark, no intersectional groups, no models actually tuned for fairness (the candidates are generic classifiers with different inductive biases; their disagreement may differ from that among fairness-intervention models). EEG eye-state and Electricity are time-ordered series split at random; the resulting high accuracy and the (mis)calibration seen in my check are artefacts of this choice and should not be mixed with the "settings".

**E3. Efficiency is summarised by a median over estimands, then by a median over settings, with changing composition.**

- The cell median is taken over 108 estimands, most of which are comparisons between weak or very different models (36 pairs of nine models, including naive Bayes and depth-3 trees). The paper separately reports the "top-3 pairs" (5.8/6.8/6.6/10.5), which is the more decision-relevant quantity and is equal to the headline only coincidentally.
- The number of settings changes with the budget (11, 10, 8, 6 settings at L = 250/500/1000/2000, because of the constraint L ≤ N/4). The headline sequence "5.8, 6.3, 6.9, 15.5" and the statement "the gain grows with the budget" therefore mix a budget effect with a composition effect. I recomputed the PATTERN-DIF efficiency on the seven non-degenerate settings common to L = 250, 500 and 1000: the medians are 5.31, 5.26, 5.98, i.e. essentially flat between 250 and 500 and 12% higher at 1000. The jump to 15.5 at L = 2000 is the six large pools only (including law school at 11,241; without it 9.42, made of Adult-sex, Adult-race, Bank, Credit-card, Electricity).
- Efficiency ratios computed over estimands that are close to degenerate (law school; Churn at L = 500: 60.8) are not meaningful, as the authors say. They should not enter the abstract's headline medians.
- Medians hide the lower tail. The paper reports only "lowest efficiency in any cell outside law school was 3.5". A table with the 10th/25th/75th percentile over estimands, and bootstrap confidence intervals over settings, is needed.

**E4. Efficiency is defined against SRS with a plug-in ratio estimator.**

- The natural baseline for a fairness audit is not SRS; it is group-stratified sampling with proportional or equal allocation, or Neyman allocation on group × true class. The paper's GROUP design allocates with the Neyman rule on the proxy and gets about 1.0 (plain) — i.e. it is no better than SRS, which is surprising for an oversampling design and is worth a sentence of explanation (the paper offers one: disagreements are spread over both groups). A group × proxy-score stratification with the difference estimator (λ = 1) and the same allocation is not in the main comparison (only the tuned-λ StratPPI, which undercovers).
- SRS undefined samples (≤ 0.5%) are dropped from both designs in the MSE ratio. This favours SRS slightly and is conservative; fine.

**E5. Baselines are not placed on an equal footing.**

- (i) The StratPPI baseline uses tuned λ (known to undercover per the authors' own results) and 10 proxy bins × group strata. The fair comparison to isolate the value of the K-strata is "10 proxy bins × group, λ = 1, same allocation", which is not reported in Sec. 5 (the A1 ablation has related variants, which tie, but not this one).
- (ii) The baselines study uses different seeds from the main study; PATTERN-DIF is 6.6 at L = 250 there against 5.8 in Sec. 4, a 14% discrepancy for the same estimator and the same design. This is Monte Carlo noise of the order the authors quote (about 10% per MSE with R = 200). It also means that the reported StratPPI-to-PATTERN ratios of 1.01–1.12 are within noise.
- (iii) The Discussion says StratPPI "did not beat PATTERN-DIF in any of the 35 cells" (median ratio 0.5–0.6). That statement is about the per-estimand ratio of MSEs and I confirm it from the JSON (0 of 35). However, Table 4 (efficiency by setting) shows StratPPI with the larger median efficiency than PATTERN-DIF (plain) in Bank at L = 2000 (24.5 vs 22.3, and it is printed in bold). The two summaries should be reconciled and one metric chosen.
- (iv) B1 (importance sampling) draws with replacement and uses about 1.1 draws per label; B3's design and variance are conditional, and the explanation of why disagreement-only designs do not improve with L ("a plausible reason… we did not isolate this effect") is untested.
- (v) The strongest competitor in the literature (proxy-driven Poisson/fixed-size sampling with a PPI estimator) is not run (see Sec. 2).

**E6. Coverage claims.**

- Mean coverage over estimands and cells (0.940–0.952) averages over many easy estimands. The per-estimand worst case is 0.59 (L = 250), 0.84 (500), 0.89–0.90 (≥ 1000); at L = 250 the minimum is below 0.88 in four cells (Adult-sex 0.85, Law school 0.59, MAGIC 0.825, EEG 0.875, per the JSON), and at L = 500 in Bank (0.84) and Churn (0.86). The statement "reliable for every estimand from about 1,000 labels on" rests on 8 cells.
- With R = 150–200 the Monte Carlo SE of one coverage is about 0.015–0.018, and the minimum over 108 estimands is expected to sit near 0.90 even under exact nominal coverage (as the authors note). The worst-case statistic is thus mostly noise at this R and cannot discriminate between 0.88 and 0.92. R ≥ 1000 for coverage, and a per-estimand-type breakdown (TPR/FPR/ERR, the minority-cell sizes), are needed.
- Coverage is reported for the pool value only (see T2).
- The conclusions on the floors ('zero', 'max') and the conservative interval are in-sample by the authors' own admission.

**E7. Equalised odds, the headline fairness criterion, is not what is estimated.**

- The title keyword and the introduction (the 135,000-label certification figure for the EO gap from the authors' earlier study) emphasise equalised odds; the inferential target (Sec. 2.2) is the signed TPR-gap and FPR-gap difference and the error difference. The EO gap itself is non-smooth (max of absolute values) and is handled only as a plug-in ranking exercise in the ablation A5, where the gain over SRS is about 2 (1.94–2.21), the method is worse than SRS in all Electricity cells (0.57–0.73) and in MAGIC at L = 1000, the plug-in bias is upward (0.024), and the probability of picking the truly best model is 0.50 at L = 250 (0.46 excluding degenerate settings). That is a very different message from "5.8 times fewer labels". The "gap-targeted" allocation is better for EO but gives up 26% for the contrasts.
- Signed gap differences are also hard to interpret for fairness: θ^TPR_mj < 0 means m has the smaller *signed* gap, which can mean a *larger* absolute gap in the opposite direction. The sentence "A model with small gaps treats the two groups more similarly" (Sec. 2.2) does not apply to a negative θ when the gaps have opposite signs. The paper should either target |Δ|-based contrasts (with appropriate non-smooth inference) or say plainly that the main results do not support fairness *rankings*.

**E8. No multiplicity control and no decision study.** The paper reports 108 marginal 95% intervals per cell and Sec. 6 Step 5 states that "a contrast whose interval excludes zero supports a statement that one model has a smaller gap". With 36 pairs × 3 estimand types, such statements are subject to multiplicity and selection (e.g. picking the "best" pair after looking). The section is titled "Inference and decisions" but contains only marginal coverage; no power, no rate of correct sign/ranking, no labels-to-target-width study (which the authors' own literature sweep lists as expected metrics).

**E9. Reproducibility.**

- Positive: cached pools and results are in the repository, the seeding is documented (crc32 of key + 17L + replicate), OpenML versions and the model hyper-parameters are given in Appendix A, and there is a unit test (tests/test_labeling_dif.py: exact-enumeration and Monte Carlo).
- Negative: (i) README.md still describes a different paper ("Trust confidence sets") and its "Reproduce" block does not cover manuscript_v2 (label_efficiency_dif.py, label_baselines.py, label_ablations_dif.py, theory_check*.py, make_v2_assets.py); (ii) no pinned versions (scikit-learn, numpy) although the pools depend on scikit-learn's model implementations and `fetch_openml`; (iii) the pool caches cannot be regenerated deterministically across library versions, so the paper's headline numbers depend on files, not on code plus a pinned environment; (iv) the baselines, the ablations, the theory checks and the main study use different seeds and replicate counts (R = 150, 200, 1000, 2000), which is why numbers differ by 10–15% between sections; (v) the main-study and ablation protocols still carry "an earlier version" of the study that is not reported but constrains random streams. A single script that regenerates every table and figure, with a fixed environment file and one seed policy, is needed.

## 5. Concerns

### MAJOR

**M1. The target of inference and the first-phase variance (T2).** The intervals are valid for the pool value only; for the deployment population, they ignore pool-sampling variance, which is not small relative to the design variance at the paper's own efficiencies (L·Eff/N from about 0.2 to above 2 in the reported cells; implied coverage of a deployment contrast roughly 0.73–0.90 in those cells).
*Requests:*
(a) State the target explicitly in the abstract and Sec. 2.
(b) Add a two-phase variance estimator (pool sampled from a population; Särndal–Swensson) and show how the efficiency claims change when the target is the population contrast.
(c) Run the coverage experiment against a population value: draw pools of size N from the full OpenML data set (or a large held-out reference), run the whole pipeline (train → pool → label), and report coverage of the full-data value. This experiment also addresses E1.

**M2. Evidence rests on one split, one model set and mostly non-fairness groups (E1, E2).** *Requests:*
(a) Repeat the full pipeline over at least 20 random train/pool splits per data set, with model refits, and report between-split variability of efficiency and coverage.
(b) Add real fairness benchmarks beyond Adult/Bank/Credit-card (e.g. ACS income/employment by state and race or sex, COMPAS, German credit) and intersectional groups.
(c) Report fairness-only summaries separately from the arbitrary-group settings.
(d) Treat time-ordered data sets properly or drop them.
(e) Pre-register the exclusion rule for data sets (Nomao was dropped because its arbitrary group had no positives; the earlier study had twelve settings) and the "live estimand" rule.

**M3. Headline numbers are not comparable across budgets and are inflated by degenerate cells (E3).** The statement that the gain grows with the budget is contradicted by the common-settings subset (5.3, 5.3, 6.0 at L = 250/500/1000, checked), and the L = 2000 value is computed on a different set of six large pools. *Requests:*
(a) Report medians on a fixed set of settings across budgets.
(b) Remove Law school and Churn-at-500 from headline medians (or state results with and without in the abstract, which currently gives only the with-law-school figures).
(c) Give inter-quartile ranges over estimands and confidence intervals over settings (bootstrap over settings and replications).
(d) Make the "top-3 pairs" analysis a main result and extend it to the pairs a practitioner would plausibly compare.

**M4. The inference theory does not cover the regime in which the method is used and the coverage evidence at small budgets is weak (T1, T4, E6).** *Requests:*
(a) Replace "reliable on average from 250 labels on" by a statement tied to the measured shortfalls and to stratum sample sizes, or give a theorem that is uniform in ρ and n_h (e.g. a Berry–Esseen-type bound for the stratified sum and a ratio-skewness correction).
(b) Report coverage with R ≥ 1000 and broken down by estimand type, by minority-cell size (expected number of sampled minority positives/negatives) and by stratum sample size.
(c) Try interval constructions appropriate to ratios with few events (studentised/Fieller, or a bootstrap with finite-population correction) rather than only t, floors and percentile bootstrap; the authors have identified the failure mode (SE too small, bias small) but not tested the explanation.
(d) Evaluate the 'zero' floor out of sample (on new data sets or new splits, not those used to design it).

**M5. The allocation and its supposed robustness: at the budgets where most cells lie, allocation ≈ equal allocation (T6).** *Requests:*
(a) Add "equal allocation over (G,K) strata" and "proportional-to-N_h^{1/2}" baselines, and a table of n_h by stratum for representative cells.
(b) Report how many labels the floor consumes and vary the floor (e.g. 5, 10, 20).
(c) Try coarser K (e.g. K ∈ {0}, {1–2}, {3–6}, {7–8}, {9}) to reduce H, since at L = 250, H = 19 strata with ~13 labels each is a poor use of the budget; show the efficiency–H trade-off.
(d) Re-run the proxy ablation (reversed, shuffled, constant) at L = 1000–2000 where the allocation is not flat. Until then the sentence "the allocation is insensitive to the proxy" should be restricted to small budgets.

**M6. Baselines and positioning (E4, E5, Sec. 2).** *Requests:*
(a) Run a one-shot fixed-size proxy-driven design with the same difference estimator (active inference with robust or conditional-Poisson sampling; Zrnic–Candès; Li et al.; Skalnes–Oberst), and a LURE-style estimator.
(b) Add group × proxy-bin strata with the λ = 1 difference estimator and Neyman allocation (the proper control for the K-strata), and an equal-footing comparison with the same seeds as the main study.
(c) Add group-stratified sampling on group × proxy-bin with equal allocation as the "fairness auditor's default".
(d) Reconcile the "35 cells" statement with Table 4.
(e) Cite the missing work listed in Sec. 2 and discuss cross-fitting.

**M7. What is estimated is not what motivates the paper (E7).** Equalised-odds is the motivating criterion, but results for it are weak and mixed, and signed gap differences do not rank fairness. *Requests:*
(a) Either develop inference for the EO gap and for |Δ|-based contrasts (e.g. subsampling, or the bootstrap/m-out-of-n, with a discussion of the non-regular point θ = 0) and report their coverage, or remove the equalised-odds emphasis from the title keywords, abstract and introduction.
(b) State clearly when θ^TPR < 0 does or does not imply a smaller absolute gap.

**M8. Claims about forecastability and guidance outrun the evidence (T5, Sec. 4.4, Sec. 6).** The Discussion says the benefit "can be forecast from the disagreement rate … before any label is collected" and Sec. 6 Step 1 gives threshold rules (< 5% disagreement ⇒ 5.4–15.0; 8.6% ⇒ 3.8–5.0) based on a handful of pools. The evidence: Spearman −0.58/−0.53/−0.71 on 10/9/7 settings; the closed form overstates the nine-model gain by about 2× (Table 3); the label-free forecast ANT fails in the pool with most extreme efficiency; the accurate predictor LIN uses the true labels. *Request:* either provide a label-free predictor of efficiency that is validated out of sample (e.g. fit on some settings, test on others), or soften the claim to "efficiency decreases with disagreement".

**M9. Multiplicity and decision-oriented evaluation (E8).** *Requests:* provide simultaneous intervals (or at least a Bonferroni/Holm or max-t version over the contrasts of interest) and study power and sign-error rates; remove or qualify the Step 5 decision rule.

### MINOR

1. **(A3)/(A4) presentation.** In (A4) the factor N² in "n_h(1−f_h)S²_{e,h}N² → ∞" is a scaling artefact (it is S²_x with x = Ne). State it as such, and add a sentence connecting (A3) to the rare-disagreement regime (T1).
2. **Prop. 5(c)/Eq. (dif-closed).** The closed forms assume |D_i| = 1 on the stratum, D and π uncorrelated within it, and δ_g, γ_g negligible. These are strong and rarely true for strata in the middle K values; say so next to the equations, and clarify that "gains of one to three" is an illustration (the observed gain is 1.3–1.7).
3. **"Calibrated".** Replace "the predictions here are calibrated" (Sec. 3.2, Discussion) by a calibration check (reliability by stratum) and say that the guarantee G ≥ 1 holds under a working model that these data violate.
4. **Efficiency definitions.** State consistently whether "efficiency" is the median of per-estimand MSE ratios (as in the JSON) or the ratio of medians (as in Table 4); the two differ, which causes the StratPPI inconsistency. State the clipping used in the ablation ([1e-4, 1e4]) and its effect on medians.
5. **Law school.** Report it in a separate row and remove ">100" and 10^20-type entries from any summary; state that for it the exclusion of 51 estimands uses pool labels (the "live" rule is label-dependent, although it affects only this pool).
6. **Estimator of the minimum stratum size.** The merge rule (< 50) and the floor (10) are heuristics; the A1 ablation shows they matter (floor consumes the budget). Add a sensitivity analysis.
7. **Bias.** The plug-in ratio has O(r²) bias; the paper reports "bias² share" in the JSON but not in the text. A line stating that bias is negligible relative to variance (it is, in my reading of the JSON: median bias² share ≤ 0.02) would help.
8. **Per-estimand coverage statistics.** Report the fraction of estimands with coverage < 0.90 rather than only the minimum.
9. **Wording.** The abstract states that "most of the gain comes from the disagreement strata", which is true on the measured decomposition (design 3.3–8.6 vs estimator 1.3–1.65), but the decomposition is not additive (the authors note it), so "most" should be quantified (e.g. share of log-efficiency).
10. **Intersection of the two main claims.** The difference estimator gain over the plain estimator is 1.3–1.7 and comes largely from the known structure of D_i (the paper shows that a constant proxy keeps most of the gain for rate differences), not from the models' probabilities. This is a valuable observation and deserves a sentence in the abstract: "the proxy matters little; the structure does".
11. **Typography/consistency.** Sec. 4.1 says the earlier SRS/GROUP/PATTERN samples "coincide with an earlier version of this study"; the paper repeatedly refers to unreported earlier studies (earlier plug-in study, earlier certification analysis). A reader cannot check these; either include the minimal needed results in the appendix or drop the references.
12. **Software.** The repository README should describe manuscript_v2, pin the environment and give a one-command reproduction of all tables.

## 6. Recommendation

**Major revision.**

*Reasons.* The problem is real and practically relevant, and the design is sensible, well described, and transparent about its non-novelty. The design-based algebra and the proofs, as stated, are correct (I found no error in the unbiasedness, linearisation, normality, influence-value, allocation or Kantorovich results). The simulation checks of the theory (Sec. 5) are careful, and the authors are unusually frank about failures (floors designed post hoc, StratPPI beating PATTERN, ANT failing, coverage 0.59). The paper reads as an honest empirical study of a well-motivated combination of known tools. However:

- The inferential target is the finite pool, with inference that does not carry over to the deployment population at the paper's own efficiencies.
- The evidence for generality is a single split of eleven settings (six with arbitrary groups, one degenerate).
- The headline medians are composition-dependent.
- The proxy-based allocation, which is presented as a contribution, is essentially equal allocation at small budgets.
- The strongest competitor class (proxy-driven active inference and score strata with λ = 1) is absent.
- The equalised-odds criterion that motivates the paper is not what is estimated.

None of these is fatal, but together they make the present claims ("5.8–6.9 times fewer labels", "reliable intervals from 250 labels") stronger than the evidence.

## 7. Which concerns, if left unaddressed, would most likely lead to rejection

1. **M1 (target of inference / first-phase variance), together with M2 (single split, arbitrary groups, few real fairness settings).** A statistics referee will see that coverage is reported only against the pool value on a single realisation of pool and models, and that the headline efficiencies imply sampling error in the pool larger than the design error. Without a population-level coverage experiment and repeated splits the central inferential claim is unsupported.
2. **M4 (small-budget coverage and the asymptotic regime).** Worst-estimand coverage of 0.59–0.85 at L = 250–500, an assumption (A3) that is violated in the rare-disagreement regime of interest, and a floor tuned in-sample. A claim of "valid design-based intervals" cannot stand without either a uniform result or a clean out-of-sample demonstration.
3. **M6 (baselines and novelty).** With every ingredient known, the contribution is the combination, so it has to be shown that the combination beats the obvious alternatives on equal footing. Missing active-inference/fixed-size PPI baselines and the missing score-strata-with-λ = 1 control make this unconvincing, and a reviewer familiar with PPI will say so first.
4. **M3 (composition and degenerate cells in the headline).** The abstract's trend over budgets does not survive a fixed set of settings. Reviewers who recompute it will lose trust in the other numbers.
5. **M7 (equalised odds).** If the paper keeps equalised-odds in the title keywords and motivation while the results for the EO gap are weak and the signed-gap contrasts do not rank fairness, a referee from the fairness side will consider the contribution mis-sold.
