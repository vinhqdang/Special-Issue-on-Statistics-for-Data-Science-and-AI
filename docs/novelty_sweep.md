# Novelty sweep for the trust confidence set (TCS)

Date of sweep: 3 November 2026.

Target claim under test: a paired-bootstrap, studentised, intersection-union confidence set that contains, with asymptotic probability at least 1-alpha, every population model that is not strictly dominated (worse on every criterion) by another model, for a finite list of trained models, a vector of lower-is-better trust criteria, and one common test sample.

## How this was done and what the limits are

- Web searches (about 25 queries) over: tests of Pareto efficiency and multivariate or stochastic dominance (econometrics, statistics); inference and confidence regions for Pareto sets and fronts (multi-objective optimisation, ranking and selection, bandits); multi-loss or multi-criteria extensions of the model confidence set, the reality check and the superior predictive ability test; multi-metric significance testing of ML models; fairness-accuracy trade-off uncertainty; "uncertain / robust / statistical Pareto" terms.
- Every item below was checked against at least one primary record: Crossref (DOI record), the arXiv API, a PMLR / NeurIPS / publisher page, or the paper's own full text (downloaded and read where marked "full text read").
- "Summary basis" says how much of each paper I actually saw. Where I saw only an abstract or a search-engine description I say so. Nothing here was completed from memory.
- Not covered: Google Scholar, Scopus and Web of Science were not reachable, and the search tool is US-only and snippet-based. Several relevant literatures (ranking-and-selection, stochastic-dominance efficiency in finance) are large; the sweep is systematic but not exhaustive. Re-run close to submission.

## 1. Closest prior work (read these first)

### 1.1 Zhao and Eckman, one-shot screening of simulated systems (closest in logic)

- Reference: Zhao, J. and Eckman, D. J. (2026). One-Shot Screening of Simulated Systems for Acceptability. arXiv:2609.04400 (v1, 3 September 2026), stat.AP. Preprint, no journal venue listed.
- Link: https://arxiv.org/abs/2609.04400
- Summary basis: full text read (abstract, Sections 2-4, experiments).
- What it does: for a finite set of simulated systems with d responses, it retains a system if it is acceptable under some plausible configuration inside a confidence region for the unknown mean-response matrix. For acceptability defined as Pareto optimality (non-dominated in all responses) it proves a "set-wise probability of acceptable selection" guarantee, P(A(M0) subset of S) >= 1-alpha, where A(M0) is the efficient set, and notes this yields a 1-alpha confidence region for the Pareto front. The practical procedure uses box confidence regions (Bonferroni or Sidak across responses, normality assumption) and keeps system i if its lower confidence vector is not dominated by the upper confidence vector of any rival j.
- Closeness to TCS: very high in target and logic. Set-wise coverage of the whole efficient set, finite candidate list, "retain unless certifiably dominated" is exactly the TCS target. This is the single most important item for the novelty statement.
- Differences: (i) setting is discrete-event simulation ranking and selection, not trained ML models on a test sample; (ii) construction is a union of marginal box intervals with Bonferroni/Sidak, not a paired studentised bootstrap of pairwise differences with a max-type critical value; the paper treats common random numbers only as an aside; (iii) dominance is the usual weak Pareto relation (at least as good everywhere, strictly better somewhere), not strict dominance on every criterion; (iv) normality of outputs assumed, with no treatment of non-smooth functionals such as ECE or an equalised-odds maximum; (v) no trust criteria, no fairness, calibration or robustness application; (vi) it is a preprint.

### 1.2 paretoinfer R package (same target, software only)

- Reference: Leite, A., Ospina, R. and Ferraz, C. paretoinfer: sequential identification of epsilon-Pareto-optimal alternatives. R package, version 0.1.0 (build timestamp 24 October 2026), GPL (>= 3).
- Link: https://cran.r-universe.dev/paretoinfer/doc/manual.html
- Summary basis: package manual read in full via fetch. I found no associated paper or arXiv record.
- What it does: builds simultaneous anytime-valid confidence sequences (betting-type, bounded outcomes) for every objective of every alternative, level alpha/(K m). With probability >= 1-alpha every certificate is correct. It reports a "plausible" set (alternatives not certified dominated), a frontier of point estimates among plausible alternatives, "certified optimal" and "uncertain" alternatives.
- Closeness to TCS: the "plausible set" is the TCS target in different words (not certified dominated), with a simultaneous guarantee.
- Differences: sequential, bounded outcomes, marginal Bonferroni boxes per alternative and objective, so no pairing and no studentised pairwise differences; epsilon-dominance; no ML trust-criteria context; no statistics of functionals of the empirical distribution (ECE, gap measures).

### 1.3 Pareto set identification with fixed-confidence guarantees (bandits and ranking and selection)

These deliver "return the Pareto set (or an epsilon-relaxation) with probability >= 1-delta". They are the main anticipation of the idea that a statistical guarantee on a Pareto set of a finite list exists.

- Auer, P., Chiang, C.-K., Ortner, R. and Drugan, M. (2016). Pareto Front Identification from Stochastic Bandit Feedback. Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 51:939-947. https://proceedings.mlr.press/v51/auer16.html. Basis: PMLR page (abstract). Confidence-bound algorithm to identify the set of operating points not dominated in mean; sample-complexity bounds depending on geometry; adaptive sampling, not a fixed test sample.
- Kone, C., Kaufmann, E. and Richert, L. (2023). Adaptive Algorithms for Relaxed Pareto Set Identification. Advances in Neural Information Processing Systems 36, pp. 35190-35201 (Crossref DOI 10.52202/075280-1528); arXiv:2307.00424. https://arxiv.org/abs/2307.00424. Basis: arXiv record and search description. Fixed-confidence identification of the Pareto set or relaxations, one sampling rule with several stopping rules; applied to vaccine strategies on several immunogenicity criteria.
- Crepon, E., Garivier, A. and Koolen, W. M. (2024). Sequential learning of the Pareto front for multi-objective bandits. AISTATS, PMLR 238:3583-3591. https://proceedings.mlr.press/v238/crepon24a.html. Basis: PMLR page. Stopping-rule algorithm returning the Pareto-optimal arms with probability >= 1-delta, asymptotically optimal sample complexity.
- Ararat, C. and Tekin, C. (2023). Vector Optimization with Stochastic Bandit Feedback. AISTATS, PMLR 206:2165-2190; arXiv:2110.12311. https://proceedings.mlr.press/v206/ararat23a.html. Basis: PMLR page. (epsilon, delta)-PAC identification of the Pareto set for polyhedral ordering cones.
- Andradottir, S. and Lee, J. S. (2021). Pareto set estimation with guaranteed probability of correct selection. European Journal of Operational Research 292(1):286-298. DOI 10.1016/j.ejor.2020.10.021. Basis: Crossref record and bibliographic description only; abstract not retrieved. Ranking-and-selection procedure with a guaranteed probability of correct selection of the Pareto set.
- Closeness to TCS: the guarantee type (probability >= 1-delta of containing/identifying the Pareto set of a finite list) is the same family. Differences: all are designed around sequential or adaptive sampling with indifference-zone or PAC machinery, and most aim at the exact or approximate set rather than a superset that contains the non-dominated models; none works with a one-shot paired bootstrap on a fixed shared test sample of instances, none handles functional criteria such as ECE or equalised-odds gaps, and none addresses trust criteria.

### 1.4 Auerbach, Liang, Okumura and Tabord-Meehan, testing fairness-accuracy improvability

- Reference: Auerbach, E., Liang, A., Okumura, K. and Tabord-Meehan, M. (2024). Testing the Fairness-Accuracy Improvability of Algorithms. arXiv:2405.04816 (submitted 8 May 2024). Preprint; no journal venue found.
- Link: https://arxiv.org/abs/2405.04816
- Summary basis: abstract plus targeted reading of the extracted full text (null hypothesis, intersection-union and bootstrap sections).
- What it does: an econometric framework to test the null that a given algorithm cannot be improved on fairness without compromising other pre-specified objectives (for example accuracy), within a class of algorithms. The test is a union of component nulls handled by the intersection-union method, with bootstrap critical values, sample splitting, large-sample validity and consistency. Applied to a healthcare algorithm.
- Closeness to TCS: high in spirit and ingredients (econometric Pareto-type test, fairness plus accuracy, intersection-union, bootstrap, same test data).
- Differences: it tests whether one algorithm is Pareto-improvable over a class (null is "not improvable"), with sample splitting to pick the improving candidate; it does not return a confidence set of non-dominated models from a fixed candidate list; two or three objectives with a fairness-improvement structure; no model-confidence-set framing.

### 1.5 Optimal predictive ability and robust forecast comparison (econometrics, multi-loss extension of SPA)

- Arvanitis, S., Post, T., Poti, V. and Karabati, S. (2021). Nonparametric tests for optimal predictive ability. International Journal of Forecasting 37(2):881-898. DOI 10.1016/j.ijforecast.2020.10.002 (Crossref). Basis: Crossref plus abstract and search-extracted descriptions; full text not read.
  - What it does (per abstract): generalises the superior predictive ability hypothesis from a single loss function to a whole class of loss functions; the null is cast as moment inequalities and tested with a blockwise empirical-likelihood ratio statistic. Related descriptions state that optimality across a class is stronger than Pareto efficiency, and that a model is efficient (non-dominated) iff some loss in the class prefers it to some competitor.
  - Precursor: Post, T., Poti, V. and Karabati, S. (2018). Nonparametric Tests for Superior Predictive Ability. SSRN working paper, DOI 10.2139/ssrn.3251944 (Crossref).
- Jin, S., Corradi, V. and Swanson, N. R. (2017). Robust Forecast Comparison. Econometric Theory 33(6):1306-1351. DOI 10.1017/s0266466616000426 (Crossref lists the issued year as 2016, probably online-first). Basis: Crossref and search description only. Forecast comparison robust to the choice of loss within general or convex classes, using a stochastic-dominance-type characterisation.
- Closeness to TCS: medium. These are the true econometric descendants of the reality check and SPA that go beyond a single loss; they are multi-loss, with a Pareto-efficiency interpretation, and test a null on a common forecast sample. Differences: the vector is a continuum or class of scalarising loss functions applied to the same forecast errors (one error series per model), not a finite vector of heterogeneous criteria such as fairness gap, ECE and robustness drop; targets optimality or superiority of a benchmark rather than a confidence set of non-dominated models; time-series forecasting setting.
- Action: the manuscript's sentence "those tools handle one loss function" is not accurate for the generation after SPA. These papers must be cited and the contrast stated correctly. I could not read them in full, so any statement of exact equivalence (for example whether a Pareto-efficiency test inside this framework coincides with the TCS target) must be verified from the full text before submission.

### 1.6 GSD-front, statistical multicriteria benchmarking (ML, NeurIPS 2024)

- Reference: Jansen, C., Schollmeyer, G., Rodemann, J., Blocher, H. and Augustin, T. (2024). Statistical Multicriteria Benchmarking via the GSD-Front. Advances in Neural Information Processing Systems 37, pp. 98143-98179 (Crossref DOI 10.52202/079017-3115); arXiv:2406.03924.
- Link: https://arxiv.org/abs/2406.03924
- Summary basis: abstract plus extracted full text (hypothesis and test sections).
- What it does: compares classifiers on several metrics across a benchmark suite using generalised stochastic dominance, proposes the GSD-front as a less conservative alternative to the Pareto front, gives a consistent estimator and permutation tests for whether a classifier is in the GSD-front (static and dynamic versions, level-alpha proven), plus robustness to violations of i.i.d. sampling of datasets using imprecise probabilities.
- Closeness to TCS: medium. It is the best-known ML paper that adds statistical tests to multi-criteria comparison and explicitly criticises the plain Pareto front.
- Differences: the sampling unit is the dataset in a benchmark suite, not test instances on one shared test set; the order is generalised stochastic dominance, not Pareto dominance of population criteria; the output is a test of whether one classifier is in the (GSD) front, not a simultaneous confidence set for the non-dominated set; no fairness or calibration functionals on a single test sample.

## 2. Related inference-for-sets and dominance-testing literature

### 2.1 Tests of dominance and efficiency in econometrics and statistics

All verified through Crossref; summary basis for each is the bibliographic record plus the descriptions returned by search.

- Davidson, R. and Duclos, J.-Y. (2000). Statistical Inference for Stochastic Dominance and for the Measurement of Poverty and Inequality. Econometrica 68(6):1435-1464. DOI 10.1111/1468-0262.00167.
- Barrett, G. F. and Donald, S. G. (2003). Consistent Tests for Stochastic Dominance. Econometrica 71(1):71-104. DOI 10.1111/1468-0262.00390.
- Linton, O., Maasoumi, E. and Whang, Y.-J. (2005). Consistent Testing for Stochastic Dominance under General Sampling Schemes. Review of Economic Studies 72(3):735-765. DOI 10.1111/j.1467-937x.2005.00350.x.
- Scaillet, O. and Topaloglou, N. (2010). Testing for Stochastic Dominance Efficiency. Journal of Business and Economic Statistics 28(1):169-180. DOI 10.1198/jbes.2009.06167. Tests whether a given portfolio is dominance-efficient relative to all portfolios built from a set of assets, with block-bootstrap inference.
- Batana, Y. and Duclos, J.-Y. (2008). Multidimensional Poverty Dominance: Statistical Inference and an Application to West Africa. SSRN working paper, DOI 10.2139/ssrn.1139916 (Crossref). Per search description it uses an intersection-union test to infer strict multidimensional dominance, with bootstrap.
- Verdict: these establish (a) bootstrap tests of dominance, (b) intersection-union tests of strict dominance across coordinates, (c) tests of efficiency of a given object relative to a set. The TCS borrows (a) and (b) in spirit, so the intersection-union device itself is not new. They compare distributions or portfolios by dominance over a grid or class of utilities, not a vector of scalar population criteria for a finite model list, and they do not produce a simultaneous confidence set of non-dominated models.
- Multivariate stochastic dominance for ML benchmarking: Rioux, G., Nitsure, A., Rigotti, M., Greenewald, K. and Mroueh, Y. (2024), Multivariate Stochastic Dominance via Optimal Transport and Applications to Models Benchmarking, arXiv:2406.06425 (published at NeurIPS 2024 per the proceedings listing found by search); and Ma, Y., Liu, H. and Zhuang, W. (2025), Multidimensional Stochastic Dominance Test Based on Center-outward Quantiles, arXiv:2512.19966. Both give bootstrap-valid pairwise multivariate dominance statistics on joint metric distributions. Pairwise tests, not a confidence set over a model list.

### 2.2 Confidence sets, model selection and ranks

- Hansen, P. R., Lunde, A. and Nason, J. M. (2011), Econometrica 79(2):453-497 (model confidence set); White, H. (2000), A Reality Check for Data Snooping, Econometrica 68(5):1097-1126, DOI 10.1111/1468-0262.00152; Hansen, P. R. (2005), A Test for Superior Predictive Ability, JBES 23(4):365-380, DOI 10.1198/073500105000000063; Romano, J. P. and Wolf, M. (2005), Stepwise Multiple Testing as Formalized Data Snooping, Econometrica 73(4):1237-1282, DOI 10.1111/j.1468-0262.2005.00615.x. All verified on Crossref. I found no multivariate or vector-valued MCS in the literature. The MCS R package accepts user-supplied losses but each run is single-loss; applied papers run it once per loss and compare sets (search description; not a methodological extension).
- Mogstad, M., Romano, J. P., Shaikh, A. M. and Wilhelm, D. (2024). Inference for Ranks with Applications to Mobility across Neighbourhoods and Academic Achievement across Countries. Review of Economic Studies 91(1):476-518. DOI 10.1093/restud/rdad006 (Crossref). Marginal and simultaneous confidence sets for ranks of many populations. Analogous in spirit (simultaneous confidence set for a ranking-type object) but univariate. Related: Petrou-Zeniou, A. and Shaikh, A. M. (2024), Inference After Ranking with Applications to Economic Mobility, arXiv:2410.19212.
- Verdict: no vector-valued model confidence set found; the ranks literature is a good citation for the idea of confidence sets for rank-type objects.

### 2.3 Confidence regions and uncertainty for Pareto fronts, and multi-objective ML selection

- Fallah, A., Jordan, M. I. and Ulichney, A. (2025). The Statistical Fairness-Accuracy Frontier. arXiv:2508.17622 (v1 25 August 2025; page lists later revisions). https://arxiv.org/abs/2508.17622. Basis: abstract. Finite-sample, worst-case-optimal estimation of the fairness-accuracy Pareto frontier over a model class, with uniform bounds along the frontier. Targets the frontier of all attainable models, not a set of named candidate models; theory is minimax rather than a confidence set.
- Laufer-Goldshtein, B., Fisch, A., Barzilay, R. and Jaakkola, T. (2022/2023). Efficiently Controlling Multiple Risks with Pareto Testing. arXiv:2210.07913; ICLR 2023 (venue per the conference poster page found by search). Builds a Pareto frontier of candidate configurations and runs multiple hypothesis testing on that frontier to select configurations whose risks are controlled (distribution-free, finite sample). Different target: risk-controlled selection against thresholds, not a confidence set of non-dominated models; Pareto frontier is used to order tests.
- Zhang, T., Georgiopoulos, M. and Anagnostopoulos, G. C. (2018). Pareto-Optimal Model Selection via SPRINT-Race. IEEE Transactions on Cybernetics 48(2):596-610. DOI 10.1109/tcyb.2017.2647821 (Crossref). Already cited in the manuscript; racing with sequential tests, fixed-confidence.
- Currie, C. S. M. and Monks, T. (2021). A Practical Approach to Subset Selection for Multi-objective Optimization via Simulation. ACM Transactions on Modeling and Computer Simulation 31(4), article 20, pp. 1-15. DOI 10.1145/3462187. Basis: Crossref abstract. Two-stage bootstrap procedure (BootComp) returning a subset of good designs under multiple objectives, works with any output distribution and common random numbers. Closest bootstrap-and-pairing precedent in multi-objective selection; the abstract states no coverage theorem for the Pareto set, and it is a simulation-optimisation procedure.
- Lee, L. H., Chew, E. P., Teng, S. and Goldsman, D. (2010). Finding the non-dominated Pareto set for multi-objective simulation models. IIE Transactions 42(9):656-674. DOI 10.1080/07408171003705367; Applegate, E. A., Feldman, G., Hunter, S. R. and Pasupathy, R. (2020). Multi-objective ranking and selection: optimal sampling laws and tractable approximations via SCORE. Journal of Simulation 14(1):21-40 (Crossref lists issue date 2019, DOI 10.1080/17477778.2019.1633891); Rojas Gonzalez, S., Branke, J. and van Nieuwenhuyse, I. (2022), Bi-objective Ranking and Selection Using Stochastic Kriging, arXiv:2209.03919. Multi-objective ranking and selection for correctly identifying the Pareto set (budget allocation, probability of correct selection, opportunity cost). Same conceptual target as 1.3, simulation context.
- Zeng, L., Chen, X., Wang, Y. and Chen, X. (2026). Shorter, but Still Trustworthy? An Empirical Study of Chain-of-Thought Compression. arXiv:2604.04120. Appendix G (read in extracted full text) resamples test examples in 10,000 paired bootstrap replicates, recomputes metrics and reports frequency of Pareto-frontier membership and exact-set recovery. This is the TCS "Pareto-membership probability" companion index, used descriptively with no coverage claim. Shows that item is not novel on its own.
- Other uncertainty-of-front papers, found but peripheral: Tu, B., Kantas, N., Lee, R. M. and Shafei, B. (2024), Random Pareto front surfaces, arXiv:2405.01404 (Bayesian posterior over fronts); Xu, L. and Jiang, W. (2025), Population-Based Search Method Using Uncertainty-related Pareto Front for Robust Multi-objective Optimization, arXiv:2510.16386 (probabilistic robustness of points under input perturbation); Borchert, O., Salinas, D., Flunkert, V., Januschowski, T. and Guennemann, S. (2022), Multi-Objective Model Selection for Time Series Forecasting, arXiv:2202.08485 and Williams, P., Kendall, W. and Hooten, M. (2018), Model Selection using Multi-Objective Optimization, arXiv:1810.10669 (Pareto-based model selection with no inference). None provides a confidence guarantee on the non-dominated set.

### 2.4 Multi-metric comparison of ML models, fairness uncertainty

- de Campos, C. P. and Benavoli, A. (2016). Joint Analysis of Multiple Algorithms and Performance Measures. New Generation Computing 35(1):69-86. DOI 10.1007/s00354-016-0005-8 (Crossref). Joint frequentist (likelihood ratio) and Bayesian (Dirichlet) tests for comparing algorithms across several measures without a scalar score. Works on win/loss patterns across problems; pairwise; no population-criteria Pareto set.
- Cherian, J. J. and Candes, E. J. (2023). Statistical Inference for Fairness Auditing. arXiv:2305.03712. Bootstrap-based simultaneous bounds on performance disparities over many subgroups for one model; peripheral (inference over groups, not over models and criteria).
- Du, W. (2025). When +1% Is Not Enough: A Paired Bootstrap Protocol for Evaluating Small Improvements. arXiv:2511.19794. Paired bootstrap and sign-flip test for accuracy improvements; extension to several metrics noted as open. Peripheral.
- Items already in the manuscript (Miller et al. 2021; Ferrara et al. 2025; Bouthillier et al. 2021; Benavoli et al. 2017; Demsar 2006; Dietterich 1998) remain correct; they treat one metric or one model.

## 3. Verdict by the three sweep areas

1. Tests of Pareto efficiency and multivariate or stochastic dominance (econometrics, statistics). Established: bootstrap tests of dominance, intersection-union tests of strict dominance across dimensions, tests of whether a given object is dominance-efficient relative to a set (Scaillet-Topaloglou; Batana-Duclos; Arvanitis-Post-Poti-Karabati). Not found: a simultaneous confidence set for the non-dominated members of a finite list under a vector of heterogeneous scalar criteria.
2. Confidence regions for Pareto sets. This is where prior work is closest. Set-wise coverage guarantees for the efficient set of a finite list exist in ranking and selection (Zhao and Eckman 2026 preprint; Andradottir and Lee 2021) and in bandit Pareto set identification (Auer et al.; Kone et al.; Crepon et al.; Ararat and Tekin), plus a CRAN package (paretoinfer) implementing a "plausible Pareto set". All are built on marginal confidence boxes or sequential stopping rules, none on a studentised paired bootstrap of pairwise differences over a shared test sample.
3. Multi-loss or multi-criteria extensions of the MCS, SPA and reality check, and multi-metric ML significance testing. No vector-valued model confidence set found. Multi-loss generalisations of SPA exist through classes of scalarising losses (Arvanitis et al.; Jin-Corradi-Swanson). ML multi-metric tests exist but are pairwise or across datasets (de Campos and Benavoli; GSD-front; Rioux et al.). Fairness-plus-accuracy econometric testing exists for improvability of one algorithm (Auerbach et al.).

## 4. Overall assessment

Is the TCS claim novel, partly anticipated, or already done? Partly anticipated. The statistical idea "return all systems that cannot be certified as dominated, with a set-wise 1-alpha guarantee" is already in the literature for finite candidate lists: it is the central construction of Zhao and Eckman (2026, preprint) and is implemented in the paretoinfer package, and fixed-confidence Pareto set identification gives the same guarantee type in the bandit and ranking-and-selection literatures. The intersection-union device for strict dominance and bootstrap critical values for maxima of studentised differences are also standard. The manuscript's current statement that SPRINT-Race "is, to our knowledge, the closest" and that no confidence set was found is therefore out of date and must be revised.

What I did not find, and which therefore remains a defensible contribution:

- A paired, studentised bootstrap confidence set built from pairwise differences on one common test sample (not marginal boxes or sequential sampling), with an explicit proof under bootstrap consistency, aimed at a strict-dominance target for which exclusion can be certified by a test.
- Its use for vector-valued ML trust criteria that are functionals of the empirical distribution (error, equalised-odds gap, ECE, robustness drop), positioned as the multi-criteria counterpart of the model confidence set.
- The empirical demonstration that the Pareto front of point estimates fails to retain non-dominated models, in a design with a tied model, and that on two public benchmarks no model can be eliminated.

Caveats that limit even this claim: the simulation uses mean-type criteria only; the econometric optimal-predictive-ability papers and the Zhao-Eckman paper could, in a full-text comparison, cover more than I could verify (I read Zhao-Eckman, GSD-front and parts of Auerbach et al. in full text, but not the Arvanitis et al. or Jin et al. papers); and the "Pareto-membership probability" is already used descriptively (Zeng et al. 2026).

### Defensible wording

Defensible (to our knowledge, limited to the literature reviewed):

"We propose a paired-bootstrap, studentised, intersection-union confidence set for the set of population models that are not strictly dominated on a vector of trust criteria, for a fixed list of trained models evaluated on a common test sample. Confidence sets with set-wise guarantees for Pareto-optimal alternatives exist for simulated systems and bandits, built on confidence boxes or sequential sampling; to our knowledge a paired bootstrap construction for functionals of a shared test sample, with application to trustworthiness criteria (error, fairness gap, calibration, robustness), has not been proposed."

Not defensible:

- "the first confidence set for the Pareto set" or "no statistical guarantee exists for Pareto selection" (contradicted by items in 1.1-1.3).
- "no confidence set for Pareto-optimal models exists" without qualification.
- "existing MCS-type tools handle a single loss function" (contradicted by the optimal predictive ability literature).
- Presenting the Pareto-membership probability or the intersection-union step as contributions.

### Suggested manuscript edits (not applied; this file is the only change)

1. Replace the novelty sentence in Related work (Multi-objective model selection) with the narrower statement above, and add Zhao and Eckman (2026), paretoinfer, Auer et al., Kone et al., Andradottir and Lee, Auerbach et al., Arvanitis et al., Jansen et al., Scaillet and Topaloglou, Batana and Duclos.
2. Correct the Introduction sentence on single-loss tools, and the Limitations item (vi), to name what is now known.
3. State the technical differences explicitly: pairwise paired differences with a max critical value versus marginal Bonferroni boxes; strict dominance versus weak Pareto dominance; fixed shared test sample versus sequential sampling; functional criteria.
4. A head-to-head simulation against the box-based set-wise procedure (Zhao-Eckman style, Bonferroni over criteria and models) would show whether the paired bootstrap actually buys power; this is the comparison a referee familiar with ranking and selection will ask for.
5. Re-run this sweep close to submission; the 2026 preprint and package indicate the area is moving.
