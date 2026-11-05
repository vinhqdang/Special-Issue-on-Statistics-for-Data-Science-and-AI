Quang-Vinh Dang
British University Vietnam, Hung Yen, Vietnam
vinh.dq4@buv.edu.vn

5 November 2026

The Editors
International Journal of Data Science and Analytics
Special Issue on Statistics for Data Science and AI
(Prof. Paola Cerchiello, Prof. Tomaso Aste, Prof. Kian Guan Lim)

**Submission: "Labelling where models disagree: label-efficient comparison of classifiers on group fairness" (Regular Paper)**

Dear Professors Cerchiello, Aste and Lim,

I submit the enclosed manuscript for consideration in the special issue on Statistics for Data Science and AI.

**The problem.** The call asks for statistical methods that measure the trustworthiness of machine-learning models in real applications. Group-fairness criteria such as equal opportunity and equalised odds depend on the true outcome, so every comparison between candidate models needs labels, and under simple random sampling the comparison of fairness needs far more labels than the comparison of accuracy: group rates have small denominators, and the quantities of interest are differences of group gaps between models. The manuscript asks how a fixed labelling budget should be spent on an unlabelled pool when several trained models are to be compared on group-specific true and false positive rates and on error.

**The contribution.** Two facts are exploited: models differ only where their predictions differ, which is rare among competent models, and group membership is known before labelling. The proposed design (PATTERN-DIF) stratifies the pool by group and by the number of models predicting the positive class, allocates labels by a weighted generalised Neyman rule that uses the models' probabilities as proxies, and estimates all pairwise contrasts with a stratified difference estimator and linearisation variance. The manuscript gives design-based theory (unbiasedness for any proxy, asymptotic normality, the influence-value structure that explains why disagreement strata carry the information, the condition under which the difference estimator beats the plain one, the optimal allocation and its robustness to a misspecified proxy), and a testing layer: Wald, non-inferiority and equivalence tests with asymptotic size, a power function whose efficiency ratio equals the design effect (so that the labels needed for a given power shrink by that factor), strong asymptotic control of the familywise error rate for simultaneous tests of all contrasts, and the allocation that maximises worst-case power, with proofs in the Supplementary Information.

I state the novelty narrowly. Sampling where models disagree, stratification on classifier scores, prediction-powered and model-assisted estimation, and discordant-pair designs in diagnostic testing are all established, and the manuscript cites and compares with them. What I contribute is their combination and adaptation to ratio-type group-fairness contrasts across several models under a fixed budget, the associated theory and tests, and empirical evidence on public data.

**Evidence.** On eleven data settings with nine models each, with the held-out labels used as ground truth, the median efficiency relative to simple random sampling is 5.8, 6.3 and 6.9 at 250, 500 and 1000 labels, against 3.7, 3.9 and 4.4 for the plain estimator on the same design and about 1.0 for stratification by group alone. Most of the gain comes from the design rather than from the estimator. Simulated tests have nominal size (rejection rates 0.048 to 0.053 at the 5% level), and the number of labels needed for equal power matches the theoretical prediction to within 20%. The manuscript also reports where the method falls short: the worst-covered contrast has coverage 0.59 at 250 labels in one setting, the simultaneous max-t procedure has familywise error slightly above nominal (0.058) in simulation, six of the eleven settings use arbitrary groups and therefore carry no fairness meaning, and the evidence rests mostly on one train/pool split per setting (three further splits for the six largest settings).

**Fit with the special issue.** The paper is a statistical contribution to making the measurement of fairness in machine learning accurate and reliable: it supplies confidence statements and tests with stated error control, and tells practitioners how many labels a comparison needs.

**Files and availability.** The submission consists of the main paper and a separate Supplementary Information file (proofs, additional theory, simulations, baseline and ablation details, data details), prepared with the Springer Nature LaTeX template (two-column option). All data sets are public (OpenML), and all code, simulation outputs and sources are available at https://github.com/vinhqdang/Special-Issue-on-Statistics-for-Data-Science-and-AI.

**Declarations.** [Author to confirm before submission: the manuscript is original, has not been published and is not under consideration elsewhere; there are no competing interests (or state them); funding statement; use of AI tools, if any, as required by the journal's policy.]

**Suggested reviewers (optional).** [Author to complete, or delete this paragraph.]

Thank you for considering this submission. I look forward to your decision.

Sincerely,

Quang-Vinh Dang
British University Vietnam
vinh.dq4@buv.edu.vn
