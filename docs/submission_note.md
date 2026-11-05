# Submission note

**Status:** submitted (reported by the author on 5 November 2026).
**Journal:** International Journal of Data Science and Analytics (Springer), special issue "Statistics for Data Science and AI"; editors P. Cerchiello, T. Aste, K. G. Lim.
**Article type:** Regular paper.
**Title:** Labelling where models disagree: label-efficient comparison of classifiers on group fairness.
**Author:** Quang-Vinh Dang, British University Vietnam (vinh.dq4@buv.edu.vn).
**Manuscript ID / submission date in the system:** [fill in from the confirmation email]

## Version submitted
- Repository state at the last commit before submission: `67c2243` on `main` (https://github.com/vinhqdang/Special-Issue-on-Statistics-for-Data-Science-and-AI).
- Files: `manuscript_v2/springer/manuscript_sn.tex` (main paper, 42 pages in the two-column Springer Nature template) and `manuscript_v2/springer/supplement_sn.tex` (Supplementary Information, 53 pages), with their figures and `references.bib`; PDFs built with the official template (December 2024): `manuscript_sn_draft.pdf`, `supplement_sn_draft.pdf`.
- Cover letter: `submission/cover_letter.md` / `.docx`.
- Regenerate: `python manuscript_v2/build_springer_v2.py /path/to/sn-article-template`.

## What the paper claims (for answering reviewers)
- Method: PATTERN-DIF = stratification by group x number of models predicting 1, proxy-based weighted Neyman allocation, stratified difference estimator, linearisation intervals.
- Main result: median efficiency vs simple random sampling 5.8 / 6.3 / 6.9 at 250 / 500 / 1000 labels (11 / 10 / 8 settings; `results/label_efficiency_dif*.json`); plain estimator on the same design 3.7 / 3.9 / 4.4; group strata alone about 1.0. The design contributes more than the estimator.
- Theory: unbiasedness for any proxy, asymptotic normality, influence-value structure, conditions for the difference estimator to win, optimal allocation; tests with asymptotic size, power function and Pitman efficiency = design effect, strong FWER control for max-t/Holm, maximin-power allocation (proofs in the Supplement; adversarially checked, `docs/testing_proof_check.md`).
- Novelty is stated narrowly: ingredients are known (Sawade et al. 2012; stratified PPI; survey sampling; discordant designs); the contribution is the combination, the theory/tests and the evidence.

## Known weaknesses (likely reviewer points)
1. Evidence: one train/pool split per setting (three extra splits for six settings), nine models, six of eleven settings with arbitrary groups, law school degenerate.
2. Small budgets: worst-contrast coverage 0.59 at 250 labels (law school); the variance floor was designed after seeing the failures (in-sample).
3. Max-t slightly above nominal FWER (0.058) in simulation; on real pools its simultaneous coverage is below Bonferroni's at small budgets.
4. Testing theory is asymptotic with fixed strata and contrasts and rests on high-level conditions (A5), (A4_K), (P2)/(P3); the K > H counterexample and the Romano-Wolf pin cite are not verified in the Supplement.
5. Two-phase (population-target) results are from a synthetic population only; the equalised-odds-gap ranking fails for the difference estimator in law school and is unexplained.
6. Not compared: sequential active inference; only fixed-budget one-shot designs.
7. Simulated referee report with more detail: `docs/referee_report_simulated.md`; audits: `docs/audit_manuscript_v2.md`, `docs/final_review_main.md`, `docs/final_review_si_code.md`.

## To verify against what was actually submitted
- Funding and competing-interest statements (left as bracketed placeholders in the draft files), department/ORCID, any AI-use statement required by the journal's policy, suggested reviewers.
- The class and bst files were submitted from the official template (not stored in the repository).

## If revisions are requested
Useful next steps, in order: repeat the main study over several splits for all settings; real-data check of the two-phase interval; a refit-aware variant; sequential/active comparison; tighten the weakest assumptions (A5), (A4_K), (P3).
