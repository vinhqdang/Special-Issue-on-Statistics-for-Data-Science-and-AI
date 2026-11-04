# Final review of the main paper (after the split from the supplement)

Scope: manuscript_v2/sections/{abstract, introduction, related, setting, design, theory, numcheck, empirical, extra_splits, extra_population, baselines, ablations, guidance, discussion}.tex and the tables they input. Supplementary files were read for cross-checking but not edited. Scratch scripts (numbers, labels, citations) are in the session scratchpad (c1-c4.py, b1.py, abl_report.py, keys.txt). Build: `python manuscript_v2/build_springer_v2.py /tmp/sn_official/x/sn-article-template` run once after the edits: main paper 35 pages, supplement 38 pages, 0 unresolved '??', 0 log errors, 0 undefined citations or references; main abstract about 248 words.

## 1. Numbers recomputed and found correct

All of the following were recomputed from results/*.json and match the text and tables (rounding as printed):

- Main study (results/label_efficiency_dif.json, 35 cells, R=200): medians over settings of cell medians for PATTERN-DIF, PATTERN, SRS-DIF, GROUP, GROUP-DIF, StratPPI, DIF-resid at the four budgets, with and without law school; design effect 3.3/3.3/3.7/8.6 (4.9 without law), estimator effect 1.65/1.54/1.51/1.31; by-estimand medians; top-3-pair medians; the seven-setting subset (5.3/5.3/6.0 and 3.5/3.8/4.4); median within-setting increase 23%; 34 of 35 cells; StratPPI above PATTERN in 18 of 35 cells and below PATTERN-DIF in all 35 (paired); StratPPI median efficiency above PATTERN-DIF only in Bank at L=2000; lowest efficiency outside law school 3.55; range 3.5 to 12.1 at L=250; law-school efficiencies 274, 510, 11241; Churn 60.8; Adult-race at L=1000 (23.9, 20.6, 3.7); 51 of 108 degenerate estimands (law school only); SRS undefined-estimate rate 0.5%, stratified designs 0.
- Tables efficiency_dif (328 numbers), decomposition_dif, inference_dif, data.tex (N, P0, P1, disagreement recomputed from results/pool_cache). Coverage: means, worst values, floor and 'max' floor values, medians across settings (0.943, 0.949), worst 0.59 (law school) and 0.825 minimum elsewhere at L=250.
- Proxy: AUC 0.76 to 1.00; 10-bin ECE 0.02 to 0.07 in ten settings and 0.195 in EEG (results/proxy_calibration.json).
- Earlier study quoted in the introduction and setting: 36% (range 12 to 96%) and 1,104 vs 135,354 labels over twelve settings (realdata_validation/realdata_multi, certify_realdata); 712/19,537 = 3.6%.
- Splits (label_efficiency_splits.json plus split 0 from the main study): all 12 rows of the table incl. SD and range, CV 0.068, max/min ratio 1.18, medians per split, pooled 6.7/7.3, original split lowest at both budgets, coverage 0.948 to 0.951 and 0.939 to 0.941, worst 0.84 to 0.88, ranking of settings in every split.
- Two-phase study (two_phase_check.json): every table entry, coverage ranges, widths (2 to 10%, 12%, 21 to 71%), MSE ratios 1.04 to 2.6, predicted coverage 0.93/0.83/0.67, arithmetic (6+1.5)/(1+1.5)=3.
- Baselines (label_baselines.json, R=200, 35 cells): Table base-budget, Table base-setting, 19 of 35, seven of ten, IS 1.12L, IS top-3 medians and 10 of 35, coverage means and worsts, undefined-estimate rate at most 1%.
- Ablations (label_ablations_dif.json via the report script run into the scratchpad): Tables abl-design, abl-inference, prose for A1 to A5, DIF over HT 1.67 in all 29 cells, shuffled/noisy proxy DIF/HT 0.96, M-bins 17.2/12.5/4.8/3.8, Spearman, 'zero' and 'max' floors, five cells below 0.85.
- Numerical check of theory (theory_check_dif.json): 0.98 over 34 cells, gain median 1.47.

## 2. Errors and gaps found and fixed (all in the main-paper files)

Numbers and sources
1. empirical.tex (group-only stratification): "median 1.0 at every budget" is false at L=2000 (1.25; 1.0 only without law school). Fixed with the actual values.
2. introduction.tex: the Spearman correlation -0.68 includes law school; the empirical section gives -0.58 without it. Both now stated in both places (also -0.66 and -0.81 with law school at L=500, 1000).
3. extra_splits.tex: "at least 4.1" in all 48 values (minimum is 4.19, table shows 4.2) and "about 4.5 (Electricity) to about 15" at L=1000 (split means are 4.9 and 15.4). Fixed.
4. baselines.tex: "medians differ by a few per cent between the two studies" is wrong: per-cell ratios of PATTERN-DIF/PATTERN efficiencies between the main study and the baselines study range 0.73 to 1.20 (median deviation 5%), and medians differ by up to 14% (PATTERN-DIF at L=250: 6.6 against 5.8). Now stated with numbers and with a pointer to which study the abstract uses.
5. baselines.tex: factor "1.3--1.6" for the model-assisted extensions contradicted "1.3--1.7" a paragraph later (PATTERN-PPI gives 1.68). Unified to 1.3--1.7.
6. baselines.tex: "the two settings with the lowest disagreement rate (Law school, Churn)" is wrong by every measure used in the paper (credit card has the second lowest top-3 disagreement rate, Bank has the smallest disagreement set). Reworded to "the two settings where PATTERN gains most (Law school, Churn; disagreement sets of 19% of the pool)".
7. guidance.tex: "variance decreases in proportion to the budget" is wrong; now "roughly inversely proportional (ignoring the finite-population correction)".
8. ablations.tex: "the ensemble mean was never beaten there" was stronger than the table (it is the median that is never beaten); reworded.
9. design.tex: "well under a second on a laptop" was unsourced; now the measured maxima (0.6 s design, 0.05 s per replicate, results/label_baselines.json) and "our implementation". "about 2%" for tuned weights now "2 to 3%" (ablations 1.6%, baselines 2.7%) with both sections cited.
10. theory.tex: "two to three orders of magnitude" for agreement-stratum variance was unsourced; checked on the ten non-degenerate real pools (median ratio per pool 2e-5 to 1e-2); reworded to "typically at least two orders" and restricted to the pairs among the three most accurate models and to the seven settings in which agreement strata hold most of the pool.

Claim discipline
11. 'Calibrated' proxy: design.tex and discussion.tex called the ensemble proxy "reasonably calibrated"; ECE is 0.02 to 0.07 (0.20 in EEG) and the worst probability quintile is off by 0.04 to 0.20. Now "only moderately miscalibrated", with those figures.
12. Abstract: added that six of the eleven settings use arbitrary groups, that the worst contrast covered 0.59 at 250 labels, and that the two-phase result is from a synthetic-population experiment (abstract kept at about 248 words by shortening other sentences).
13. introduction.tex limitations: single split was stated as universal although six settings were repeated on four splits; rewritten, plus new caveats that the variance floor is in-sample and the two-phase result is synthetic. discussion.tex limitations updated in the same way; the source study (main study versus ablation study) is named for each floor number.
14. empirical.tex: floor result flagged as in-sample; "reliable for every estimand from about 1,000 labels" now says that 0.89 to 0.90 is about what chance gives over 108 estimands.
15. ablations.tex: the "independent check on synthetic pools" of the floor (results/floor_check.json) never exercised the floor (zero_se_share = 0 in all 16 rows), so it shows only that widths do not change, not that coverage is repaired. Stated.
16. ablations.tex: the equalised-odds ranking paragraph omitted that in all three law-school cells the difference estimator is about 10^-4 times as efficient as SRS (best model identified in 0.45 to 0.59 of replicates against 1.00), whereas the plain estimator on the same design is 38 to 208 times better. Added, and the recommendation on gap-targeted allocation restricted accordingly.
17. extra_population.tex: states in the first sentence of the experiment that it is synthetic; the sample is called PATTERN-DIF (it used the difference estimator) instead of "a PATTERN sample".
18. discussion.tex: "its benefit can be forecast from the disagreement rate" overclaimed (rank correlation about -0.6); guidance.tex "predict" now "indicate" with the correlations. "Extends directly" to more than two groups and "would increase the gain" with a better proxy changed to "should extend" and "could increase ... (untested)".
19. introduction.tex: Pepe and Alonzo / Schatzkin / Chock cover sensitivity and specificity only as ratios when the doubly negative cell is unverified; the sentence said "directly". Fixed.
20. related.tex: Kuboyama (2026) "settles the accuracy ranking ... exactly" overstated an empirical finding; now "settled the choice of the most accurate model in all 108 experimental conditions" (docs/label_efficient_lit.md 1.4).

Split hygiene and notation
21. ablations.tex: tables and captions use "A1", "A4", "A5", "plug-in", but these were defined only in the SI; now defined in the first paragraph. Added the missing pointer to Supplementary Table for the weight lambda.
22. empirical.tex: "drawn exactly as in an earlier version of this study" had no referent; now names results/label_efficiency.json and notes the law-school difference. setting.tex and introduction.tex: the "earlier study of ours" (twelve settings, Nomao included) is now identified as outside the paper with its result files; the forward reference to Table 1 (in Section 6) is stated.
23. Notation: extra_population.tex used rho for the variance ratio and G for the gradient, clashing with the pair disagreement rate rho and the gain G of the theory; now varpi and bold b. theory.tex notes that pi_h (stratum prevalence) is not the proxy pi_i and that the moment c_h is not the weight c_h.
24. Roadmap in the introduction checked against the sections (related, setting, design, theory with the numerical check, empirical, splits, population, baselines, ablations, guidance, discussion): consistent. All \ref/\eqref/\SIref targets resolve, SIref targets exist only in the SI, no label occurs on both sides; no "Appendix" or "Table below" left in the main files.

Citations
25. All 79 distinct keys cited in the main files exist in references.bib. 36 characterisations were compared with docs/label_efficient_lit.md (Sawade 2012, Musgrove, Kuboyama, Kher, Hartmann, Yan, Ji 2020, Ji 2021, Gao, Gronsbell, Shanmugam, Chugg, Cherian, Das, Fisch, Zrnic, Li, Skalnes, Chatzi, Hara, Kay, Karimi, Okanovic, Ashury-Tahan, Dorner, Bennett, Kumar, Katariya, Marsigli, Liu, Ashouritaklimi, Berrada, Yilmaz, Marx, Pepe, McNamee); only items 19 and 20 needed changes.

## 3. Unresolved or for the authors (file:line)

- introduction.tex:5 and setting.tex:31: the twelve-setting earlier study is not published or cited; the numbers (36%, 1,104, 135,354) rest on repository result files only. Cite it or move it to the SI.
- ablations.tex:33: the failure of the difference estimator for equalised-odds ranking in law school is reported but not explained (the plain estimator on the same design is fine).
- baselines.tex:3 versus empirical.tex:16: two sets of efficiencies (different seeds) for PATTERN-DIF and PATTERN are in the paper (6.6/6.2/6.6 against 5.8/6.3/6.9); a rerun with shared seeds would remove the issue.
- extra_population.tex: the full OpenML pipeline against full-data references was not run; only the synthetic population is covered. extra_splits.tex: three new splits only, learner seeds fixed, six largest settings only.
- theory.tex:125 and appendix_proofs.tex:123-127: the symbols pi_h and c_h are overloaded in the SI proofs as well; renaming needs a change in the SI (not edited).
- SI items noticed but not edited: simulation.tex:356 quotes the floor check ("changes nothing") without saying that the floor never triggered there; si_ablations.tex:13 and :19 and the SI summary refer to "the earlier study" without definition; simulation.tex:297 and the law-school note in the earlier audit (docs/audit_manuscript_v2.md) about the different estimand mask still apply.
- Build: several display equations are 40 to 170 pt too wide in the two-column Springer layout (about ten overfull hbox entries in the build's manuscript_sn.log, e.g. at log lines 256, 290, 364, 383, 452); they pre-date this review and need `split`/`multline` before submission. The abstract is at about 248 words, at the 250-word limit.
- Bibliography: duplicate entries for the same works (bennett2010 and bennett2010online, kumar2018 and kumar2018classifier, musgrove2023 and musgrove2023discordant) are used under different keys in different sections.
- Stale compiled files manuscript_v2/main.pdf, main.aux are not rebuilt by this review.
