# Final review: Supplementary Information, code, reproducibility

## 1. SI content

Verified against results/*.json (recomputed): Tables of the simulation section (real pools, synthetic, proxy, difference estimator real and synthetic), floor numbers, baseline prose numbers (coverage means, worst coverages, ratios, runtimes, per-type medians), ablation numbers (A1-A5, spot-checked against the scratch report). Dataset sizes in appendix_data.tex agree with pool sizes (0.4 N). No numerical discrepancy found. Note: per-type PATTERN medians in si_baselines (6.2, 4.8) include the Law school L=2000 cell with efficiency 1e26 in the median (35 cells); excluding it gives 6.0 and 4.6.

Proofs re-read line by line. Fixed in SI:
- appendix_proofs.tex (validity (b)): compact set K was a subset of [0,1]^C, but the difference estimator's tau-hat can leave [0,1]^C. Now [-1,2]^C.
- appendix_proofs.tex: the main-text claim "bias O(r^2)" had no proof. Added a bias paragraph (needs truncation of theta-hat to the estimand range on the rare event E^c; the code returns NaN there).
- appendix_proofs.tex / si_theory_extra.tex, Prop. proxy (e): the bound "1/eta times SRS" was proved only for Phi_0 (no finite-population correction). Added the exact bound (1-eta f)/(eta(1-f)), f=L/N.
- appendix_proofs.tex, allocation (b): added assumptions (A_h>0, l_h<=N_h, sum l_h<=L<=sum N_h). Documented that the code (tcs/labeling.py `_round_alloc`) gives each stratum its floor first and distributes the rest proportionally to N_h A_h, which is not exactly the KKT solution (differs by the floor shift); validity unaffected.
- appendix_proofs.tex: Cauchy-Schwarz line in proxy (b) written correctly (sqrt(omega x) sqrt(omega/x)).
- si_baselines.tex: stale sentence "we did not extend the theory to it" replaced (Prop. validity covers lambda=1; Prop. dif-lambda covers tuned weights).
- simulation.tex: clarified that the rho x efficiency range pools the three estimand types.

Cross-references: no undefined, duplicate or crossed labels (checked all \ref/\eqref/\Mref/\SIref, both directions); no `??` in either build. Every statement in theory.tex has a proof or reference in the SI.

Unresolved, in main-paper files (not editable here):
- theory.tex, around the influence proposition: c_h is defined as N_h^2(1-f_h)/n_h and again as N_h^{-1} sum Y_i D_i; pi_h (stratum rate) vs pi_i (proxy); e_h vs residual e_i; r_h vs r. Suggest renaming the stratum moments.
- theory.tex, proxy paragraph: "weighted variance at most 1/eta times SRS" should read "when L/N is small" (or use (1-eta f)/(eta(1-f))).
- theory.tex states E_pi S^2_e = sigma-bar^2 without the SI qualification (gradient at the model-expected totals T_pi).
- baselines.tex line 9: same stale sentence "we did not extend the theory ... to it".
- Duplicate bibliography keys cited in main (introduction.tex, related.tex): bennett2010/bennett2010online, kumar2018/kumar2018classifier, musgrove2023/musgrove2023discordant print twice in the main reference list. Replace the short keys by the long ones in those two files, then delete the short entries from references.bib.
- design.tex cites results/proxy_calibration.json, which no script produces.

## 2. Reproducibility

- All scripts and results files named in the SI text exist.
- Tests: tests/test_labeling_dif.py passes (4 s); tests/test_labeling_two_phase.py passes (median ratio 1.021). Removed an unused variable (`ctx0`) from the latter.
- README.md reproduce section rewritten to match each script's usage (arguments and R values read from the JSON files): preference_validation.py (the argument is the replicate count of an old trust-set study; it caches the pools as a side effect), label_efficiency(_dif).py, make_v2_assets.py, label_baselines.py, label_ablations(_dif).py (`report` mode), label_efficiency_splits.py (build/run/report), theory_check.py, theory_check_dif.py (incl. `floor`), floor_check.py, two_phase_check.py, tests, builds.
- No hard-coded absolute paths, no debug code, in tcs/ and experiments/ (the only "todo" hits are a variable name). Modules without docstring: tcs/__init__.py, experiments/make_figures.py, experiments/make_tables.py, experiments/label_ablations_dif_report.py has one but no main (called via label_ablations_dif.py report). Scripts need to run from the repository root.
- Referenced but not reproducible by a documented command: results/proxy_calibration.json. results/preference_validation.json and label_ablations.json are produced by documented commands but not cited in the paper.

## 3. LaTeX health

- SI had 17 overfull display equations (38 to 190 pt, two-column layout) and one proposition heading colliding with its first item. All rewritten with align/gather line breaks or a shorter title (si_theory_extra.tex, appendix_proofs.tex, si_baselines.tex). After the final build no SI box exceeds 8 pt. SI 38 pages, main 35 pages, 0 unresolved `??`, 0 log errors.
- Still overfull in the main paper (not editable): 16 boxes above 20 pt in manuscript_sn.log (21 to 162 pt), mostly display equations in the two-column layout (e.g. eq. for the stratum variances, allocation, design effect) and the efficiency table around manuscript_sn.tex lines 654-703 (127 pt). They need the same align/split treatment.
- references.bib: removed uncited duplicate `sawade2012`; removed the printed internal note "Verified: Crossref" from four entries (it appeared in both reference lists); removed three redundant long `url` fields (ji2020fairness, kossen2021active, okanovic2025; arXiv ids kept in note) that overflowed by 20 to 168 pt. Uncited, left in place: bickel1984, black2022multiplicity, krewski1981, wright2012, wu2001. refs_*.bib are not used by either build.
