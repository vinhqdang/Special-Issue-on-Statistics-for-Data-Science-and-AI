# Labelling where models disagree

Code, experiments and manuscript for a study of label-efficient comparison of several trained classifiers on group-fairness criteria
(submission to the Special Issue on *Statistics for Data Science and AI*, International Journal of Data Science and Analytics).

**Idea.** Comparing classifiers on group-specific true and false positive rates needs many labels under simple random sampling. The proposed
design (PATTERN-DIF) stratifies an unlabelled pool by group and by the number of models predicting the positive class, allocates a labelling
budget with a proxy-based generalised Neyman rule, and estimates all pairwise contrasts with a stratified difference estimator and
linearisation confidence intervals.

## Layout
- `tcs/` method: `labeling.py` (design, plain estimator), `labeling_dif.py` (difference estimator, the proposed method), `labeling_baselines.py`,
  `labeling_variants*.py` (ablation variants), plus earlier material on model confidence sets under several trust criteria (`core.py`, `certify.py`, `preference.py`)
- `experiments/` studies (`label_efficiency_dif.py` main study, `label_baselines.py`, `label_ablations_dif.py`, `theory_check_dif.py`, ...)
- `tests/` unit tests for the difference estimator and the two-phase intervals
- `results/` JSON outputs; `results/pool_cache/` cached model predictions (not tracked)
- `manuscript_v2/` full-length manuscript (LaTeX, PDF, `springer/` for the journal template build); `manuscript/` earlier manuscript on trust confidence sets
- `docs/` notation, literature sweeps and notes

## Reproduce
All commands run from the repository root (scripts use relative paths and `sys.path.insert(0, ".")`). Requirements:
`pip install numpy scipy scikit-learn pandas matplotlib` (data sets are fetched from OpenML on first use).

```bash
# 0. Pools: trains the nine models per setting and caches predictions in results/pool_cache/ (12 settings, 'nomao' is not used in the paper).
#    The argument is the number of replications of an earlier trust-set validation that this script also runs; 1 is enough. It also writes results/preference_validation.json.
python experiments/preference_validation.py 1
# 1. Main study (35 setting-budget cells, R=200); `report` writes tables/figures and results/label_efficiency_dif_summary.json
python experiments/label_efficiency_dif.py run 200 && python experiments/label_efficiency_dif.py report
python experiments/label_efficiency.py 200            # plain-estimator study -> results/label_efficiency.json (used by theory_check.py)
python experiments/make_v2_assets.py                  # tables/figures and label_efficiency_summary.json from it
# 2. Baselines (R=200) -> results/label_baselines.json, manuscript_v2/tables/baselines_*.tex, figures/base_efficiency
python experiments/label_baselines.py run 200 && python experiments/label_baselines.py report
# 3. Ablations (R=150, B=999) -> results/label_ablations_dif.json, tables abl_dif_*.tex, figures abl_dif_*
python experiments/label_ablations_dif.py 150 && python experiments/label_ablations_dif.py report
python experiments/label_ablations.py 150             # earlier plug-in ablation -> results/label_ablations.json (optional, for comparison)
# 4. Repeated train/pool splits (R=100, seeds 1-3) -> results/label_efficiency_splits*.json
python experiments/label_efficiency_splits.py build 1,2,3 && python experiments/label_efficiency_splits.py run 100 1,2,3 && python experiments/label_efficiency_splits.py report
# 5. Theory checks (SI Section "Checking the theory by simulation")
python experiments/theory_check.py 200 1000           # -> results/theory_check.json, figures fig_sim_real, fig_sim_rho
python experiments/theory_check_dif.py 2000 2000      # -> results/theory_check_dif.json, figures fig_sim_dif_*
python experiments/theory_check_dif.py floor 1000     # part D (variance floor), merged into the same json
python experiments/floor_check.py 200                 # independent floor check -> results/floor_check.json
# 6. Population (two-phase) coverage, R=300 -> results/two_phase_check.json (+ results/two_phase_truth.npz cache)
python experiments/two_phase_check.py run 300 && python experiments/two_phase_check.py report
# 7. Tests
python tests/test_labeling_dif.py
python tests/test_labeling_two_phase.py
# 8. Manuscript (full-length version) and journal build (main paper + Supplementary Information)
cd manuscript_v2 && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && python manuscript_v2/build_springer_v2.py /path/to/sn-article-template   # needs the Springer Nature template (sn-jnl.cls, bst/)
```
Not reproducible by a documented command: `results/proxy_calibration.json` (calibration and AUC of the proxy quoted in the design section; no script in the repository writes it).
Cached pools (`results/pool_cache*/`) are not tracked; step 0 and `label_efficiency_splits.py build` recreate them.
