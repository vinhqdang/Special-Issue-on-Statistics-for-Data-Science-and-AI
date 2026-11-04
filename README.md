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
- `tests/` unit tests for the difference estimator
- `results/` JSON outputs; `results/pool_cache/` cached model predictions (not tracked)
- `manuscript_v2/` full-length manuscript (LaTeX, PDF, `springer/` for the journal template build); `manuscript/` earlier manuscript on trust confidence sets
- `docs/` notation, literature sweeps and notes

## Reproduce (main study)
```bash
pip install numpy scipy scikit-learn pandas matplotlib
python experiments/preference_validation.py 1      # trains the nine models per setting and caches the pools
python experiments/label_efficiency_dif.py run 200 # main study (35 setting-budget cells)
python experiments/label_efficiency_dif.py report  # tables and figures
python tests/test_labeling_dif.py
cd manuscript_v2 && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
