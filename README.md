# Trust confidence sets

Manuscript and code for a submission to the Special Issue on *Statistics for Data Science and AI* (International Journal of Data Science and Analytics).

**Idea.** Compare trained models on several trustworthiness criteria (error, equalised-odds gap, calibration, robustness) and return a bootstrap *confidence set* of models that are not significantly dominated, instead of a point-estimate winner or Pareto front.

## Layout
- `tcs/` method (`trust_confidence_set`, `pareto_front`)
- `experiments/` `simulate.py`, `casestudy.py`, `make_figures.py`, `make_tables.py`
- `results/` JSON outputs of the experiments
- `figures/`, `manuscript/` (LaTeX source, PDF)
- `docs/` brainstorm and literature review, novelty sweep, and notes on the four extensions (step-down and margin, refit bootstrap, rank-graduation criteria)

## Reproduce
```bash
pip install numpy scipy scikit-learn pandas matplotlib
python experiments/simulate.py      # about 2 min on 4 cores
python experiments/casestudy.py     # downloads Adult and credit-g from OpenML
python experiments/simulate_box.py && python experiments/simulate_box.py 0.0   # comparison with a marginal-box procedure
python experiments/make_figures.py && python experiments/make_tables.py
cd manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex
```
