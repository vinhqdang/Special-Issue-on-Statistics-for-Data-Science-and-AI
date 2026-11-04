# Overfull fixes (main paper, two-column build)

All equations keep their number and `\label`; the mathematics is unchanged. Fix type "gathered" means the formula is wrapped in `\begin{gathered}...\end{gathered}` inside the existing `equation`, with one clause per line (breaks at commas, `=` or `+`).

| Equation / label | File | Fix |
|---|---|---|
| `eq:PQ`, `eq:TPFP` (P_g, Q_g, TP, FP, E_m) | setting.tex | align rows: TP, FP and E_m on separate rows (`\notag`, label kept on the E_m row) |
| Delta^TPR_m, Delta^FPR_m (unlabelled) | setting.tex | gathered, one gap per line |
| `eq:theta` | setting.tex | gathered, one estimand per line |
| `eq:DIF` | design.tex | gathered, estimator and stratum mean on separate lines |
| `eq:ht` | theory.tex | gathered, three lines |
| `eq:psi` | theory.tex | gathered, psi_i and xi_i on separate lines |
| `eq:dif` | theory.tex | gathered, break before the second `=` and before F_h |
| `eq:resid` | theory.tex | align rows split (e_i / tilde z_i / hat V_psi / hat e_i / V_psi), `\notag` except the last row (label kept) |
| `eq:infl-forms` | theory.tex | gathered, one influence value per line |
| `eq:stratvar-tpr`, `eq:stratvar-err` | theory.tex | align: break after `+` in the TPR variance, e_h on its own row |
| `eq:agree-var` | theory.tex | gathered, one variance per line |
| `eq:dif-deff` | theory.tex | gathered, ratio and G on separate lines |
| `eq:anticipated` | theory.tex | gathered, plain and difference rule on separate lines |
| `eq:pop-variance` | extra_population.tex | gathered, two lines |
| hat V_pool display (unlabelled) | extra_population.tex | gathered, two lines |
| Efficiency table (`\input{tables/efficiency_dif.tex}`, 126 pt too wide) | empirical.tex | `\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{...}`: scales down only if wider than the text width |
| `\texttt{results/label_efficiency_splits_summary.json}` (14 pt) | extra_splits.tex | `\allowbreak` after each escaped underscore |

Remaining overfull boxes in manuscript_sn.log: three bibliography URLs (7.7 to 16.7 pt, from the .bbl, not in the section files). The supplement has two display overfulls (7.3 pt and 6.5 pt) in files outside the task list.
