# Brainstorm and literature review

Target: Special Issue on *Statistics for Data Science and AI*, International Journal of Data Science and Analytics (Springer). The call asks for statistical methods that make ML models accurate, explainable, robust and fair ("trustworthy"), and for approaches that *measure* trustworthiness in real applications.

## Candidate ideas considered

| # | Idea | Why it fits | Main risk |
|---|------|-------------|-----------|
| 1 | Conformal prediction under imputed covariates with a trust report | Missing data, coverage guarantees | Already covered by Zaffran et al. (ICML 2023) |
| 2 | Confidence intervals for composite trust scores (SAFE / rank-graduation metrics) | Fits journal's existing line of work | Needs deep knowledge of those metrics' estimators |
| 3 | **Confidence sets for model selection under a vector of trust criteria** (chosen) | Statistical inference layer for trustworthiness; cross-field (econometrics MCS + ML fairness/robustness) | Conservativeness; conditional on trained models |
| 4 | Fairness-metric hypothesis tests for intersectional groups | Timely | Ferrara et al. (NeurIPS 2025) cover this closely |

Idea 3 was chosen because the gap is clear: single-metric inference exists (model confidence set, fairness CIs), and multi-criteria Pareto selection exists (racing, evolutionary), but we found no bootstrap confidence set for Pareto-nondominated models under trust criteria.

## Literature map (all entries checked against Crossref, arXiv or publisher pages)

- **Confidence sets / multiple comparison.** Hansen, Lunde & Nason (2011, Econometrica 79:453) model confidence set; White (2000, Econometrica 68:1097) reality check; Hansen (2005, JBES 23:365) SPA test; Romano & Wolf (2005, Econometrica 73:1237) stepwise multiple testing.
- **Comparing learners.** Dietterich (1998); Demšar (2006, JMLR 7:1); Benavoli et al. (2017, JMLR 18(77)); Bouthillier et al. (2021) variance in benchmarks.
- **Pareto model selection.** Zhang, Georgiopoulos & Anagnostopoulos (2018, IEEE T-Cyb 48:596) SPRINT-Race: fixed-confidence racing, sequential, search-oriented.
- **Uncertainty of fairness measures.** Miller et al. (2021, PMLR 149) subpopulation metrics; Ferrara et al. (NeurIPS 2025, arXiv 2506.10586) size-adaptive tests for statistical parity.
- **Multiplicity.** D'Amour et al. (2022, JMLR 23(226)); Marx, Calmon & Ustun (ICML 2020).
- **Trust metrics.** Hardt, Price & Srebro (2016) equalised odds; Guo et al. (2017) calibration; Giudici, Raffinetti & Riani (2024, IJDSA 20:1043); Babaei, Giudici & Raffinetti (2025, ESWA 259:125239); Wu (2026, IJDSA 22); Nikitina & Mouriño (2026, IJDSA 22:362).

## Gap and proposal

Trust assessment compares models on several measured criteria, but the selection (composite winner, Pareto front of estimates) is reported without sampling uncertainty. Proposal: the **trust confidence set** (TCS), a bootstrap, studentised, intersection-union confidence set for the population-nondominated models, with asymptotic family-wise retention control (manuscript Proposition 1), plus descriptive Pareto-membership probabilities.

## Findings that shaped the paper

1. A first version used a minimum over criteria in the critical value; it under-retained non-dominated models (85-94% at rho=0). Switching to a maximum (least-favourable intersection-union configuration) restored retention at the cost of power.
2. The simulation's first design was too easy (naive Pareto front also retained everything). Adding a model *tied* with a rival on one criterion exposed the naive rule's failure.
3. On real benchmarks the honest result is that the TCS removes no model when four criteria are required; the composite-score winner is unstable.

## Open items before submission

- Retraining/seed variance is not covered (conditional inference); a refit bootstrap is the main extension.
- Simulation uses mean-type criteria; ECE and the equalised-odds gap are non-smooth and not simulated.
- Re-run the literature search close to submission; the novelty statement is limited to what was reviewed.
- Reformat to the journal template (Springer Nature `sn-jnl`) and complete the author/affiliation block.
