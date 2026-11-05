# Referee-style check of manuscript_v2/sections/testing.tex and si_testing.tex

Scope: mathematical statistics only. Line numbers refer to testing.tex (T) and si_testing.tex (S). Scripts and outputs are in the session scratchpad
(`fp.py`, `power_mc.py`, `lform.py`, `twophase.py`, `design*.py`, `fwer.py`). Nothing in the repository other than this file was changed.

## Verdicts

| Result | Verdict |
|---|---|
| Thm 1(a)-(c) Size (pool target) | correct as stated |
| Thm 1(d) Size (population target) | correct with missing hypotheses (P2 is not well defined as written; see 1d) |
| Thm 2 Power curve | correct as stated |
| Cor 1 Labels needed / ARE | correct with a missing hypothesis (feasibility lambda_h L_d <= N_h) plus several overclaims |
| Thm 3 Joint asymptotics | correct as stated, given (A4_K) and (A5), which carry most of the weight |
| Thm 4 FWER, single-step and step-down max-abs-t, Bonferroni, Holm, bands | correct as stated (pointwise, not uniform; point nulls only) |
| Thm 5(a) single-contrast design | correct as stated |
| Thm 5(b) maximin, dual, saddle, support <= H | correct as stated (Danskin gradient needs A_h > 0) |
| Thm 5(c) default allocation | incorrect as stated for PATTERN-DIF; correct for the same S^2 in n(omega) and in V_k; the "iff" fails for K > H |
| Thm 5(d) rounding | correct as stated |
| Prop EO(a) | correct as stated |
| Prop EO(b) | incorrect for kappa < 0; regime of bounded |a|/r not covered |
| Prop EO(c) | correct as stated |

## 1. Size (Thm 1, S lines 37-45, 125-152)

Claim. Under (A1)-(A4) (and (A5)) the studentised statistic is asymptotically N(0,1) at the true value; two-sided, one-sided, non-inferiority and TOST tests have limsup size <= alpha. For the population target the sum of design and pool variance gives a valid statistic, and the pool-conditional statistic has size 2{1-Phi(z/sqrt(1+varpi))}.

Check of (a)-(c). The uniform Slutsky lemma (S 18-35) is right: the Polya step, the sandwich on Omega_eta, the Lipschitz constant phi(0), and the left limits are all handled. The TOST algebra is right: Z(-delta) = (X+a)/rho and Z(delta) = (X-b)/rho give the event z rho - a < X < b - z rho. The sup over theta_0 is legitimate because X_nu and rho_nu do not depend on theta_0, and the error terms do not depend on kappa. Small gaps:
- The lemma needs z_- > 0, that is alpha < 1/2 for one-sided and TOST (and for the Holm levels alpha/(2m_0)); say so. The TOST needs 1-2alpha > 0.
- "any constants theta_0, delta": for a sequence of pools theta_k = theta_0 forces theta_k to be constant; state it with theta_{0,nu}.
- Pointwise along sequences; the uniform version is only in S Cor "uniformity" and is not mentioned in T.

Numerics (power_mc.py: N = 20000, L = 1500, six strata, difference estimator with the mean-probability proxy, 6000 replicates). Rejection probabilities at kappa = 0, 0.5, 1, 1.5, 2, 3, -1, -2.5: 0.049, 0.073, 0.171, 0.325, 0.512, 0.851, 0.169, 0.701 against Psi: 0.050, 0.079, 0.170, 0.323, 0.516, 0.851, 0.170, 0.705 (MC s.e. about 0.003-0.006). One-sided and TOST boundary sizes also agree (0.0507 against 0.0500).

Check of (d). The decomposition D + P, the use of Levy continuity under the uniform sup, the bound on tV_D^{1/2}/s, the characteristic-function product, and the mediant argument for rho are all correct. Gaps:
- (P2) says "pool events E_N on which (A1)-(A4) hold". (A1)-(A4) are limit statements about sequences (min n_h -> infinity, a Lindeberg dichotomy, tau -> tau_inf, a constant c); they are not properties of a single pool. Step 1 needs the quantitative class in S Cor "uniformity": every sequence u_N in E_N satisfies (A1)-(A4) with common constants. This should be written as a hypothesis (for example Lindeberg ratio <= eps_N, min n_h >= m_N, V >= c r^2 with fixed c). It is a strong, unverified assumption, and (P3) and the continuity of the allocation rule are also unverified (T 153 admits this).
- The population version of the power theorem and of the label formula (Cor, T 82) is used but is not part of Thm 1(d), which states only (a)-(c). Step 5 of the proof does cover it; add it to the statement.
- The text calls varpi = V_pool/V_design in the statement but v_P/v_D (constants) in the proof; they agree only through (P3).

Numerics (twophase.py, infinite population of 3e6, pools N = 4000 and 10000, 2500 replicates). Pool-conditional size against the formula: 0.081 against 0.083 (varpi = 0.28), 0.194 against 0.185 (varpi = 1.19), 0.150 against 0.150 (varpi = 0.85). Size of the two-phase statistic with the estimated V_pool: 0.050 to 0.052. The values 0.07 at varpi = 0.2 and 0.17 at varpi = 1 (T 53) are correct.

## 2. Power and Corollary (Thm 2, Cor 1)

Thm 2 is a direct consequence of the lemma. The TOST power formula is correct (a = (theta+delta)/V^{1/2}, b = (delta-theta)/V^{1/2}).

Cor 1. V^{(d)}(L) = v/L - w is exact for the linearised variance with n_h = lambda_h L, and L_d = kappa^2 v/(Delta^2 + kappa^2 w) is the exact root. Verified in lform.py: with v, w from the six strata, the formula gives power 0.800 (target 0.80) and 0.900 (target 0.90) when V is recomputed at the rounded L. The population version L = kappa^2 v/{Delta^2 + kappa^2 (w - v_P)} is correct, and V_tot > v_P at every L.
Issues:
1. Missing hypothesis. The formula is valid only while lambda_h L_d <= N_h in every stratum (T 71 says so, but Cor 1 does not). Beyond it a capped stratum contributes 0, not a negative term, so the true variance is larger than the formula and the formula understates the labels needed. Also ell_h bounds and integrality. L_d <= v/w always, but v/w >= min_h N_h/lambda_h can exceed the cap.
2. The limit L_2/L_1 -> v_2/v_1 needs the sampling fractions at L_d to tend to zero (S 56 shows this); it is not "L/N -> 0" for arbitrary L.
3. S 56: "Psi(kappa) <= Phi(kappa-z) + Phi(-2z) for kappa >= z, hence the root satisfies kappa <= z + z_{1-beta}" cites the wrong direction. The upper bound on the root uses Psi >= Phi(kappa-z); the displayed inequality gives the lower bound kappa* >= z + Phi^{-1}(1-beta-Phi(-2z)) (valid when 1-beta > Psi(z)). The numbers are fine (root and z + z_{1-beta} agree to 6 digits at alpha = 0.05, beta = 0.2).
4. "Pitman ARE" is used loosely. There is no local sequence of alternatives or parametric model; the statement is the ratio of labels at equal power for a fixed effect, which tends to v_2/v_1. State it that way.
5. T 71: "S_h^2 are stratum variances of the residual influence values. SRS is the one-stratum design" together with "the efficiency of the paper, the product of the stratification gain and G" (T 82): the paper's SRS comparator uses the plain estimator (design.tex line 53), whose S^2 is that of z, not e. v_d must be defined with the estimator-specific influence value. The product G x stratification gain is an expectation under a calibrated proxy (Prop dif-deff), not a pool identity.
6. "Labels needed shrink by exactly that factor" (T 82) is exact only in the limit above; T 87 qualifies it, but T 82 should say "asymptotically".

## 3. Joint CLT (Thm 3, S 74-81)

Linearisation, bound V_k <= 2B'^2 r^2, Cramer-Wold with b_nu bounded, application of (A4_K) to y = b^T x/r, the Hajek condition via the sup bound, the subsequence principle with the compact set of correlation matrices, and the case q_infty = 0 (L^2 convergence) are all correct. Hajek's theorem is quoted correctly (N, n, N-n -> infinity and the Lindeberg-type condition with threshold eta {n(1-f)}^{1/2} S; necessary and sufficient). R-hat consistency is correct: |Sigma-hat_kl - Sigma_kl| <= r^2 o_p(1) and V_k >= c_k r^2. The floored matrix Sigma-hat + diag(V-hat^flat - V-hat) is PSD, so R-hat is a correlation matrix.
Remarks:
- (A4_K) is quantified over sequences of combinations with variance of exact order r^2, which makes the proof short but moves the content into the assumption. Stating the stratum-wise condition n_h(1-f_h) b^T Sigma_h^x b -> infinity (S 78) as the assumption would be checkable.
- (A5) is high level. The parenthetical sufficient condition (floor "adds o(V_k)" in strata of small share) is not automatic: the floor value pi(1-pi)d^2 is label-free and can be large where the actual S_e^2 is small. theory.tex line 217 and si_theory_extra.tex (floor remark) assert that the floor "does not affect Proposition validity(c)-(d)" without (A5); that is stronger than testing.tex and should be reconciled.
- Singular R (rank <= 24 for K = 108) is handled correctly; the rank count 3(M-1) = 24 is right (plug-in contrasts are exactly additive).

## 4. FWER (Thm 4, S 83-123)

Lemma crit: (i) continuity and strict monotonicity of F_R (support is range(R) != {0}), (ii) z_{1-alpha/2} <= c_alpha(R) <= z_{1-alpha/(2K)}, (iii) monotonicity in the index set, (iv) joint continuity via R^{1/2} continuity: all correct. Single-step proof: (max_{I0}|W|, c_alpha(R-hat)) converges jointly because c_alpha(R-hat) -> a constant; the inequality c_alpha(R_inf) >= c_alpha(R_{inf,I0}) is right, and so is the equality when I_0 is everything. Step-down: if j is the first step rejecting a true null then I_0 is in S_j, so the critical value is >= c_alpha(R-hat_{I0}); correct. Holm: the set form is equivalent to ordered p-values; |S_j| >= m_0 gives the threshold z_{1-alpha/(2m_0)}; correct. Part (c): the induction S^M_j in S^H_j is correct (also after max-t stops, because the active set is then constant and satisfies |Z_k| <= c). For every data set only requires R-hat to be a PSD unit-diagonal matrix, which it is.
Numerics (fwer.py): c_alpha(R) in [z, Bonferroni] for ranks 1, 2, 4, 12 with K = 12 (for example rank 4: 2.61-2.71 against 2.865), c(R_S) <= c(R) for all sampled subsets, step-down max-t FWER 0.041 and Holm 0.029 in a rank-4, K = 12 Gaussian configuration with 8 true nulls; Holm rejections were a subset of max-t rejections in every one of 5000 draws.
Caveats: strong control is for point nulls theta_k = theta_{k,0} (two-sided); composite one-sided nulls are not covered. The statement is pointwise for each sequence of pools, not uniform over sequences of R or over theta_{k,0}, although the proof (finitely many I_0, subsequences) would give uniformity over a class as in S Cor "uniformity". K = 108 is fixed, so finite-sample accuracy with a rank-24 matrix and small budgets is unverified. The citation of Romano and Wolf (2005, Econometrica) for "subset pivotality": that paper avoids Westfall-Young subset pivotality by a monotonicity argument; cite the specific lemma/theorem, and Westfall and Young (1993) or Romano and Wolf (2005, JASA) for subset pivotality. I could not verify the pin cite offline. Marcus, Peritz and Gabriel (1976) is in refs_testing.bib but is not cited anywhere; closed testing would give a shorter proof of the step-down claim.

## 5. Design (Thm 5, S 154-167)

(a) Right (Psi strictly increasing; one-sided power Phi(kappa-z) increasing).
(b) Sion applies (X = N compact convex, Y = simplex, F convex continuous in n because ell_h > 0, linear in w). The equality chain t* = g(w*) <= F(n, w*) <= max_w F(n, w) = t* shows every primal optimum minimises F(., w*) for every dual optimum; uniqueness follows from strict convexity when A_h(omega*) > 0 for all h; the KKT form n_h = min{N_h, max(ell_h, N_h A_h/lambda)} is right; the Caratheodory argument in R^H/R1 gives <= H atoms and the new weights remain a saddle. The unbounded formula for g is right. The gradient V_k(n(omega))/delta_k^2 (Danskin) needs the unique minimiser, that is A_h(omega) > 0; in the theorem statement it follows an "if" that does not visibly govern it.
Numerics (design2.py, design3.py; H = 5, K = 3, 4, 5, 8; balanced instances): dual optimum equals primal epigraph optimum to 1e-13 relative; the primal minimiser equals n(omega*) to 1e-4; support of w* is 2 or 3 (<= H); active relative variances equal to 5 digits.
(c) Problems:
1. The default allocation of the paper uses the plain-influence rule for both estimators (design.tex line 35; theory.tex line 255; si_theory_extra.tex alloc-dif: "the plain rule is what the design uses"). T 126 defines n^def = n(omega^def) with A_h(omega)^2 = sum_k omega_k S_{hk}^2 where S_{hk}^2 is the residual variance (T 116). For PATTERN-DIF these are different: n^def minimises sum_k V_k^{plain}/sigma_k^2 and not sum_k V_k^{DIF}/sigma_k^2; the claim "up to a factor it minimises sum_k V_k/V_k^SRS" and the maximin criterion refer to V_k^{DIF}. The statement is correct only if the same S_{hk}^2 (plain, or residual for the residual rule) is used in n(omega) and V_k. Si_theory_extra gives a loss factor up to 9/8 for nu_max = 3 between the two rules, so this is not negligible against a maximin comparison.
2. "Optimal iff relative variances are equal." Sufficiency is proved correctly (w_k proportional to delta_k^2 omega_k^def, scale invariance of n(omega)). Necessity needs K <= H, linearly independent (S_hk)_h, no stratum at a bound (and A_h > 0). T 126 states those conditions, but the paper's family has K = 108 > H, so the "iff" says nothing for the application, and for K > H it is false: design4.py, H = 2, K = 3, gives n^def with relative variances (1, 1, 0.01) that is maximin optimal (brute-force minimax 1.00008 against default max 1.0). For K <= H = 5, K = 3 independent: equal relative variances give max-rel 0.5773 = optimum; unequal ones give a loss ratio 1.17. The sentence in T 126 "optimal if equal, and also necessary when ..." is accurate; any headline version without the conditions is not.
3. The certificate max_k V_k(n^def)/delta_k^2 / g(w) is valid (weak duality). "the factor by which the labels needed exceed the optimum" holds only when finite-population corrections are negligible.
(d) Right; the rounding existence (floor or ceil with sum L) is fine.
Overclaim, T 133: "the maximin design maximises a lower bound on the power of each contrast, Phi(min nu_k - c_alpha(R))". c_alpha(R) depends on the allocation through R, so maximising min nu_k does not maximise that bound. It does maximise the design-free bound Phi(min nu_k - z_{1-alpha/(2K)}).

## 6. Equalised-odds gap (Prop EO, S 169-181)

(a) Proof correct (Omega_nu sandwich, signs preserved, |a-hat| > |b-hat|). It holds for |a| > |b| with (|a|-|b|)/r -> infinity; the |b| > |a| case has variance V_b, not V_a.
(b) The statement "(|a|-|b|)/r -> kappa in R" with limit max(s_a Z_a, s_b Z_b - kappa) is valid only for kappa >= 0 (a denotes the larger gap; S 175 assumes |a| >= |b| without loss of generality, which forces kappa >= 0). For kappa < 0, EO = |b| and (E-hat - EO)/r -> max(s_a Z_a + kappa, s_b Z_b); the stated formula is the limit of (E-hat - |a|)/r. Hence "its mean is at least max(0, -kappa)" is wrong for the true law when kappa < 0 (it is >= 0 only). Simulation (fwer.py): kappa = -1: empirical mean of (E-hat - EO)/r = 0.201, stated formula 1.201, corrected formula 0.201; kappa = +1: 0.200 matches. The kappa = 0 example (mean 1/sqrt(pi) = 0.5642, simulated 0.5644; density 2 phi(x) Phi(x)) is right. Regime not covered: |a|/r and |b|/r both bounded and not both zero; only the point a = b = 0 is given.
(c) Correct: union of four one-sided nulls, IUT, Thm 1(c) applied to the active one. Needs alpha < 1/2.
Overclaims: "the standard bootstrap is inconsistent for such directionally differentiable maps" (T 149) is stronger than Fang and Santos, who show it fails in general at the points of non-differentiability (consistency of the standard bootstrap is equivalent to full differentiability for their class) under a smooth-estimator limit; a design-based finite-population bootstrap is not covered. "A plug-in or pre-test choice cannot give uniformly valid inference" is an informal claim without a theorem.

## 7. Bibliography

Crossref checks: Berger 1982 (Technometrics 24(4):295-300), Berger and Hsu 1996 (Stat. Sci. 11(4):283-319), Schuirmann 1987 (JPB 15(6):657-680), Sion 1958 (Pacific J. Math. 8(1):171-176), Romano and Wolf 2005 JASA 100(469):94-108 and Econometrica 73(4):1237-1282, Hothorn et al. 2008 (Biom. J. 50(3):346-363), Marcus et al. 1976 (Biometrika 63(3):655-660), Genz and Bretz 2009 (LNS 195), Sarndal and Swensson 1987 (Int. Stat. Rev. 55(3):279-294) all match the .bib entries. Holm 1979 has no DOI in the .bib; the volume, issue and pages (6(2):65-70) are standard.
- fang2019: the entry gives year 2018, "advance access" and no volume, while the key and the in-text year say 2019. The journal version is Rev. Econ. Stud. 86(1):377-412 (2019); Crossref returns no volume or pages for the DOI, so I could not confirm the pagination from Crossref. Make key, year and volume consistent.
- hajek1960: not in Crossref; the .bib gives "number = {3}" for Publ. Math. Inst. Hungar. Acad. Sci. 5, pp. 361-374; the issue number is unverified and may be spurious (the series is cited as vol. 5, 1960, 361-374).
- vandervaart1998 Lemma 2.11 (Polya) and Ch. 14 (relative efficiency of tests) are the right pointers.
- marcus1976, lehmann2005, wellek2010 and boyd2004 are in refs_testing.bib but not cited in testing.tex or si_testing.tex.

## 8. Notation clashes with the rest of the paper

- K: number of contrasts (T 14) against K_i (number of models predicting 1, the stratum variable), and the compact set K in the proof of Prop validity, which S 8 re-uses in the same sentence.
- kappa: non-centrality (T 60), kappa_{alpha,beta}, the tie location in EO(b), and kappa_h = N_h/(N_h-1) in theory.tex.
- Delta: effect size (T 74), the simplex Delta_K (T 125), the gap Delta^{TPR}, and Delta_nu (S 173).
- nu: sequence index in SI, non-centrality nu_k (T 116), and nu_h (signal-to-noise) in theory.tex.
- rho: sqrt(V-hat/V) (S 12), disagreement rate rho_h (theory.tex), and rho = V_pool/V_design in the docs (the paper uses varpi).
- W: vector of statistics (S 105) against W_h = N_h/N; g: the dual g(w), the gradient g (S 127) and the group index; D: design error, D_i = y_mi - y_ji and D = diag(V^{1/2}); P: pool error, P_g, and probability; R: correlation matrix, R_h (stratum correlation), R_N (remainder).
- E-hat_m (EO estimate) against E_m (total error count, notation_v2.md).
- delta: margin, delta_g (group TPR difference), delta_k (design effect); a, b: TOST non-centralities, a_m, b_m gaps, a_k gradient, a_h = c_h sigma_h^2, a_{hk}, b = grad f(N tau), b_k; lambda: Neyman multiplier, shares lambda_h, difference coefficient lambda = 1; Phi: normal cdf against the objective Phi(n) of Prop allocation (and Phi_omega, T/S use both); phi: density against the contrast map phi_k; z_q quantiles against influence values z_i; m_h = {n_h(1-f_h)}^{1/2} against model index m; c_h: weight (theory.tex) with the same symbol as the stratum moment, and c, c_k, c_alpha used for constants and critical values.

## 9. Statements in testing.tex that exceed the proofs

1. T 82, 87: "labels needed shrink by exactly that factor" (asymptotic, small sampling fractions, G in expectation under a calibrated proxy).
2. T 126: n^def = n(omega^def) for PATTERN-DIF, and "minimises sum of inverse efficiencies" (holds only with the same S^2 in allocation and in V_k).
3. T 133: lower bound on power maximised by the maximin design (R depends on n).
4. T 142: EO(b) for kappa < 0 and the mean bound.
5. T 149: bootstrap inconsistency and the impossibility of uniformly valid plug-in or pre-test inference.
6. T 47 and T 30: (P2)'s "events on which (A1)-(A4) hold" is not a well-defined hypothesis; Thm 1(d) is conditional on it, on (P3) and on the allocation rule being continuous.
7. T 101-106: strong FWER control is stated for every configuration but is pointwise and for point nulls only; K = 108 fixed with rank 24.
8. T 27 against theory.tex line 217: the floor is stated to be harmless without (A5) in theory.tex, but (A5) is needed in testing.tex.
