# Style pass on the main paper

Scope: manuscript_v2/sections/{abstract, introduction, related, setting, design, theory, numcheck, empirical, extra_splits, extra_population, baselines, ablations, guidance, discussion}.tex. No other file was edited.

## Kinds of change

- Removed staged phrasing and emphasis contrasts ("not X but Y", "rather than ... , not ...") where a plain statement suffices; where the contrast carries content it was kept.
- Removed or merged restating sentences and inflated wording ("substantial", "the honest conclusion", "serious", "what matters is").
- Results are now in the present tense (empirical, splits, ablations, guidance, discussion); procedure descriptions keep the past tense.
- Shortened long sentences, split a few, and removed repeated sentence frames (for example "It is ... which ...").
- Removed duplicated explanations in the introduction (limitations of the main study are stated once), in related work (the debt to diagnostic-test designs, the model-assisted point) and in the discussion (proxy and disagreement statements point to the sections that hold them).
- British spelling and the terms PATTERN, PATTERN-DIF, SRS, GROUP, proxy, contrast, estimand are unchanged.

## Verification

For every file the multiset of numeric tokens, math fragments, equation environments and \cite/\ref/\label/\SIref/\Mref/\eqref/\input arguments is identical before and after (script in the session scratchpad, tok.py). No caveat added by the final review was removed. Build run once: main paper 35 pages, supplement 38 pages, 0 unresolved '??', 0 log errors; abstract about 248 words.

## Words per file (before / after, whitespace-delimited, including LaTeX markup)

| file | before | after |
|---|---|---|
| abstract | 263 | 263 |
| introduction | 2220 | 2090 |
| related | 2538 | 2477 |
| setting | 845 | 811 |
| design | 1214 | 1201 |
| theory | 2279 | 2267 |
| numcheck | 288 | 272 |
| empirical | 1697 | 1657 |
| extra_splits | 661 | 647 |
| extra_population | 1012 | 983 |
| baselines | 1179 | 1159 |
| ablations | 1634 | 1635 |
| guidance | 585 | 566 |
| discussion | 935 | 901 |
| total | 17350 | 16929 |

The total reduction is about 2.4%, below the 10 to 15% target. Because every number, label and citation had to remain, and the audited caveats had to stay, further cuts would have meant removing sentences whose meaning could not be confirmed unchanged. The numerical results in empirical, baselines and ablations are dense and mostly irreducible.
