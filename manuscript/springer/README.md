# Springer Nature version

`manuscript_sn.tex` is the single-file version for the journal (Springer Nature template, `iicol` two-column option, numeric citations). It is generated from `../manuscript.tex` by `python manuscript/build_springer.py`; edit the source and rebuild rather than editing this file.

The template class (`sn-jnl.cls`) and the reference style (`sn-mathphys-num.bst`) are not included. Download the current official Springer Nature LaTeX template package (see the journal's submission guidelines), copy those two files next to `manuscript_sn.tex` and compile with `pdflatex` twice. `manuscript_sn_draft.pdf` was compiled with a mirrored copy of the October 2023 class file, so check the layout again with the official files.

Journal rules respected: one `.tex` file, figures in the same folder, captions in the text, 4-6 keywords, abstract of 150-250 words, numbered citations.
Still to complete before submission: department, city and ORCID in the affiliation, and any disclosure the journal's policy requires.
