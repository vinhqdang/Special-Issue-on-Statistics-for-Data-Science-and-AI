# Springer Nature version of the full-length manuscript

`manuscript_sn.tex` is generated from `../main.tex` and the files in `../sections` and `../tables` by
`python manuscript_v2/build_springer_v2.py` (run from the repository root). Edit the sources and rebuild rather than editing this file.

Single file, flat folder (figures in the same directory), two-column `iicol` option, numeric citations, tables and sections inlined.
The template class (`sn-jnl.cls`) and the reference style (`sn-mathphys-num.bst`) are not included: copy the official Springer Nature
files next to `manuscript_sn.tex` and compile with `pdflatex`, `bibtex`, `pdflatex` twice. Check the layout again with the official files.

Still to complete before submission: department, city and ORCID in the affiliation, and any statement the journal's policies require.
