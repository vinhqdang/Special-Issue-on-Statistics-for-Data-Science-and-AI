# Springer Nature version: main paper and Supplementary Information

`manuscript_sn.tex` (main paper) and `supplement_sn.tex` (Supplementary Information) are generated from `../main.tex`-style sources in
`../sections` and `../tables` by

```bash
python manuscript_v2/build_springer_v2.py /path/to/sn-article-template   # directory holding sn-jnl.cls and bst/
```

Edit the sources and rebuild rather than editing the generated files. The script compiles both documents with the official Springer Nature
template (December 2024 version), resolves references between the two documents (main text uses `\SIref`, the Supplement uses `\Mref`; the
numbers are read from the `.aux` files and written into the preamble, so each file is a single self-contained `.tex`), and copies the PDFs back
as `manuscript_sn_draft.pdf` and `supplement_sn_draft.pdf`.

Each file is single, in a flat folder (figures next to the `.tex`), two-column `iicol`, numeric citations. The template class and the `.bst` file
are not included here; submit them with the source as the journal asks. Supplementary items are numbered with the prefix S.
Still to complete before submission: department, city and ORCID in the affiliation, and any statement the journal's policies require.
`../main.tex` is the complete long version in a plain article class, with the Supplement as appendices.
