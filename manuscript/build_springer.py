"""Build manuscript/springer/manuscript_sn.tex (Springer Nature template, single file, flat files)
from manuscript/manuscript.tex. Run from the repository root."""
import re, shutil, pathlib
root = pathlib.Path("manuscript"); out = root / "springer"; out.mkdir(exist_ok=True)
s = (root / "manuscript.tex").read_text()
body = s[s.index("\\section{Introduction}"): s.index("\\bibliographystyle")]
bib = s[s.index("\\begin{thebibliography}"): s.index("\\end{thebibliography}")]
body = body.replace("\\section*{Declarations}", "\\section*{Statements and Declarations}")
# citations -> numeric
body = re.sub(r"\\citeyearpar\{", r"\\cite{", body); body = re.sub(r"\\citep\{", r"\\cite{", body)
body = re.sub(r"\\citeyear\{", r"\\cite{", body)
body = body.replace("(Section~2)", "(Section~\\ref{sec:related})").replace("Section~2)", "Section~\\ref{sec:related})")
body = body.replace("\\section{Related work}", "\\section{Related work}\\label{sec:related}")
body = body.replace("\\newcommand", "\\newcommand")
# inline tables
def inline(m):
    return (root / "tables" / m.group(1)).read_text()
body = re.sub(r"\\input\{tables/([^}]+)\}", inline, body)
# wide tables and figures span both columns
for lab in ("tab:sim", "tab:box", "tab:nonsmooth", "tab:realval", "tab:multi", "tab:adult", "tab:german"):
    pat = re.compile(r"\\begin\{table\}\[h\]((?:(?!\\end\{table\}).)*?\\label\{" + re.escape(lab) + r"\}(?:(?!\\end\{table\}).)*?)\\end\{table\}", re.S)
    body = pat.sub(lambda m: "\\begin{table*}[t]" + m.group(1) + "\\end{table*}", body)
body = body.replace("\\begin{table}[h]", "\\begin{table}[t]")
body = re.sub(r"\\begin\{figure\}\[h\]\\centering\n\\includegraphics\[width=\\textwidth\]\{\.\./figures/([^}]+)\.pdf\}",
              lambda m: "\\begin{figure*}[t]\\centering\n\\includegraphics[width=0.9\\textwidth]{" + m.group(1) + "}", body)
body = body.replace("\\end{figure}", "\\end{figure*}")
body = re.sub(r"\\caption\{(Retention \(left\))", r"\\caption{\1", body)
body = body.replace("\\begin{tabular}{ll}\\toprule\nCriterion & Definition", "\\begin{tabular}{lp{5.4cm}}\\toprule\nCriterion & Definition")
body = body.replace("\\begin{table}[t]\\centering\\small\n\\caption{Number of models retained", "\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\caption{Number of models retained")
# bibliography: drop natbib labels
bib = re.sub(r"\\bibitem\[[^\]]*\]\{", r"\\bibitem{", bib)
bib = bib.replace("\\providecommand{\\natexlab}[1]{#1}\n", "")
abstract = r"""Trustworthy machine learning is assessed on several criteria at once: error, group fairness, calibration and robustness. Candidate models are usually compared through point estimates on one finite test sample, and the winner or the Pareto front of those estimates is reported as if it were a population fact. We describe a trust confidence set (TCS), a data-dependent set of models that contains every population-nondominated model with probability at least $1-\alpha$, asymptotically. It uses a paired bootstrap of one shared test sample, studentised pairwise differences, an intersection--union step across criteria and a maximum over rivals, so it accepts criteria such as calibration error and fairness gaps. In simulation, the TCS retained all nondominated models in at least 99.7\% of replications, whereas the Pareto front of point estimates did so in 34--42\%, and the TCS excluded more dominated models than a Bonferroni box procedure. Validated against large held-out references on twelve real data settings, the point-estimate Pareto front recovered the reference set in a median of 39\% of test samples of 500 rows and the composite-score winner matched in a median of 36\%; the TCS always retained the reference set but returned almost every model, because few were truly dominated. Step-down, equivalence margins and a refit bootstrap added little."""
words = len(re.sub(r"\$[^$]*\$|\\[a-z]+", "x", abstract).split())
print("abstract words (approx):", words)
head = r"""%Version 3 October 2023 template; build with the official sn-jnl.cls
\documentclass[pdflatex,sn-mathphys-num,iicol]{sn-jnl}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{url}
\theoremstyle{thmstyleone}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\theoremstyle{thmstyletwo}
\newtheorem{remark}{Remark}
\newcommand{\E}{\mathbb{E}}
\renewcommand{\Pr}{\mathbb{P}}
\raggedbottom
\begin{document}
\title[Trust confidence sets]{Trust confidence sets: statistical inference for selecting models under several trustworthiness criteria}
\author*[1]{\fnm{Quang-Vinh} \sur{Dang}}\email{vinh.dq4@buv.edu.vn}
\affil*[1]{\orgname{British University Vietnam}, \orgaddress{\city{Hung Yen}, \country{Vietnam}}}
\abstract{""" + abstract + r"""}
\keywords{trustworthy machine learning, model selection, Pareto dominance, confidence set, bootstrap, algorithmic fairness}
\maketitle
"""
tail = "\\end{thebibliography}\n\\end{document}\n"
(out / "manuscript_sn.tex").write_text(head + body + bib + tail)
for f in ("fig_simulation", "fig_pareto_prob", "fig_rank_intervals"):
    shutil.copy(f"figures/{f}.pdf", out / f"{f}.pdf")
