"""Build manuscript_v2/springer/manuscript_sn.tex: the Springer Nature (sn-jnl, iicol) single-file version of
manuscript_v2/main.tex, with sections and tables inlined, flat figure files and numeric citations.
Run from the repository root:  python manuscript_v2/build_springer_v2.py
Then copy the official sn-jnl.cls and sn-mathphys-num.bst next to the output and run pdflatex/bibtex."""
import re, shutil, pathlib
root = pathlib.Path("manuscript_v2"); out = root / "springer"; out.mkdir(exist_ok=True)

def read(path): return (root / path).read_text()

def inline(text):
    text = re.sub(r"\\input\{(sections/[^}]+)\}", lambda m: inline(read(m.group(1) + ".tex")), text)
    text = re.sub(r"\\input\{(tables/[^}]+?)(?:\.tex)?\}", lambda m: read(m.group(1) + ".tex"), text)
    return text

main = read("main.tex")
def sec(name): return inline(read(f"sections/{name}.tex"))
abs_src = read("sections/abstract.tex")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", abs_src, re.S).group(1).strip()
kw = re.search(r"\\textbf\{Keywords:\}\s*(.*?)\s*\\end\{abstract\}", abs_src, re.S)
keywords = (re.search(r"Keywords:\}\s*([^\n]*)", abs_src).group(1).strip() if re.search(r"Keywords:\}", abs_src) else "")
abstract = re.sub(r"\\medskip\\noindent\\textbf\{Keywords:\}.*", "", abstract, flags=re.S).strip()
keywords = keywords.replace(";", ",")
body_names = ["introduction", "related", "setting", "design", "theory", "simulation", "empirical", "baselines", "ablations", "guidance", "discussion"]
app_names = ["appendix_proofs", "appendix_ablations", "appendix_data"]
body = "\n".join(sec(n) for n in body_names)
appendix = "\n".join(sec(n) for n in app_names)

def fix(t):
    t = re.sub(r"\\citeyearpar\{", r"\\cite{", t)
    t = re.sub(r"\\citealp\{", r"\\cite{", t)
    # floats: wide tables and figures span both columns
    t = re.sub(r"\\begin\{table\}(\[[^\]]*\])?", r"\\begin{table*}[t]", t); t = t.replace("\\end{table}", "\\end{table*}")
    t = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?", r"\\begin{figure*}[t]", t); t = t.replace("\\end{figure}", "\\end{figure*}")
    t = re.sub(r"\\includegraphics(\[[^\]]*\])?\{figures/([^}]+)\}", lambda m: "\\includegraphics" + (m.group(1) or "") + "{" + pathlib.Path(m.group(2)).stem + "}", t)
    t = t.replace("\\centering\\small", "\\centering\\footnotesize")
    return t
body, appendix = fix(body), fix(appendix)
figs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body + appendix))
for f in figs:
    src = root / "figures" / (f + ".pdf")
    if src.exists(): shutil.copy(src, out / (f + ".pdf"))
    else: print("missing figure", f)

head = r"""%Version 3 October 2023 template; compile with the official sn-jnl.cls
\documentclass[pdflatex,sn-mathphys-num,iicol]{sn-jnl}
\usepackage{graphicx}\usepackage{multirow}\usepackage{amsmath,amssymb,amsfonts}\usepackage{amsthm}
\usepackage{mathrsfs}\usepackage{xcolor}\usepackage{textcomp}\usepackage{booktabs}\usepackage{url}
\usepackage[title]{appendix}\usepackage{algorithm}\usepackage{algorithmicx}\usepackage{algpseudocode}
\theoremstyle{thmstyleone}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{thmstyletwo}
\newtheorem{remark}{Remark}
\newtheorem{assumption}{Assumption}
\theoremstyle{thmstylethree}
\newtheorem{definition}{Definition}
\raggedbottom
\begin{document}
\title[Labelling where models disagree]{Labelling where models disagree: label-efficient comparison of classifiers on group fairness}
\author*[1]{\fnm{Vinh} \sur{Dang}}\email{vinh.dang3@rmit.edu.vn}
\affil*[1]{\orgname{RMIT University Vietnam}, \country{Vietnam}}
"""
doc = head + "\\abstract{" + abstract + "}\n\\keywords{" + keywords + "}\n\\maketitle\n" + body + \
      "\n\\bibliography{references}\n\\begin{appendices}\n" + appendix + "\n\\end{appendices}\n\\end{document}\n"
(out / "manuscript_sn.tex").write_text(doc)
shutil.copy(root / "references.bib", out / "references.bib")
print("abstract words (approx):", len(re.sub(r"\$[^$]*\$|\\[a-zA-Z]+", "x", abstract).split()), "| keywords:", keywords)
