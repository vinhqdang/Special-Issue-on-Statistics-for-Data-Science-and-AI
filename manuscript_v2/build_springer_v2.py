"""Build the Springer Nature (sn-jnl, iicol) main paper and Supplementary Information from manuscript_v2.

    python manuscript_v2/build_springer_v2.py [OFFICIAL_TEMPLATE_DIR]

OFFICIAL_TEMPLATE_DIR must contain sn-jnl.cls and bst/sn-mathphys-num.bst (the Springer Nature template package);
the class files are copied into a scratch build directory and are not committed. The script writes
    manuscript_v2/springer/manuscript_sn.tex      main paper (single file, flat folder)
    manuscript_v2/springer/supplement_sn.tex      Supplementary Information (single file)
    manuscript_v2/springer/*.pdf (figures), references.bib
and, with the template available, compiles both to PDF in a scratch directory and copies the PDFs back as
manuscript_sn_draft.pdf and supplement_sn_draft.pdf.

Cross-references between the two documents: main text uses \\SIref{key} for labels that live in the SI and the SI uses
\\Mref{key} for labels that live in the main paper. The script compiles both once, reads the numbers from the .aux
files and writes them into the preamble (no \\input of other files, as the journal requires), then recompiles.
"""
import re, sys, shutil, pathlib, subprocess, tempfile, os
root = pathlib.Path("manuscript_v2"); out = root / "springer"; out.mkdir(exist_ok=True)
OFFICIAL = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None

MAIN = ["introduction", "related", "setting", "design", "theory", "numcheck", "empirical", "extra_splits",
        "extra_population", "baselines", "ablations", "guidance", "discussion"]
SI = ["appendix_proofs", "si_theory_extra", "simulation", "si_baselines", "si_ablations", "appendix_ablations", "appendix_data"]


def read(p): return (root / p).read_text()


def inline(text):
    text = re.sub(r"\\input\{(sections/[^}]+)\}", lambda m: inline(read(m.group(1) + ".tex")), text)
    text = re.sub(r"\\input\{(tables/[^}]+?)(?:\.tex)?\}", lambda m: read(m.group(1) + ".tex"), text)
    return text


def sec(n):
    f = root / "sections" / f"{n}.tex"
    return inline(f.read_text()) if f.exists() else ""


def fix(t):
    t = re.sub(r"\\citeyearpar\{", r"\\cite{", t); t = re.sub(r"\\citealp\{", r"\\cite{", t)
    t = re.sub(r"\\begin\{table\}(\[[^\]]*\])?", r"\\begin{table*}[t]", t); t = t.replace("\\end{table}", "\\end{table*}")
    t = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?", r"\\begin{figure*}[t]", t); t = t.replace("\\end{figure}", "\\end{figure*}")
    t = re.sub(r"\\includegraphics(\[[^\]]*\])?\{figures/([^}]+)\}", lambda m: "\\includegraphics" + (m.group(1) or "") + "{" + pathlib.Path(m.group(2)).stem + "}", t)
    t = t.replace("\\centering\\small", "\\centering\\footnotesize")
    return t


abs_src = read("sections/abstract.tex")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", abs_src, re.S).group(1)
keywords = re.search(r"Keywords:\}\s*([^\n]*)", abs_src).group(1).strip().replace(";", ",")
abstract = re.sub(r"\\medskip\\noindent\\textbf\{Keywords:\}.*", "", abstract, flags=re.S).strip()

PRE = r"""%Version 3.1 December 2024 template; compile with the official sn-jnl.cls
\documentclass[pdflatex,sn-mathphys-num,iicol]{sn-jnl}
\usepackage{graphicx}\usepackage{multirow}\usepackage{amsmath,amssymb,amsfonts}\usepackage{amsthm}
\usepackage{mathrsfs}\usepackage{xcolor}\usepackage{textcomp}\usepackage{booktabs}\usepackage{url}
\usepackage{algorithm}\usepackage{algorithmicx}\usepackage{algpseudocode}
\theoremstyle{thmstyleone}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}[proposition]{Lemma}
\newtheorem{corollary}[proposition]{Corollary}
\theoremstyle{thmstyletwo}
\newtheorem{remark}{Remark}
\newtheorem{assumption}{Assumption}
\theoremstyle{thmstylethree}
\newtheorem{definition}{Definition}
\raggedbottom
\newcommand{\SIref}[1]{\ifcsname siref@#1\endcsname\csname siref@#1\endcsname\else\textbf{??}\fi}
\newcommand{\Mref}[1]{\ifcsname mref@#1\endcsname\csname mref@#1\endcsname\else\textbf{??}\fi}
"""
SI_PRE = r"""\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}
\renewcommand{\theproposition}{S\arabic{proposition}}
\renewcommand{\theremark}{S\arabic{remark}}
\renewcommand{\theassumption}{S\arabic{assumption}}
\renewcommand{\thedefinition}{S\arabic{definition}}
"""
TITLE = "Labelling where models disagree: label-efficient comparison of classifiers on group fairness"
AUTH = (r"\author*[1]{\fnm{Vinh} \sur{Dang}}\email{vinh.dang3@rmit.edu.vn}" "\n"
        r"\affil*[1]{\orgname{RMIT University Vietnam}, \country{Vietnam}}" "\n")


def mapping(prefix, labels):
    return "".join("\\expandafter\\gdef\\csname %s@%s\\endcsname{%s}\n" % (prefix, k, v) for k, v in sorted(labels.items()))


def write_docs(siref, mref):
    body = fix("\n".join(sec(n) for n in MAIN))
    sibody = fix("\n".join(sec(n) for n in SI))
    main = (PRE + "\\makeatletter\n" + mapping("siref", siref) + mapping("mref", mref) + "\\makeatother\n"
            "\\begin{document}\n\\title[Labelling where models disagree]{" + TITLE + "}\n" + AUTH +
            "\\abstract{" + abstract + "}\n\\keywords{" + keywords + "}\n\\maketitle\n" + body +
            "\n\\bibliography{references}\n\\end{document}\n")
    si = (PRE + SI_PRE + "\\makeatletter\n" + mapping("siref", siref) + mapping("mref", mref) + "\\makeatother\n"
          "\\begin{document}\n\\title[Supplementary Information]{Supplementary Information for: " + TITLE + "}\n" + AUTH +
          "\\abstract{Proofs, additional theoretical results, numerical checks of the theory, baseline definitions, ablation studies and data details for the main paper. "
          "Sections, tables, figures, equations and propositions are numbered with the prefix S; numbers without a prefix refer to the main paper.}\n"
          "\\keywords{supplementary information}\n\\maketitle\n" + sibody + "\n\\bibliography{references}\n\\end{document}\n")
    (out / "manuscript_sn.tex").write_text(main); (out / "supplement_sn.tex").write_text(si)
    figs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body + sibody))
    for f in figs:
        src = root / "figures" / (f + ".pdf")
        if src.exists(): shutil.copy(src, out / (f + ".pdf"))
        else: print("missing figure", f)
    shutil.copy(root / "references.bib", out / "references.bib")


def aux_labels(path):
    labs = {}
    if not path.exists(): return labs
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^{}]*)\}", path.read_text()):
        if m.group(2): labs[m.group(1)] = m.group(2)
    return labs


def compile_doc(build, name):
    for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):
        args = cmd + ([name] if cmd[0] == "bibtex" else ["-interaction=nonstopmode", name + ".tex"])
        subprocess.run(args, cwd=build, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    write_docs({}, {})
    print("main abstract words (approx):", len(re.sub(r"\$[^$]*\$|\\[a-zA-Z]+", "x", abstract).split()), "| keywords:", keywords)
    if OFFICIAL is None:
        print("no template directory given: sources written, nothing compiled"); sys.exit(0)
    build = pathlib.Path(tempfile.mkdtemp(prefix="snbuild_"))
    def stage():
        for f in out.iterdir():
            if f.suffix in (".tex", ".pdf", ".bib") and "draft" not in f.name: shutil.copy(f, build / f.name)
    cls = (OFFICIAL / "sn-jnl.cls").read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    (build / "sn-jnl.cls").write_bytes(cls); shutil.copy(OFFICIAL / "bst" / "sn-mathphys-num.bst", build / "sn-mathphys-num.bst")
    siref, mref = {}, {}
    for it in range(3):
        write_docs(siref, mref); stage()
        compile_doc(build, "manuscript_sn"); compile_doc(build, "supplement_sn")
        new_si, new_m = aux_labels(build / "supplement_sn.aux"), aux_labels(build / "manuscript_sn.aux")
        if new_si == siref and new_m == mref: break
        siref, mref = new_si, new_m
    write_docs(siref, mref); stage(); compile_doc(build, "manuscript_sn"); compile_doc(build, "supplement_sn")
    for n in ("manuscript_sn", "supplement_sn"):
        pdf = build / f"{n}.pdf"
        if pdf.exists():
            shutil.copy(pdf, out / f"{n.replace('_sn', '')}_sn_draft.pdf" if False else out / f"{n}_draft.pdf")
            pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
            print(n, [l for l in pages.splitlines() if l.startswith("Pages")])
            txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
            print("  unresolved '??':", txt.count("??"), "| log errors:", len(re.findall(r"^! ", (build / f"{n}.log").read_text(), re.M)))
        else: print(n, "NOT COMPILED; see", build / f"{n}.log")
    print("build dir:", build)
