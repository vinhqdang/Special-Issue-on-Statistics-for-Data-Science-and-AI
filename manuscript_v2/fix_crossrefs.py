"""Rewrite cross-document references: main files use \\SIref for labels defined in SI files, SI files use \\Mref for labels defined in main files."""
import re, pathlib
S = pathlib.Path("sections")
MAIN = ["introduction", "related", "setting", "design", "theory", "numcheck", "testing", "testing_sim", "empirical", "extra_splits", "extra_population", "baselines", "ablations", "guidance", "discussion"]
SI = ["appendix_proofs", "si_theory_extra", "si_testing", "si_testing_sim", "simulation", "si_baselines", "si_ablations", "appendix_ablations", "appendix_data"]
def expand(t):
    return re.sub(r"\\input\{tables/([^}]+?)(?:\.tex)?\}", lambda m: (pathlib.Path("tables") / (m.group(1) + ".tex")).read_text(), t)
lab = lambda t: set(re.findall(r"\\label\{([^}]+)\}", expand(t)))
main_labels = set().union(*[lab((S / f"{n}.tex").read_text()) for n in MAIN])
si_labels = set().union(*[lab((S / f"{n}.tex").read_text()) for n in SI])
both = main_labels & si_labels; print("labels defined in both:", both)
def rewrite(path, other, macro):
    t = path.read_text(); n = 0
    def rep(m):
        nonlocal n
        cmd, keys = m.group(1), m.group(2)
        ks = [k.strip() for k in keys.split(",")]
        if all(k in other and k not in lab_here for k in ks):
            n += 1
            inner = ",".join("\\%s{%s}" % (macro, k) for k in ks) if len(ks) > 1 else "\\%s{%s}" % (macro, ks[0])
            return "(" + inner + ")" if cmd == "eqref" else inner
        return m.group(0)
    lab_here = lab(t)
    t2 = re.sub(r"\\(ref|eqref)\{([^}]+)\}", rep, t)
    if t2 != t: path.write_text(t2)
    return n
for n in MAIN: print(n, rewrite(S / f"{n}.tex", si_labels, "SIref"))
for n in SI: print(n, rewrite(S / f"{n}.tex", main_labels, "Mref"))
# any remaining refs to labels that exist nowhere
allref = set()
for n in MAIN + SI:
    allref |= set(re.findall(r"\\(?:ref|eqref|SIref|Mref)\{([^}]+)\}", expand((S / f"{n}.tex").read_text())))
print("refs to labels defined nowhere:", sorted(k for k in allref if k not in main_labels | si_labels))
