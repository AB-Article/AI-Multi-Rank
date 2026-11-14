#!/usr/bin/env python3
"""Every "Word~\\ref{label}" in the three .tex files, resolved against the
environment that actually declares that label.

Round 18 added this fifth guard after one statement was found carried as a
`corollary` environment in the supplement but cited as "Proposition" in the
technical report, so the same number was printed under two different names.
The check is purely structural: it compares the word an author typed in front
of a reference with the environment the label sits in, across documents, which
is what `xr-hyper` makes easy to get wrong.  Run from anywhere:

    python3 scripts/_guard_refwords.py

Prints CLEAN or lists the offenders.
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.dirname(HERE)) + os.sep
DOCS = ["paper.tex", "supplement.tex", "theory_supplement.tex"]
# The article's LaTeX sources are not part of the public archive: this guard
# audits the archive AGAINST them, so it is an author tool and skips cleanly
# when they are not beside it (a clone of the published repository, say).
_SRC = [SRC + n for n in ("paper.tex", "supplement.tex", "theory_supplement.tex")]
if not all(os.path.exists(f) for f in _SRC):
    print("%s: SKIPPED -- the article sources are not beside this archive, so "
          "there is nothing to audit against. This is expected in a clone of "
          "the public repository; the guard is an author tool."
          % os.path.basename(__file__))
    raise SystemExit(0)

STMT = ("theorem", "lemma", "proposition", "corollary", "definition", "remark",
        "assumption", "claim", "conjecture", "example", "algorithm")

# label -> declaring environment
owner = {}
for f in DOCS:
    S = open(os.path.join(SRC, f)).read()
    for m in re.finditer(r"\\begin\{(" + "|".join(STMT) + r")\*?\}(?:\[[^\]]*\])?"
                         r"((?:(?!\\begin\{(?:" + "|".join(STMT) + r")\*?\}).)*?)"
                         r"\\end\{\1\*?\}", S, re.S):
        for lab in re.findall(r"\\label\{([^}]+)\}", m.group(2)):
            owner[lab] = m.group(1)
    for env in ("table", "figure"):
        for m in re.finditer(r"\\begin\{" + env + r"\*?\}(.*?)\\end\{" + env + r"\*?\}", S, re.S):
            for lab in re.findall(r"\\label\{([^}]+)\}", m.group(1)):
                owner.setdefault(lab, env)
    for m in re.finditer(r"\\(section|subsection|subsubsection|paragraph)\*?"
                         r"\{[^}]*\}\s*\\label\{([^}]+)\}", S):
        owner.setdefault(m.group(2), "section")

FAMILY = {"section": "section", "subsection": "section", "subsections": "section",
          "appendix": "section", "table": "table", "tables": "table",
          "figure": "figure", "figures": "figure"}
for k in STMT:
    FAMILY[k] = k
    FAMILY[k + "s"] = k

bad = []
for f in DOCS:
    txt = " ".join(open(os.path.join(SRC, f)).read().split())
    for m in re.finditer(r"\b([A-Z][a-z]+)(?:~|\s)\\ref\{([^}]+)\}", txt):
        word, lab = m.group(1).lower(), m.group(2)
        if word not in FAMILY or lab not in owner:
            continue
        want, got = FAMILY[word], owner[lab]
        if want != got:
            bad.append("%s: %s~\\ref{%s} -- but %s is a %s"
                       % (f, m.group(1), lab, lab, got))

print("REFWORDS CLEAN" if not bad
      else "REFWORD MISMATCHES (%d):\n  - %s" % (len(bad), "\n  - ".join(sorted(set(bad)))))
