#!/usr/bin/env python3
"""The README's structural self-claims, resolved against the archive itself.

Round 18 added this fourth guard after two of the README's counts drifted: a
table was added to the paper, and two result files that the note called absent
were restored, which silently falsified three further sentences about them.
Every number this file checks is one the README states in prose.  Run from
anywhere:

    python3 scripts/_guard_readme.py

Prints CLEAN or lists the offenders.  Needs no arguments; reads only the archive
it sits in and the manuscript sources one level above.
"""
import os, re, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(HERE) + os.sep
SRC = os.path.dirname(ARCH.rstrip(os.sep)) + os.sep
# Checks that read the article's LaTeX sources are author-only: a clone of the
# public repository has no paper.tex beside it.  HAVE_SRC gates those checks so
# the archive-internal ones still run and the guard still reports CLEAN.
HAVE_SRC = all(os.path.exists(SRC + n) for n in
               ("paper.tex", "supplement.tex", "theory_supplement.tex"))
if not HAVE_SRC:
    print("_guard_readme: the article sources are not beside this archive; "
          "checking only the claims that do not need them.")


README = open(os.path.join(ARCH, "README.md")).read()
# the notes live in a blockquote: strip the "> " prefixes and collapse the
# wrapping so a sentence that breaks across lines still matches as one string.
FLAT = re.sub(r"\n>\s?", " ", README)
FLAT = re.sub(r"[ \t]+", " ", FLAT)
PAPER = open(os.path.join(SRC, "paper.tex")).read() if HAVE_SRC else ""
bad = []


def claim(text, ok, got=""):
    if not ok:
        bad.append("%s (%s)" % (text, got))


def claim_src(text, ok, got=""):
    """A claim that can only be checked against the article's LaTeX sources.
    Silently skipped when they are not beside the archive."""
    if HAVE_SRC:
        claim(text, ok, got)


WORD_N = {"none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "thirteen": 13, "seventeen": 17}
WORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
        12: "twelve", 13: "thirteen", 17: "seventeen"}

# -- the paper's own shape, as the note on numbering states it ---------------
nfig = len(re.findall(r"\\begin\{figure", PAPER))
ntab = len(re.findall(r"\\begin\{table", PAPER))
m = re.search(r"current paper has (\w+) figures? and (\w+) tables?", FLAT)
claim_src("note on numbering: figure/table count", m and
      m.group(1).lower() == WORD.get(nfig) and m.group(2) == WORD.get(ntab),
      "paper has %d/%d, README says %s/%s" % (nfig, ntab, m.group(1), m.group(2)) if m else "sentence gone")

env = r"(theorem|lemma|proposition|corollary|definition|remark|assumption|claim)"
nst = len(re.findall(r"\\begin\{" + env + r"\}", PAPER))
m = re.search(r"numbered statements run `1`--`(\d+)`", FLAT)
claim_src("note on numbering: statement range", m and int(m.group(1)) == nst,
      "paper has %d, README says %s" % (nst, m.group(1)) if m else "sentence gone")

# -- figures shipped beside the sources -------------------------------------
shipped = set(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SRC, "figures", "*.pdf"))) if HAVE_SRC else set()
m = re.search(r"figure directory\s+beside the sources holds all (\w+)", FLAT)
claim_src("Layout: figures shipped", m and m.group(1).lower() == WORD.get(len(shipped)),
      "%d shipped, README says %s" % (len(shipped), m.group(1)) if m else "sentence gone")

seg = FLAT.split("holds all")[1].split("A note on numbering")[0] if m else ""
parts = dict((w, int(WORD_N.get(w, 0))) for w in re.findall(
    r"(?:paper's|Appendix's|report's) (\w+)", seg))
tally = sum(parts.values())
claim_src("Layout: 3 + 2 + 6 decomposition sums to the figures shipped",
      tally == len(shipped), "parts %s sum to %d, %d shipped" % (parts, tally, len(shipped)))
listed = set(re.findall(r"`(fig_[a-z_]+)`", seg))
claim_src("Layout: every figure it names is shipped", listed == shipped,
      "named-not-shipped %s, shipped-not-named %s" % (sorted(listed - shipped), sorted(shipped - listed)))

# -- absent result files, and the rows that name them ------------------------
rows = [l for l in README.splitlines() if l.startswith("|")]
toks = set()
for l in rows:
    toks |= set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:csv|npy|json))`", l))


def where(b):
    r = os.path.exists(os.path.join(ARCH, "results", b))
    d = os.path.exists(os.path.join(ARCH, "data", b)) or any(
        b in fs for _, _, fs in os.walk(os.path.join(ARCH, "data")))
    return r, d


# Every file a table in README.md names must exist in the archive.  This is
# checked directly rather than against a sentence that counts the exceptions,
# so there is no prose to keep in step and no way to satisfy it by editing text.
absent = sorted(t for t in toks if not any(where(os.path.basename(t))))
claim("every result file the tables name exists", not absent,
      "absent: %s" % sorted(os.path.basename(a) for a in absent))

# -- the data/ vs results/ split the note quotes -----------------------------
INPUTS = {"grid_factorial.csv", "matbench_models.csv", "nb201_accuracy.csv",
          "nb201_hwnas.csv", "tabarena_metrics.csv"}
both = set(); onlydata = set()
for t in toks:
    b = os.path.basename(t)
    r, d = where(b)
    if r and d:
        both.add(b)
    elif d and not r and b not in INPUTS and not b.startswith("field_"):
        onlydata.add(b)
m = re.search(r"(\w+) of the files the maps name\s+sit under `data/`", FLAT)
claim("note: result files sitting under data/", m and m.group(1).lower() == WORD.get(len(onlydata)),
      "%d misplaced, README says %s" % (len(onlydata), m.group(1)) if m else "sentence gone")
m = re.search(r"and (\w+) more are in both places", FLAT)
claim("note: files present in both trees", m and m.group(1).lower() == WORD.get(len(both)),
      "%d in both, README says %s" % (len(both), m.group(1)) if m else "sentence gone")

# -- the cover letter's claims about the Online Appendix ---------------------
LETTER = os.path.join(SRC, "cover_letter.txt")
if HAVE_SRC and os.path.exists(LETTER):
    L = open(LETTER).read()
    SUP = open(os.path.join(SRC, "supplement.tex")).read()
    senv = r"(theorem|lemma|proposition|corollary|definition|remark|assumption|claim)"
    labs = set()
    for mm in re.finditer(r"\\begin\{" + senv + r"\}(?:\[[^\]]*\])?"
                          r"((?:(?!\\begin\{" + senv + r"\}).)*?)\\end\{\1\}",
                          PAPER, re.S):
        labs |= {x for x in re.findall(r"\\label\{([^}]+)\}", mm.group(2))
                 if not x.startswith(("eq:", "fig:", "tab:"))}
    proved = set(re.findall(r"\\begin\{proofof\}\{[^}]*?\\ref\{([^}]+)\}", SUP))
    claim("cover letter: the Appendix proves every numbered statement",
          labs <= proved, "unproved: %s" % sorted(labs - proved))
    npf = len(re.findall(r"\\begin\{proofof\}", SUP)) + len(re.findall(r"\\begin\{proof\}", SUP))
    nenv = len(re.findall(r"\\begin\{" + senv + r"\}", SUP))
    m = re.search(r"in (\d+) proofs across (\d+) numbered environments", L)
    claim("cover letter: proof and environment counts",
          m and int(m.group(1)) == npf and int(m.group(2)) == nenv,
          "actual %d/%d, letter says %s/%s" % (npf, nenv, m.group(1), m.group(2)) if m else "sentence gone")

# -- the cover letter's page counts, when the PDFs are built beside the sources
def _pages(pdf):
    import subprocess
    try:
        out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None


if HAVE_SRC and os.path.exists(LETTER):
    pg = {d: _pages(os.path.join(SRC, d + ".pdf")) for d in
          ("paper", "supplement", "theory_supplement")}
    if all(pg.values()):
        L = open(LETTER).read()
        m = re.search(r"manuscript \(paper\.pdf\) is (\d+) pages", L)
        claim("cover letter: paper page count", m and int(m.group(1)) == pg["paper"],
              "built %d, letter says %s" % (pg["paper"], m.group(1)) if m else "sentence gone")
        m = re.search(r"\(supplement\.pdf, (\d+) pages\)", L)
        claim("cover letter: Online Appendix page count",
              m and int(m.group(1)) == pg["supplement"],
              "built %d, letter says %s" % (pg["supplement"], m.group(1)) if m else "sentence gone")
        m = re.search(r"\(theory_supplement\.pdf, (\d+) pages\)", L)
        claim("cover letter: technical report page count",
              m and int(m.group(1)) == pg["theory_supplement"],
              "built %d, letter says %s" % (pg["theory_supplement"], m.group(1)) if m else "sentence gone")
        # the internal split: the References heading's page, read from the PDF
        m = re.search(r"theory\s+itself is pages (\d+)-(\d+) and the references pages (\d+)-(\d+)", L)
        if m:
            import subprocess
            refpg = None
            for pageno in range(1, pg["paper"] + 1):
                try:
                    txt = subprocess.run(["pdftotext", "-f", str(pageno), "-l", str(pageno),
                                          os.path.join(SRC, "paper.pdf"), "-"],
                                         capture_output=True, text=True).stdout
                except Exception:
                    break
                # the running head precedes it, so scan lines, not the page start
                if any(ln.strip() == "References" for ln in txt.splitlines()):
                    refpg = pageno
                    break
            if refpg:
                claim("cover letter: references start page",
                      int(m.group(3)) == refpg and int(m.group(2)) == refpg - 1,
                      "References on page %d, letter says theory ends %s and refs start %s"
                      % (refpg, m.group(2), m.group(3)))
                claim("cover letter: references end page",
                      int(m.group(4)) == pg["paper"],
                      "paper ends %d, letter says %s" % (pg["paper"], m.group(4)))
        tot = pg["paper"] + pg["supplement"]
        for pat, what in ((r"nothing else: (\d+) pages", "referee total"),
                          (r"Online Appendix come to (\d+) pages", "combined total")):
            m = re.search(pat, L)
            claim("cover letter: %s" % what, m and int(m.group(1)) == tot,
                  "built %d, letter says %s" % (tot, m.group(1)) if m else "sentence gone")

# -- numbered-script gaps ----------------------------------------------------
nums = sorted(int(mm.group(1)) for mm in
              (re.match(r"^(\d+)_", os.path.basename(p)) for p in
               glob.glob(os.path.join(ARCH, "scripts", "*.py"))) if mm)
gaps = [n for n in range(1, max(nums) + 1) if n not in nums]
m = re.search(r"Script numbers run to `\d+` with gaps at (.+?):", FLAT, re.S)
quoted = sorted(int(x) for x in re.findall(r"`(\d+)`", m.group(1))) if m else []
claim("note: numbered-script gaps", quoted == gaps,
      "actual %s, README says %s" % (gaps, quoted))

print("README CLEAN" if not bad else "README GUARD FOUND %d:\n  - %s" % (len(bad), "\n  - ".join(bad)))
