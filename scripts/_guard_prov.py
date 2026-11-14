#!/usr/bin/env python3
"""Every "the convention of script N" / "identical to scripts ..." claim,
checked against which scripts actually declare the bandwidth and the
noise estimator; plus a diff of results/ against a previous release.

Round 13-18 added five cheap guards against the defect classes that survived
twelve rounds.  Run all five from anywhere:

    python3 scripts/_guard_titles.py
    python3 scripts/_guard_prov.py
    python3 scripts/_guard_nums.py
    python3 scripts/_guard_readme.py
    python3 scripts/_guard_refwords.py

Each prints CLEAN or lists the offenders.  They need no arguments and read only
the archive they sit in.
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
Z = os.path.dirname(HERE) + os.sep
# the manuscript sources live one level above the archive
MSDIR = os.path.dirname(Z.rstrip(os.sep)) + os.sep

# The article's LaTeX sources are not part of the public archive: this guard
# audits the archive AGAINST them, so it is an author tool and skips cleanly
# when they are not beside it (a clone of the published repository, say).
_SRC = [MSDIR + n for n in ("paper.tex", "supplement.tex", "theory_supplement.tex")]
if not all(os.path.exists(f) for f in _SRC):
    print("%s: SKIPPED -- the article sources (paper.tex, supplement.tex, "
          "theory_supplement.tex) are not beside this archive, so there is "
          "nothing to audit against. This is expected in a clone of the public "
          "repository; the guard is an author tool." % os.path.basename(__file__))
    raise SystemExit(0)
LETTER = os.path.join(os.path.dirname(Z.rstrip(os.sep)), "cover_letter.txt")
import io,re,glob,decimal,subprocess

P=os.environ.get('PREV_RELEASE','')  # set to a previous release's root to diff
# (iii) which result files moved since a previous release.  Set PREV_RELEASE to
# that release's Zenodo/ root to enable; skipped otherwise.
if P and os.path.isdir(os.path.join(P, "results")):
    moved, new = [], []
    for f in sorted(os.listdir(Z + "results")):
        a_, b_ = os.path.join(P, "results", f), Z + "results/" + f
        if os.path.isdir(b_):
            continue
        if not os.path.exists(a_):
            new.append(f); continue
        if open(a_, "rb").read() != open(b_, "rb").read():
            moved.append(f)
    print("RESULTS DIFF vs %s" % P)
    print("  changed: %s" % (moved or "none"))
    print("  new:     %s" % (new or "none"))
    print("  -> grep every value that moved across all documents before shipping.")
else:
    print("RESULTS DIFF: skipped (set PREV_RELEASE to a previous Zenodo/ root)")
print()

# (ii) provenance: "the convention of script N" / "identical to scripts"
pat=re.compile(r'(?:convention of scripts?|identical to scripts?|those of scripts?|as in scripts?|same \w+ convention as scripts?)\s+([\d/,\s and]+)')
hits=[]
for p in sorted(glob.glob(Z+'scripts/*.py'))+[f for f in [MSDIR+'paper.tex',MSDIR+'supplement.tex',MSDIR+'theory_supplement.tex',Z+'README.md'] if os.path.exists(f)]:
    for i,l in enumerate(io.open(p,encoding='utf-8',errors='replace'),1):
        m=pat.search(l)
        if m: hits.append((os.path.basename(p),i,m.group(0).strip()))
print("\nPROVENANCE CLAIMS (%d):"%len(hits))
# the two facts to check against
# the numbered scripts only: the shared helpers (_loaders, _calib, _guard_*)
# have no leading number, so basename.split('_')[0] is '' for them and the
# sorts below cannot cast that to int
def _num(path):
    n = os.path.basename(path).split('_')[0]
    return n if n.isdigit() else None
# Four shared conventions are checkable, and a citation must resolve to one of
# them: the HFRAC/NGRID bandwidth grid, the clipped MAD, the 0.05-nat decision
# band of script 11, and the local-quadratic estimator of script 38.  With only
# the first two, two accurate citations were reported as offenders --
# 45_accuracy_only.py cites the band, _calib.py cites the estimator.
def _declaring(test):
    return {n for n in (_num(q) for q in glob.glob(Z+'scripts/*.py')
                        if test(io.open(q,encoding='utf-8',errors='replace').read())) if n}
hfrac=_declaring(lambda s: 'HFRAC' in s)
mad  =_declaring(lambda s: '0.005, 0.25' in s or '0.005,0.25' in s)
band =_declaring(lambda s: 'BAND = 0.05' in s or 'BAND=0.05' in s)
estim=_declaring(lambda s: '0.5 * ((x - xc) / h) ** 2' in s
                        or '0.5 * (x - xc) ** 2' in s)
print("  scripts declaring HFRAC/NGRID: %s"%' '.join(sorted(hfrac,key=int)))
print("  scripts implementing the clipped MAD: %s"%' '.join(sorted(mad,key=int)))
print("  scripts declaring the 0.05-nat band: %s"%' '.join(sorted(band,key=int)))
print("  scripts implementing the local quadratic: %s"%' '.join(sorted(estim,key=int)))
known=hfrac|mad|band|estim
unresolved=False
for f,i,t in hits:
    nums=re.findall(r'\d+',t)
    bad=[n for n in nums if n not in known and len(n)<=3]
    unresolved=unresolved or bool(bad)
    flag='  <-- cites a script that declares none of the four' if bad else ''
    print("  %-28s:%-5d %s%s"%(f,i,t[:72],flag))
print("  PROVENANCE: see the flagged lines above." if unresolved
      else "  PROVENANCE CLEAN: every citation resolves.")
