#!/usr/bin/env python3
"""Every high-precision number (3-6 decimals) in the three .tex files,
required to match some value in some CSV under results/ or data/.

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

# pool every numeric token appearing in any results/ or data/ CSV
pool=set()
for f in glob.glob(Z+'results/*.csv')+glob.glob(Z+'data/*.csv')+glob.glob(Z+'results/*/*.csv'):
    try: txt=io.open(f,encoding='utf-8',errors='replace').read()
    except Exception: continue
    for m in re.finditer(r'-?\d+\.\d+',txt):
        v=m.group(0)
        try: x=float(v)
        except Exception: continue
        for nd in range(1,7):
            pool.add(round(x,nd)); pool.add(round(x*100,nd)); pool.add(round(x*1000,nd))
        pool.add(round(x)); pool.add(round(x*100))
    for m in re.finditer(r'(?<![\d.])\d+(?![\d.])',txt):
        pool.add(int(m.group(0)))
print("pool: %d distinct numeric forms from %d CSVs"%(len(pool),len(glob.glob(Z+'results/*.csv'))+len(glob.glob(Z+'data/*.csv'))))
bad=[]
for p in [MSDIR+'paper.tex',MSDIR+'supplement.tex',MSDIR+'theory_supplement.tex']:
    for i,l in enumerate(io.open(p,encoding='utf-8'),1):
        if l.lstrip().startswith('%'): continue
        for m in re.finditer(r'(?<![\d.\\A-Za-z])(\d+\.\d{3,6})(?![\d])',l):
            v=float(m.group(1))
            if not any(abs(v-q)<1e-9 for q in (round(v,k) for k in range(1,7))): pass
            if v not in pool:
                bad.append((os.path.basename(p),i,m.group(1),l.strip()[:80]))
# Known and verified exceptions: a hand-worked counterexample in the appendix
# that is a self-contained calculation, not an archive measurement.  Recomputed
# exactly in round 17: nu = (delta_(2,0.2)+delta_(0.2,2))/2, x1=(1.5,0.1),
# x2=(0.1,1.5) gives h1=h2=0.327666, h12=0.029599, D=-0.077766.
ALLOW = {"0.327666", "0.029599", "0.077766"}
bad = [t for t in bad if t[2] not in ALLOW]
print("\nHIGH-PRECISION NUMBERS (3-6 dp) IN THE TEX WITH NO MATCH IN ANY CSV:")
if not bad: print("  CLEAN: none.")
for b in bad[:40]: print("  %s:%d  %s   | %s"%b)
print("  total:",len(bad))
