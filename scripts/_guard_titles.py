#!/usr/bin/env python3
"""Every quoted statement/section title in the prose documents and in the
script docstrings, resolved against the real titles in the three .tex files.

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

tex=[MSDIR+'paper.tex',MSDIR+'supplement.tex',MSDIR+'theory_supplement.tex']
titles=set(); secs=set()
for p in tex:
    s=io.open(p,encoding='utf-8').read()
    for m in re.finditer(r'\\begin\{(theorem|proposition|corollary|lemma|definition|assumption|remark|conjecture)\}\[([^\]]*)\]',s):
        titles.add(m.group(2).strip())
    for m in re.finditer(r'\\(?:sub)*(?:section|paragraph)\*?\{([^}]*)\}',s):
        secs.add(m.group(1).strip())
def norm(t):
    t=re.sub(r'\\[a-zA-Z]+\s*','',t); t=t.replace('~',' ').replace('$','')
    t=re.sub(r'[{}]','',t); t=re.sub(r'\s+',' ',t)
    return t.strip().lower()
NT={norm(t) for t in titles}; NS={norm(t) for t in secs}
ALL=NT|NS
docs=[Z+'README.md',Z+'PRACTICAL_PIPELINE.md',LETTER]
docs=[d for d in docs if os.path.exists(d)]
bad=[]
for p in docs:
    for i,l in enumerate(io.open(p,encoding='utf-8'),1):
        for m in re.finditer(r'(?:Propositions?|Theorems?|Corollar(?:y|ies)|Lemmas?|Sections?|Prop\.|Thm\.|Cor\.)\s+((?:"[^"]+"(?:,\s*)?)+)',l):
            for q in re.findall(r'"([^"]+)"',m.group(1)):
                n=norm(q)
                if n and n not in ALL and not any(n in a or a in n for a in ALL):
                    bad.append((os.path.basename(p),i,q))
print("TITLE-POINTER GUARD over %d statement titles and %d section titles"%(len(NT),len(NS)))
if not bad: print("  CLEAN: every quoted title in the four prose documents resolves.")
for b in bad: print("  DEAD: %s:%d  %r"%b)
# script docstrings
bad2=[]
for p in sorted(glob.glob(Z+'scripts/*.py')):
    s=io.open(p,encoding='utf-8').read()
    head=s[:s.find('"""',s.find('"""')+3)] if s.count('"""')>=2 else s[:2000]
    for q in re.findall(r"Prop\.?\s*'([^']+)'",head)+re.findall(r'Prop\.?\s*"([^"]+)"',head):
        n=norm(q)
        if n in ALL or any(n in a for a in ALL): continue
        # labels are also acceptable nicknames
        bad2.append((os.path.basename(p),q))
print("\nSCRIPT-DOCSTRING NICKNAMES not matching a title (may be \\label suffixes):")
labs=set()
for p in tex:
    labs|={m.group(1).split(':')[-1].lower() for m in re.finditer(r'\\label\{([^}]*)\}',io.open(p,encoding='utf-8').read())}
unresolved=[(f,q) for f,q in bad2 if q.lower() not in labs]
if not unresolved: print("  CLEAN: every nickname is either a title or a label suffix.")
for b in unresolved: print("  DEAD: %s  %r"%b)
