#!/usr/bin/env python3
"""One index of every numerical-verification result file in the archive.

Why this script exists
----------------------
The technical report carried five hand-maintained tables reproducing, in LaTeX,
the per-identity verification records that the archive already holds as CSVs --
and said so itself: "the CSV files in the archive are authoritative".  Two
copies of one record is how the 2026-10-09 audit's recurring defect arises: the
typeset copy goes stale and nothing catches it, because LaTeX compiles whatever
the table says.

This script replaces those five tables with one generated index.  For every
verification result file it reports the family, the script that writes it, the
number of checks, and the worst residual the file records, discovering the
residual column by name rather than by position so that a schema change is
visible rather than silent.

Writes results/table_verification_index.csv
       results/table_verification_index.tex   (input by the technical report)
"""
import os, re, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

# columns that, where present, carry a residual / deviation / error
RESID = ("max_deviation", "max_dev", "abs_error", "max_abs_error", "residual",
         "max_residual", "err", "max_err", "deviation", "max_dev_log_ratio",
         "worst", "max_rel_error")
# columns that name the statement a row checks
NAMES = ("result", "statement", "identity", "test", "block", "check", "name",
         "item", "theorem", "case")
# a row whose own text says a LARGE value is the intended outcome: a strict
# inequality, a non-invariance margin, a separation.  Its "residual" is a
# margin, not an error, and averaging the two would misreport both.
DIRECTIONAL = ("must be", "not invariant", "min |", "max change", "separation",
               "lower bound", "non-square")


def family(fn):
    """A readable family name from the file name."""
    t = fn.replace(".csv", "")
    t = re.sub(r"^table_", "", t)
    t = t.replace("_verification", "").replace("_identities", "")
    t = t.replace("_checks", "").replace("_archive", "")
    return t.replace("_", " ").strip() or t


def writer_of(fn):
    """The numbered script whose source mentions this file name."""
    for p in sorted(glob.glob(os.path.join(HERE, "[0-9]*.py"))):
        try:
            if fn in open(p, encoding="utf-8", errors="replace").read():
                return os.path.basename(p).split("_")[0]
        except Exception:
            continue
    return ""


def is_directional(df):
    """Per-row mask: True where the row's own text says a large value passes."""
    txt = pd.Series([""] * len(df), index=df.index)
    for c in df.columns:
        if c.lower() in NAMES:
            txt = txt + " " + df[c].astype(str).str.lower()
    return txt.apply(lambda t: any(k in t for k in DIRECTIONAL))


def worst(df):
    """The largest residual over the rows that assert an IDENTITY.

    Rows whose own text says a large value is the intended outcome -- "must be
    > 0", a non-invariance margin, a separation -- are excluded and counted
    separately: their number is a margin, not an error.  (Round 18: the first
    version of this index mixed the two and reported a 4.4 "residual" for a row
    whose text reads "alpha NOT invariant (max change, must be > 0)".)
    """
    dirmask = is_directional(df)
    sub = df[~dirmask]
    best, col = None, ""
    for c in df.columns:
        if c.lower() in RESID or any(k in c.lower() for k in ("deviation", "abs_error", "residual")):
            v = pd.to_numeric(sub[c], errors="coerce").abs().max()
            if pd.notna(v) and (best is None or v > best):
                best, col = float(v), c
    return best, col, int(dirmask.sum())


def failures(df):
    """(rows flagged as failing, rows that are reported but not checked).

    The flag is written as a bool, as 0/1 and as 1.0 in different scripts, so
    it is read numerically first and textually second.  A BLANK flag means the
    row reports a quantity with no target -- a radius, a root count -- and is
    not a check at all; counting those as failures is how the first version of
    this index reported "21 failures" in a family that has none.
    """
    for c in df.columns:
        if c.lower() in ("ok", "passed", "pass"):
            num = pd.to_numeric(df[c], errors="coerce")
            blank = int(num.isna().sum())
            if num.notna().any():
                return int((num.fillna(1) == 0).sum()), blank
            s = df[c].astype(str).str.strip().str.lower()
            return int((~s.isin(("true", "1", "1.0", "yes", "ok", "nan"))).sum()), blank
    return 0, 0


def main():
    files = sorted(set(
        glob.glob(os.path.join(RES, "*verification*.csv"))
        + glob.glob(os.path.join(RES, "*identities*.csv"))
        + glob.glob(os.path.join(RES, "*_checks.csv"))))
    files = [p for p in files
             if os.path.basename(p) != "table_verification_index.csv"]
    rows = []
    for p in files:
        fn = os.path.basename(p)
        try:
            df = pd.read_csv(p)
        except Exception as e:
            rows.append(dict(family=family(fn), file=fn, script="", checks=0,
                             identity_checks=0, directional_checks=0, reported_only=0,
                             worst_residual=float("nan"), residual_column="",
                             flagged=0, note=f"unreadable: {e}"))
            continue
        w, col, ndir = worst(df)
        nfail, nblank = failures(df)
        rows.append(dict(family=family(fn), file=fn, script=writer_of(fn),
                         checks=len(df) - nblank,
                         identity_checks=len(df) - ndir - nblank,
                         directional_checks=ndir, reported_only=nblank,
                         worst_residual=w if w is not None else float("nan"),
                         residual_column=col, flagged=nfail,
                         note="" if col else "no residual column"))
    T = pd.DataFrame(rows).sort_values("family").reset_index(drop=True)
    T.to_csv(os.path.join(RES, "table_verification_index.csv"), index=False)
    print(f"  -> results/table_verification_index.csv  ({len(T)} families)")

    # the LaTeX the report inputs, generated so it cannot go stale
    def fmt(v):
        if pd.isna(v):
            return "---"
        return f"${v:.1e}$".replace("e-0", "e{-}").replace("e+0", "e{+}").replace("e-", "e{-}")
    out = ["% GENERATED by scripts/162_verification_index.py -- do not edit by hand.",
           r"\begin{tabular}{llrrl}", r"\toprule",
           r"family & script & checks & worst residual & result file\\", r"\midrule"]
    for _, r in T.iterrows():
        f = r["file"].replace("_", r"\_")
        out.append(f"{r['family']} & \\texttt{{{r['script']}}} & {r['checks']} & "
                   f"{fmt(r['worst_residual'])} & \\texttt{{{f}}}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(RES, "table_verification_index.tex"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"  -> results/table_verification_index.tex")

    print("=" * 78)
    print("NUMERICAL VERIFICATION: ONE INDEX")
    print()
    print(f"  {'family':30s} {'script':>6s} {'checks':>7s} {'ident':>6s} {'dir':>4s} "
          f"{'worst residual':>15s} {'flagged':>8s}")
    for _, r in T.iterrows():
        w = "---" if pd.isna(r["worst_residual"]) else f"{r['worst_residual']:.2e}"
        print(f"  {r['family'][:30]:30s} {r['script']:>6s} {r['checks']:>7d} "
              f"{r['identity_checks']:>6d} {r['directional_checks']:>4d} {w:>15s} {r['flagged']:>8d}")
    print()
    print(f"  {len(T)} families, {int(T.checks.sum())} checks "
          f"({int(T.identity_checks.sum())} identities, "
          f"{int(T.directional_checks.sum())} directional), "
          f"{int(T.flagged.sum())} rows flagged by the files themselves.")
    print("  The CSVs are authoritative; this index and its .tex are generated from them.")


if __name__ == "__main__":
    main()
