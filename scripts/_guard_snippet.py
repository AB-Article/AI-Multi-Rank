#!/usr/bin/env python3
"""The README's ten-line rule must be exactly the code that is tested.

README.md prints the decision rule as a Python block, and
examples/operating_point.py carries the same block plus a self-test that runs it
on every frontier in the archive.  A README snippet that nobody executes is the
one piece of an archive that rots silently, so this guard asserts the two are
character-identical and then runs the self-test.

Prints CLEAN or the offenders.  Takes no arguments.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
README = os.path.join(ROOT, "README.md")
EXAMPLE = os.path.join(ROOT, "examples", "operating_point.py")

fail = []

if not os.path.exists(README):
    fail.append("README.md not found")
if not os.path.exists(EXAMPLE):
    fail.append("examples/operating_point.py not found")

if not fail:
    blocks = re.findall(r"```python\n(.*?)```", open(README, encoding="utf-8").read(), re.S)
    if not blocks:
        fail.append("README.md has no ```python block")
    else:
        snippet = blocks[0].strip()
        src = open(EXAMPLE, encoding="utf-8").read()
        if snippet not in src:
            fail.append("the README snippet is not present verbatim in "
                        "examples/operating_point.py")
        if "def operating_point" not in snippet:
            fail.append("the README's first python block does not define operating_point")

if not fail:
    r = subprocess.run([sys.executable, "-I", EXAMPLE],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        fail.append("examples/operating_point.py self-test failed:\n"
                    + (r.stderr or r.stdout).strip()[-600:])
    else:
        last = [l for l in r.stdout.splitlines() if "declined" in l]
        if last and not last[0].rstrip().endswith("0"):
            fail.append("the rule declined on at least one archived frontier: "
                        + last[0].strip())

if fail:
    print("SNIPPET GUARD FOUND %d:" % len(fail))
    for f in fail:
        print("  - " + f)
    sys.exit(1)
print("SNIPPET CLEAN: the README's rule is the tested code, and it reads every "
      "archived frontier.")
