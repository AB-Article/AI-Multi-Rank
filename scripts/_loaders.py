"""Shared loaders for scripts 22-29 (spectral extension). Reuses 09 and 11."""
import os, sys, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)

def _module(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def selection_rules():
    """Script 11 as a module (its __main__ block is not executed)."""
    return _module("11_selection_rules.py", "sr")

def factorial():
    """Fitting functions of script 14 (power-mean law), without running it."""
    src = open(os.path.join(HERE, "14_test_N_equals_n.py")).read()
    ns = {"__name__": "loader", "__file__": os.path.join(HERE, "14_test_N_equals_n.py")}
    exec(src.split("# ----------------------------------------------------------------- run")[0], ns)
    return ns

def all_fields():
    """Frontier generator and channel fits of script 09, without running it."""
    src = open(os.path.join(HERE, "09_all_fields.py")).read()
    ns = {"__name__": "loader", "__file__": os.path.join(HERE, "09_all_fields.py")}
    exec(src.split("rows_f, rows_law, rows_N")[0], ns)
    return ns

def save(df, name):
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, name), index=False)
    print(f"  -> results/{name}")


def all_frontiers():
    """The 117-frontier builder of script 68, as a function (its __main__ is not run)."""
    src = open(os.path.join(HERE, "68_prediction_assessment.py")).read()
    ns = {"__name__": "loader", "__file__": os.path.join(HERE, "68_prediction_assessment.py")}
    exec(src.split("# ---------------------------------------------------------------- predictors")[0], ns)
    return ns["build"]()
