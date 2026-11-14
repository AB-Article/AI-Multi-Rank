#!/usr/bin/env python3
"""Generate the deployment protocol's input files, from a KNOWN truth.

THE PROTOCOL, in three levels.  Level 0 needs ONE file and declares nothing;
Level 1 adds a bound on the price; Level 2 adds five measurements of ONE model
and the grain, and is the only level that derives a price.  All of it on the
same test set and the same device:

    measurements.csv  LEVEL 0, and the only file it needs: row 1 names the
                      candidates, one row carries the resource (its first cell
                      names the axis), every other row is one test item.
                      Accuracy and n_test are COUNTED from the item rows.
    pairs.csv         the same information as the pairwise 2x2 tables --
                      K(K-1)/2 rows instead of n_test, identical certificate,
                      and the shape to use when an item matrix cannot be
                      shipped.

    sweep.csv       one trained model, evaluated at FIVE input sizes.
                    Fixed weights -- nothing is retrained.  This is the
                    calibration curve, and it is the only thing that has to be
                    measured more than once, because kappa is a second
                    difference along ONE model's curve and three readings is
                    the floor (five gives the bias correction a 2h grid too).
    candidates.csv  every candidate, evaluated ONCE at its deployed setting.
                    Any architecture, any quantization, any provenance.
    items.csv       the per-item outcomes behind those accuracies, one column
                    per candidate.  Not a new measurement: it is what was
                    summed to get the accuracy, and it is what turns the
                    verdict into a certified set.

The truth is the cure chart of Prop. `cure`,

    eps(x) = exp{-lambda (1 - e^{-c x})},     x = ln(inference time),

with lambda = 3.930 and c = 0.4920, for which the canonical price at grain 1
is exactly c = 0.4920 nats per e-fold and eps_inf = e^{-lambda} = 0.01965
(a 98.0% accuracy ceiling).  The sweep is placed below the model's training
resolution, where accuracy rises with input size; above it accuracy falls and
the points are not Pareto at all.

Stdlib only.  Deterministic.
"""
import csv
import math
import os
import random
import statistics

LAM, C, Q0 = 3.930, 0.4920, 0.001          # 1000-class chance level
TRAIN_RES = 320                            # the model's training resolution
SWEEP_RES = (112, 143, 184, 235, 301)      # five sizes, GEOMETRIC, all <= TRAIN_RES
T_AT_112 = 6.0                             # ms on the target device
N_TEST = 50_000
W_LATENT = 0.90                            # shared item difficulty
HERE = os.path.dirname(os.path.abspath(__file__))


def time_of(res):
    """Measured inference time on the device; compute goes as pixels."""
    return T_AT_112 * (res / 112.0) ** 2


def eps_true(x):
    return math.exp(-LAM * (1.0 - math.exp(-C * x)))


def write(path, rows, header, note):
    with open(os.path.join(HERE, path), "w", newline="") as fh:
        fh.write(note)
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> {path}")


def main():
    rng = random.Random(20261008)
    nd = statistics.NormalDist()

    # ---- 1. the sweep: ONE model, five input sizes, fixed weights
    rows = []
    for res in SWEEP_RES:
        t = time_of(res)
        e_raw = eps_true(math.log(t)) * (1.0 - Q0)
        k = max(1, min(N_TEST - 1, int(round(
            rng.gauss(N_TEST * e_raw,
                      math.sqrt(N_TEST * e_raw * (1.0 - e_raw)))))))
        rows.append([f"net_b@{res}", f"{1.0 - k / N_TEST:.6f}",
                     f"{t:.2f}", res, N_TEST])
    write("sweep.csv", rows,
          ["name", "accuracy", "inference_time_ms", "input_size", "n_test"],
          f"# THE CALIBRATION SWEEP: one model (net_b, trained at"
          f" {TRAIN_RES}px), FIXED WEIGHTS,\n"
          f"# evaluated at five input sizes on the same test set and the same"
          f" device.\n"
          f"# The sizes are GEOMETRIC (ratio 1.28), because the closed forms"
          f" for the two\n"
          f"# walls assume equal steps in log-resource. Input size spans"
          f" {SWEEP_RES[-1]/SWEEP_RES[0]:.1f}x, so inference\n"
          f"# time spans"
          f" {time_of(SWEEP_RES[-1])/time_of(SWEEP_RES[0]):.1f}x, which is the"
          f" order a selection-grade price needs (F20).\n"
          f"# All sizes are at or below the training\n"
          f"# resolution: above it accuracy falls with resource and the points"
          f" are not Pareto.\n"
          f"# Generated from the cure chart (lambda={LAM}, c={C}): the"
          f" canonical price at\n"
          f"# grain 1 is exactly {C} nats per e-fold.\n")

    # ---- 2. the candidates: ONE evaluation each, any provenance
    #      off-frontier penalty delta >= 0 in nats; delta = 0 means the
    #      candidate sits on the sweep's own curve
    zoo = (("net_a_small",      4.0, 0.05),
           ("net_d_int8",       9.0, 0.02),
           ("net_c_mobile",    18.0, 0.03),
           ("net_b@224",       24.0, 0.00),
           ("net_e_hybrid",    40.0, 0.06),
           ("net_g_dominated", 30.0, 0.32),
           ("net_f_large",     90.0, 0.00))
    names = [z[0] for z in zoo]
    thr = {}
    for nm, t, delta in zoo:
        e_raw = min(eps_true(math.log(t)) * math.exp(delta), 0.9) * (1.0 - Q0)
        thr[nm] = nd.inv_cdf(1.0 - e_raw)

    # per-item outcomes on the SHARED test set: a shared latent difficulty per
    # item plus independent per-model noise, so the marginal error rates are
    # exactly the truth while failures are correlated as the archive's panel
    # measures them (phi ~ 0.5)
    sw, sw1 = math.sqrt(W_LATENT), math.sqrt(1.0 - W_LATENT)
    correct = {nm: [] for nm in names}
    rng_i = random.Random(515)
    for _ in range(N_TEST):
        z = rng_i.gauss(0.0, 1.0)
        for nm in names:
            v = sw * z + sw1 * rng_i.gauss(0.0, 1.0)
            correct[nm].append(0 if v > thr[nm] else 1)

    with open(os.path.join(HERE, "items.csv"), "w", newline="") as fh:
        fh.write("# PER-ITEM OUTCOMES on the shared test set: 1 = correct,"
                 " one column per candidate.\n"
                 "# Not a new measurement -- this is what was summed to get"
                 " the accuracies in\n"
                 "# candidates.csv. It is what makes the verdict a CERTIFIED"
                 " set (F11, F12):\n"
                 "# the pair's own 2x2 table gives both the paired variance of"
                 " the gap and the\n"
                 "# distribution-free exclusion certificate.\n")
        w = csv.writer(fh)
        w.writerow(names)
        for k in range(N_TEST):
            w.writerow([correct[nm][k] for nm in names])
    print("  -> items.csv")

    rows = []
    for nm, t, _d in zoo:
        q = sum(correct[nm]) / float(N_TEST)       # counted, not drawn
        rows.append([nm, f"{q:.6f}", f"{t:.1f}", N_TEST])
    write("candidates.csv", rows,
          ["name", "accuracy", "inference_time_ms", "n_test"],
          "# THE CANDIDATES: each evaluated ONCE, at its own deployed setting,"
          " on the same\n"
          "# test set and the same device. Architectures, quantizations and"
          " provenance may\n"
          "# all differ -- the price comes from the sweep, the score comes"
          " from the candidate\n"
          "# (F9). Accuracies are COUNTED from items.csv, so the two files"
          " cannot disagree.\n"
          "# net_g_dominated is in the list on purpose: it is both dearer and"
          " less accurate\n"
          "# than another candidate, which is the one case a certificate can"
          " settle with NO\n"
          "# price at all (F18). Real competition leaderboards always contain"
          " a few.\n")

    # ---- 3. THE ONE FILE: per-item outcomes with the resource on its own
    #      row.  Accuracy and n_test are derived from it, so nothing can
    #      disagree with anything.
    with open(os.path.join(HERE, "measurements.csv"), "w", newline="") as fh:
        fh.write("# ONE FILE, and the only one the selector needs:\n"
                 "#   row 1  the candidate names\n"
                 "#   row 2  the resource -- its first cell names the axis,"
                 " then one value per candidate\n"
                 "#   rest   one test item each, 1 = correct\n"
                 "# Accuracy and n_test are COUNTED from the item rows, so"
                 " there is nothing to keep in step.\n"
                 "#   select_model.py measurements.csv\n")
        w = csv.writer(fh)
        w.writerow(names)
        w.writerow(["inference_time_ms"] + [f"{t:g}" for _nm, t, _d in zoo])
        for k in range(N_TEST):
            w.writerow([correct[nm][k] for nm in names])
    print("  -> measurements.csv")

    # ---- 4. the same information as pairwise 2x2 tables, for deployments
    #      that will not ship an item matrix: K(K-1)/2 rows, identical
    #      certificate
    prows = []
    for ai in range(len(names)):
        for bj in range(ai + 1, len(names)):
            ni, nj = names[ai], names[bj]
            a_ = b_ = c_ = d_ = 0
            for k in range(N_TEST):
                fi = not correct[ni][k]
                fj = not correct[nj][k]
                if fi and fj:
                    a_ += 1
                elif fi:
                    b_ += 1
                elif fj:
                    c_ += 1
                else:
                    d_ += 1
            prows.append([ni, nj, a_, b_, c_, d_])
    write("pairs.csv", prows,
          ["i", "j", "both_fail", "i_only", "j_only", "neither"],
          "# The pairwise 2x2 tables -- all the certificate ever reads.\n"
          "# K(K-1)/2 rows instead of n_test, and the result is identical:\n"
          "#   select_model.py candidates.csv --pairs pairs.csv\n")

    print()
    print(f"  truth: canonical price {C} nats/e-fold at grain 1,"
          f" eps_inf = {math.exp(-LAM):.5f}"
          f" (accuracy ceiling {1-math.exp(-LAM)*(1-Q0):.4f})")
    print(f"  the sweep spans {time_of(SWEEP_RES[-1])/time_of(SWEEP_RES[0]):.1f}x"
          f" in inference time over {len(SWEEP_RES)} points")


if __name__ == "__main__":
    main()
