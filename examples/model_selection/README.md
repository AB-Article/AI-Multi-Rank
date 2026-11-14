# Picking and ranking networks from their own measurements

This is the practical form of the theory. It answers

> *given a pile of trained networks, which one should I ship, and how do the
> rest rank?*

**from the candidates' own measurements and nothing else.** No price is set by
hand, no grain, no chance level, and in the minimal mode not even a calibration
sweep.

## Level 0 — the whole answer, from ONE file

```bash
python3 make_protocol_data.py          # regenerate the demo inputs
python3 select_model.py measurements.csv
```

**One file.** Its first row names the candidates, exactly one further row has a
non-numeric first cell and is the **resource** row, and every other row is one
test item:

```
net_a_small,net_d_int8,net_c_mobile,net_b@224,net_e_hybrid,net_g_dominated,net_f_large
inference_time_ms,4,9,18,24,40,30,90
0,1,1,1,1,1,1
1,1,1,1,1,1,1
...
```

That is all. **The accuracy of each candidate is counted from the item rows and
`n_test` is their number**, so there is no second file to keep in step and
nothing that can disagree with anything.

### Or with no file at all

```bash
python3 select_model.py --axis inference_time_ms \
    --model net_a_small,0.851000,4,50000 \
    --model net_d_int8,0.925760,9,50000 \
    --model net_b@224,0.955320,24,50000 \
    --model net_f_large,0.969800,90,50000 \
    --pair net_a_small,net_d_int8,3326,4124,386,42164 \
    --pair net_d_int8,net_b@224,1775,1937,459,45829 \
    --pair net_b@224,net_f_large,1108,1126,402,47364 \
    --pair net_a_small,net_b@224,2118,5332,116,42434 \
    --pair net_a_small,net_f_large,1463,5987,47,42503 \
    --pair net_d_int8,net_f_large,1318,2394,192,46096
```

One `--model` per candidate (`name,accuracy,resource[,n_test]`), and one
`--pair` per pair (`i,j,both_fail,i_only,j_only,neither`) — the same six numbers
as a row of `pairs.csv`, and all the certificate ever reads. Drop the `--pair`
arguments and you get the conservative unpaired certificate instead; the tool
says which it used. The full form, with both options worked through, is under
[Inputs](#inputs); `scripts/157_input_routes.py` checks that they agree.

### The input shapes

| what you have | how to pass it | certificate |
|---|---|---|
| per-item outcomes + the resource, in one file | `select_model.py measurements.csv` | **exact**, distribution-free, finite-sample |
| aggregates, plus the pairwise `2x2` tables | `select_model.py candidates.csv --pairs pairs.csv` | **exact** — identical numbers, and it is `K(K-1)/2` rows instead of `n_test`, so a 50,000-item evaluation of seven candidates is **21 lines** |
| aggregates only (`accuracy`, resource, `n_test`) | `select_model.py candidates.csv` | normal, and **unpaired**: against the *paired normal* band it is conservative by `1.36x` overall and `2.1x` between adjacent models, where it also asks `4.4x` the items -- but against the *exact* certificate of the rows above it is not uniformly conservative (on this demo it is slightly narrower), so it is a convenience, not a safe fallback. Add `--rho 0.5` for the paired normal one |

The second shape exists because the certificate never reads the item matrix —
only each pair's `2x2` table — so a deployment that will not ship 50,000 rows
can ship 21 and lose nothing. The archive's `pairs.csv` and `items.csv` are
generated from the same draw, and the ladders they produce agree to every digit
printed. The two-file form `candidates.csv --items items.csv` also still works
and is equivalent to the first shape.

```

MODE ........... measurement-only: 7 candidates, one evaluation each, resource 'inference_time_ms'
  inputs used .. accuracy, inference_time_ms, n_test, per-item outcomes   [exact-paired]
  NOT used ..... no price, no calibration sweep, no grain, no chance level q0 -- none of them
                 enters the envelope, the ladder or the price-free test, and q0 cancels
                 identically from every difference of sigma.

ENVELOPE ....... who can be optimal at ALL -- 4 of 7 candidates:
  net_f_large      optimal for price in [0.0000, 0.2963] nats/e-fold of inference_time_ms
  net_b@224        optimal for price in [0.2963, 0.5177] nats/e-fold of inference_time_ms
  net_d_int8       optimal for price in [0.5177, 0.8591] nats/e-fold of inference_time_ms
  net_a_small      optimal for price in [0.8591, infinity] nats/e-fold of inference_time_ms
  never optimal at any price: net_c_mobile, net_e_hybrid, net_g_dominated
    dominated .... net_g_dominated
    in a dent .... net_c_mobile, net_e_hybrid  (Pareto-efficient, and still never selected)

BREAK-EVEN ..... the ladder, one rung per envelope breakpoint:
  cheaper          dearer           break-even     certified interval   undecided
  net_b@224        net_f_large          0.2963       [0.2516, 0.3423]    0.44 dbl
  net_d_int8       net_b@224            0.5177       [0.4708, 0.5659]    0.27 dbl
  net_a_small      net_d_int8           0.8591       [0.8176, 0.9017]    0.14 dbl
  read it as: pay for the dearer model while your price per e-fold is below the rung.

PRICE-FREE ..... 2 domination(s) certified on accuracy alone, valid at EVERY price:
  net_c_mobile beats net_g_dominated
  net_b@224 beats net_g_dominated

RANK ........... tier from the price-free order; inside a tier, by where each is optimal
   # name                   eps inference_ti  tier    optimal for price in
   1 net_f_large        0.03020           90     1          [0.000, 0.296]
   2 net_b@224          0.04468           24     1          [0.296, 0.518]
   3 net_d_int8         0.07424            9     1          [0.518, 0.859]
   4 net_a_small        0.14900            4     1            [0.859, inf]
   5 net_e_hybrid       0.03912           40     1           never optimal
   6 net_c_mobile       0.05200           18     1           never optimal
   7 net_g_dominated    0.05690           30     2           never optimal
  no sigma column, because sigma needs a price. The envelope column is the ranking by
  trade-off, and it is complete: every price is covered by exactly one candidate.

WHAT MORE WOULD BUY ...
  a price interval (--price-interval LO:HI) -> the certified SET at that interval, and tiers
                                               that use it
  a 5-point sweep (--calibration + --grain)  -> the price itself, lambda* = sqrt(kappa/t),
                                               plus N_eff, the walls and the floor
  a count axis (--passk)                     -> the floor as two measured numbers
```

Those three lines name inputs this run did **not** read. A Level 0 run reads
the accuracies, the resource column and the item counts and nothing else; if
you pass `--passk`, `--floor`, `--grain`, `--retry-budget`, `--attempt-budget`,
`--theta`, `--resources`, `--q0`, `--tie-band` or `--no-bias-correction` to a
Level 0 invocation, the report lists them back under **`NOT READ AT THIS
LEVEL`** instead of dropping them (F17).

Why this needs no price, no grain and no `q0`:

* the **envelope** is `max_i (S_i - lambda x_i)` read as a *function* of
  `lambda`, so it is computed without knowing `lambda`;
* its **breakpoints** are the break-evens, and the rung at `lambda = 0` is the
  **price-free** test;
* `q0` enters as `ln eps = ln(1-Q) - ln(1-q0)`, a constant shift of `l`, so it
  cancels identically from every difference of `sigma` and from every rung.
  Checked on all 117 archived frontiers over `q0 in {0, 0.001, 0.01, 0.1, 0.5}`:
  the largest deviation in any `sigma` difference is `1.8e-15`, in any
  break-even `5.0e-14`, and **no** envelope winner set changes
  (`scripts/156_measurement_only.py`);
* the **grain** enters only through `lambda* = sqrt(kappa/t)`, the operating
  point and the wall and depth bounds — none of which the decision uses.

Measured on all 117 archived frontiers in exactly this mode: a median of **11**
candidates, of which **5** can ever be optimal — **62.5% retired outright, with
a proof** — and a ladder of **4** rungs that tiles the whole price axis. On the
36 of them with a public item count, the median undecided interval is **0.43**
doublings and **100%** of rungs are decided on at least one side.

## Level 1 — if you can bound your own price

```bash
python3 select_model.py measurements.csv --price-interval 0.45:0.55
```

Still one file, still no grain and no price set by hand — only a *bound*. You
get the **certified set** over that interval: the candidates that cannot be
certifiably excluded, covering the price-optimal one with probability at least
`1 - alpha`, plus certified **tiers** that use the interval. On the demo that
takes 7 candidates to 3.

## Level 2 — if you will name no price at all

Then the theory supplies one. Add a five-point sweep of **one** model and the
grain, and `lambda* = sqrt(kappa/t)` is derived:

```bash
python3 select_model.py candidates.csv --grain 1 \
    --calibration sweep.csv --calibration-axis inference_time_ms \
    --items items.csv
```

`sweep.csv` is that one model, **fixed weights**, at five input sizes — nothing
retrained, one inference pass each, geometrically spaced, at or below the
training resolution, recording the resource you will pay for. This is the only
level that needs the grain, and the only one that also returns `N_eff`, the
walls and the floor.

## Which output answers which question

Two scenarios, and the tool answers both from the same inputs.

### A — pick ONE network

Stop at the first row that answers you.

| | what to read | extra inputs |
|---|---|---|
| 1 | **`PRICE-FREE`** — no dearer on any axis and certified no worse ⇒ wins at every price | **none** |
| 2 | **`ENVELOPE`** — your price falls in exactly one piece, and that piece names the winner; **`BREAK-EVEN`** certifies the boundary | **none** |
| 3 | **`CERTIFIED SET`** at a price interval you can bound | `--price-interval LO:HI` |
| 4 | **`CERTIFIED SET`** at the *derived* price, when you will name none | the five-point sweep + `--grain` |

Rows 1 and 2 are Level 0: nothing beyond the candidates' own evaluations. Row 4
is the fully automatic path, and **`PRICE SHARPNESS`** then tells you whether
the sweep is sharp enough to name a single winner, with the factor to widen it
if not.

### B — RANK all networks by the trade-off

Read the **`RANK`** table. It carries the two honest layers, and nothing else is
a ranking:

* **`tier`** — certified dominance. At Level 0 this is the *price-free* partial
  order; with a price interval it is dominance over that interval. Within a tier
  no difference is separable at the items you have; between tiers every
  difference is. Any `sigma` order *inside* a tier is presentation, not
  evidence.
* **`optimal for price in [...]`** — the envelope: the price range over which
  each candidate is the best choice, independent of any price you declare. This
  is the trade-off ranking proper, and `never optimal` is a **proof** that no
  price selects that candidate, not a statement about noise.

`RANK` therefore doubles as the deployment schedule: read down the envelope
column and you have which network to serve at which price.

## Why the sweep is five input sizes and not five trained models

The price is not a parameter. At the operating point `N_eff = 1/t`,

```
    lambda* = s / sqrt(t N_eff) = sqrt(kappa / t)
```

so once the grain `t` is declared the price is **determined** — but `kappa` is a
curvature, and a curvature needs one curve. Five *different* networks are a
Pareto envelope, not a curve, so they cannot give you `kappa` at all. You need
one model at several budgets.

Input size is the cheapest way to get that curve: fixed weights, one inference
pass per size, and because compute goes as pixels, a 2.7× range of resolution
is already a 7× range of resource. Two rules the geometry imposes:

1. **Stay at or below the training resolution.** Above it, accuracy *falls*
   with resource; those points are not Pareto and no frontier passes through
   them.
2. **Place the sizes geometrically, and record the resource you will pay for** —
   measured inference time on the target device, not the pixel count. Pixels→time
   differs per architecture, and the calibration axis must be the axis you score
   the candidates on.

`scripts/153_input_size_axis.py` checks the premise on the archive's own
input-pixels frontiers, where the class gate separates the two ways of
producing such a sweep: **retrained at each size, out of class on 3 of 3;
fixed backbone with only the input varying, in class or envelope on 3 of 3.**
Retraining changes the mechanism between points, so it is not one clock. A
fixed model fed different sizes is much closer to one — and it is the variant a
deployment would actually run.

## The demo, and the known truth behind it

`make_protocol_data.py` generates all five files from the cure chart of
Prop. `cure`, `eps(x) = exp{-lambda(1 - e^{-cx})}` with `lambda = 3.930` and
`c = 0.4920`, so

* the canonical price at grain 1 is **exactly `0.4920` nats per e-fold**,
* the accuracy ceiling is `1 - e^{-lambda}(1-q0) = 0.9804`,
* the sweep is `net_b` (trained at 320px) at 112/143/184/235/301 px — geometric,
  spanning `7.2×` in inference time,
* the seven candidates are a heterogeneous zoo at their own deployed settings,
  one of which (`net_g_dominated`) is deliberately both dearer and worse, the way
  a real leaderboard always contains a few,
* `items.csv` is 50,000 per-item outcomes with a shared latent item difficulty,
  so failures are correlated as the archive's own panel measures them (realised
  `phi = 0.59` against `0.50` on the panel), and the accuracies in
  `candidates.csv` are **counted** from it.

The derived price comes out at `0.4677` against the truth `0.4920` — `-5%`, in
the direction the theory predicts (the plug-in slope reads low, so the price
reads low). That `-5%` is the whole of the manual-pricing problem, solved.

## The output

```

grain .......... t = 1  [declared, not defaulted]
class gate ..... IN_CLASS: ln eps convex on the global chord (defect +0.24470) and on 100% of 4 wide triples
  divisibility . not refuted; curvature rises on 0 step(s) (a rise refutes; no rise proves nothing)
precision gate . N_eff = 1.249 [0.508, 10.184] at 90% from n_test alone; 18.3% of draws out of class
under-sizing ... no count axis supplied: the compute-axis target 1/t is a lower bound on the task's own, so this reading under-sizes
plane .......... 3 readings; median N_eff 1.510; implied grain [0.58, 2.90]
operate at ..... resource 18.25 (N_eff = 1.0000 solved on the plane's own line, R^2 = 0.821; inside the measured window)
  nearest ...... the closest measured point costs 0.0211 nats against the rule
walls .......... eps_inf <= 0.021539 (axiom), <= 0.011944 (divisible) at the operating reading
arc ............ 0.164 pi measured; 1.00 checkpoints' worth of pricing information (five cover a complete frontier)
price .......... inference_time_ms = 0.4677 nats/e-fold [derived]
  inference_time_ms ...... 90% [0.4012, 0.7237] from test-set noise alone (sd 0.0981); the plug-in slope reads low, so this price is biased LOW and the operating depth biased DEEP
calibration ..... grid step h = 0.494 in log-resource; N_eff needs n_test >= 891,673 items at this h -- or keep 50,000 items and widen the grid by 4.2x, which is quadratically cheaper
depth left ..... >= 0.6931 nats (axiom), >= 1.0000 (divisible); gap 0.3069
envelope test .. threshold 0.250 on the share of jagged wide triples [calibrated on this frontier's own convex minorant and noise]; measured 0.000
design rule .... this grid h = 0.494; a PRICE needs h >= 0.585 [INFEASIBLE], a channel count h >= 0.827 [INFEASIBLE]
  widen by ..... x1.18 in log-resource, or buy 391,142 items at this grid (the widening is quadratically cheaper)
paired (F11) ... 50,000 per-item outcomes for 7 candidates; median failure correlation 0.589, so every band below is the PAIRED one

CERTIFIED SET .. {net_b@224, net_c_mobile, net_d_int8}  at 90% coverage, over inference_time_ms in [0.4012, 0.7237] nats/e-fold (0.59 e-folds of price uncertainty)
  method ....... exact-paired at alpha/(K-1) = 0.0167 per comparison
  assumes ...... independent test items only: no distributional form, no asymptotics, no Law I, no N_eff, no floor
  pick ......... net_d_int8  [minimax inside the certified set]
  excluded ..... net_a_small, net_e_hybrid, net_f_large, net_g_dominated  (each beaten at every price in the interval by one rival, certified)
  the set can still cost 0.0000 nats against the best candidate, over that interval

PRICE-FREE ..... 2 domination(s) certified on accuracy alone, valid at EVERY price:
  net_c_mobile beats net_g_dominated  (no dearer on any axis, and certified no worse)
  net_b@224 beats net_g_dominated  (no dearer on any axis, and certified no worse)
  survivors .... net_a_small, net_d_int8, net_c_mobile, net_b@224, net_e_hybrid, net_f_large   [needs no price, no grain, no sweep]

ENVELOPE ....... who can be optimal at ALL, on inference_time_ms -- 4 of 7 candidates:
  net_f_large      optimal for price in [0.0000, 0.2963] nats/e-fold
  net_b@224        optimal for price in [0.2963, 0.5177] nats/e-fold
  net_d_int8       optimal for price in [0.5177, 0.8591] nats/e-fold
  net_a_small      optimal for price in [0.8591, infinity] nats/e-fold
  never optimal at any price: net_c_mobile, net_e_hybrid, net_g_dominated
    dominated .... net_g_dominated  (something is both cheaper and more accurate)
    in a dent .... net_c_mobile, net_e_hybrid  (Pareto-efficient, but NO linear price selects it)
  this is read off K straight lines sigma_i = S_i - lambda x_i; no frontier has to be
  identified first, and nothing here assumes Law I.

BREAK-EVEN ..... the ladder, rung per envelope breakpoint [exact-paired]:
  cheaper          dearer           break-even     certified interval   undecided
  net_b@224        net_f_large          0.2963       [0.2516, 0.3423]    0.44 dbl
  net_d_int8       net_b@224            0.5177       [0.4708, 0.5659]    0.27 dbl
  net_a_small      net_d_int8           0.8591       [0.8176, 0.9017]    0.14 dbl
  below the lower end the dearer model is certified better, above the upper end the
  cheaper one is, and in between the items in hand cannot separate them. No sweep,
  no grain and no derived price are used anywhere in this block.

RANKING ........ 3 certified tier(s) (exact-paired, alpha/(K-1) = 0.0167):
  tier 1 ....... net_b@224, net_c_mobile, net_d_int8   (worst-case regret 0.1143-0.2021 nats)
  tier 2 ....... net_e_hybrid, net_f_large, net_a_small   (worst-case regret 0.4389-0.7670 nats)
  tier 3 ....... net_g_dominated   (worst-case regret 0.6054-0.6054 nats)
  within a tier no difference is certifiable; between tiers every difference is.
  the tiers order certified DOMINANCE, not regret: a lower tier can carry a
  smaller worst case and still be certifiably beaten by something above it.

PRICE SHARPNESS  INFEASIBLE: the derived price spans 0.85 doublings; a candidate reigns for 0.48
  remedy ....... widen the sweep by x1.3 in log-resource (h = 0.657, a span of 14x over five points), or buy 3x more items

SCHEDULE ....... 2 network(s) serve the whole price interval: net_b@224, net_d_int8
  one instead .. net_d_int8, costing 0.1143 nats worst case over the interval

CANDIDATES ..... worst case still available to an UNSEEN network: 0.6588 nats  [concave envelope (Law I)]
  verdict ...... KEEP GOING at rho_2 = 0.0589
  next one at .. log-resource 4.4998 (resource 90) -- where the worst case is attained

BAND (v2) ...... TIE; tie set {net_b@224, net_c_mobile, net_d_int8, net_e_hybrid, net_f_large} at band 0.2620 nats (widened from 0.0589 by test-set noise)
  pick ......... net_c_mobile  [tie broken on worst-case regret over the supported price range]
  note ......... an indifference band is not a confidence set. Where the two disagree, the certified set is the one with a number attached.
  minimax ...... net_c_mobile minimises worst-case regret over the whole price range the frontier supports
  to resolve ... net_c_mobile: 200,555 items, net_d_int8: 23,038 items, net_e_hybrid: 6,150 items, net_f_large: 1,697 items  [paired]

RANK ........... tier first (what the data can separate), then sigma at the price in force
   # name                   eps inference_ti    sigma  tier    optimal for price in  cert
   1 net_b@224          0.04468           24   1.6220     1          [0.296, 0.518]  *
   2 net_c_mobile       0.05200           18   1.6048     1           never optimal  *
   3 net_d_int8         0.07424            9   1.5729     1          [0.518, 0.859]  *
   4 net_e_hybrid       0.03912           40   1.5160     2           never optimal  
   5 net_f_large        0.03020           90   1.3955     2          [0.000, 0.296]  
   6 net_a_small        0.14900            4   1.2555     2            [0.859, inf]  
   7 net_g_dominated    0.05690           30   1.2758     3           never optimal  
  'cert' marks the certified set: the candidates that cannot be certifiably excluded
  at the price interval in force. 'never optimal' is proof, not noise -- no price selects it.

INPUTS ......... strongest statement these inputs support: CERTIFIED
  q0                   NOT SUPPLIED   not an input of any decision: a constant shift of l that cancels from every difference of sigma (verified to 1e-14, F23). q0 = 0 is assumed for the floor and depth readings, which are the only things that use it
  per-item outcomes    PRESENT        the gap is paired and the exclusion certificate is distribution-free and finite-sample
! N_eff spans more than one doubling at this n_test: the operating point is not resolved by the data. N_eff is a square over a second difference and amplifies noise (corpus median straightness R^2 = 0.22, no power against a design-matched null)
```

## How to read it, line by line

The table at the bottom, **`RANK`**, is the single object that answers "rank them
all": tier first (what the data can separate), then `sigma` at the price in
force, with each candidate's envelope interval beside it. The blocks above it
are how that table was arrived at, in the order they matter:

0. **`ENVELOPE`** — which candidates a linear price can **ever** select, read off
   K straight lines `sigma_i = S_i - lambda x_i`. Four of seven here; the other
   three are never the answer at any price, one because it is dominated and two
   because they sit in *dents* — Pareto-efficient and still never selected. **No
   frontier has to be identified, and nothing here assumes Law I.** On the
   archive's already-Pareto-filtered frontiers a median **62.5%** of points fall
   into this category, so "is it on the frontier?" is the wrong question.
0b. **`BREAK-EVEN`** — the price ladder. Each rung is the price at which the next
   model up stops paying for itself, with a certified interval around it: below
   the interval the dearer model is certified better, above it the cheaper one
   is, inside it the items cannot separate them. **This block uses no sweep, no
   grain and no derived price** — and measured on 1,309 adjacent Pareto pairs of
   the archive, its rungs are *sharper* than the derived price on **96%** of
   pairs (median undecided interval `0.87` doublings against `4.86`). If you can
   say anything at all about your own price, read the ladder and stop.
1. **`PRICE-FREE`** — the strongest statement available, and it needs no price,
   no grain and no sweep. `net_g_dominated` is dearer than `net_c_mobile` *and*
   certified no better on accuracy, so it loses at **every** non-negative price.
   This is the first test to run, and on a raw leaderboard it is often the only
   one you need.
2. **`RANKING`** — three certified tiers **over the stated price interval**.
   Within a tier no difference is
   certifiable at 50,000 items; between tiers every difference is. This is the
   honest form of "rank them all": a total order would invent distinctions the
   data cannot support, and Thm. `covering` says how many — 13.8 consecutive
   models per band on a median frontier of the archive's vision panel, whose
   frontiers are its longest. **Level 1, and not price-free**: tier membership
   moves with the interval, and a bare Level-0 run prints no `RANKING` block at
   all — what it gives of the ranking is the price-free partial order in the
   `tier` column of `RANK`.
3. **`CERTIFIED SET`** — `{net_b@224, net_c_mobile, net_d_int8}` carries
   `Pr[the model optimal at the realised price is in this set] >= 90%`, from a
   one-sided distribution-free finite-sample bound on a bounded per-item
   variable, tested at `alpha/(K-1)` per comparison. Four of seven candidates are
   **certifiably** out: that is the shortlist you can defend. The `pick` inside
   it is the minimax member. **Level 1**, like the ranking above it: both consume
   the price interval and a Level-0 run prints neither. Only the `PRICE-FREE`
   block is price-free, and the order here is by how much of the price axis each
   statement commits to.

**The three diagnostics that tell you what to do next:**

4. **`PRICE SHARPNESS`** — the derived price's interval spans `0.85` doublings
   while a candidate reigns for `0.48`, so the price is a factor `1.8` too blunt
   to name a single winner. The remedy is quantitative: a `1.3×` wider sweep (a
   `14×` resource span over five points), or `3×` more items. The sweep is the
   cheap route, because `sd(lambda)/lambda` goes as `sd(l)/h²` — a factor `F` of
   sharpening is `sqrt(F)` in sweep width but `F²` in items.
5. **`SCHEDULE`** — if you would rather not sharpen the price, serve the
   interval: two networks cover it, and one compromise network costs `0.114`
   nats worst case.
6. **`CANDIDATES`** — the concave upper envelope of Law I bounds what an
   **unseen** network could still be worth, and names the resource where it would
   sit. `KEEP GOING` here, because the gap above `net_f_large` is unexplored.

**`BAND (v2)`** is the previous revision's output, kept for comparison. It calls
five of seven a tie. The certified set calls three, with a number attached.

## Running the pieces separately

```bash
# Level 0, one file: the envelope, the ladder, the price-free test, the ranking
python3 select_model.py measurements.csv

# the same, from the pairwise tables instead of the item matrix (21 lines)
python3 select_model.py candidates.csv --pairs pairs.csv

# aggregates only: the rungs fall back to a normal unpaired interval and say so
python3 select_model.py candidates.csv

# Level 1: bound the price yourself instead of deriving it -- no grain needed
python3 select_model.py measurements.csv --price-interval 0.45:0.55

# Level 2: derive it from the sweep (the only level that needs --grain)
python3 select_model.py candidates.csv --grain 1 \
    --calibration sweep.csv --calibration-axis inference_time_ms \
    --items items.csv

# several resource columns and no price: name the axis to read the envelope on
python3 select_model.py candidates.csv --items items.csv --axis inference_time_ms
```

## Which axes can serve as the sweep

The sweep must be **one mechanism**, must **move accuracy**, and must be measured
in the **resource you will pay for**. Most candidate axes fail one of the three.

| axis | verdict |
|---|---|
| **k crops / TTA samples** | best footing — a count axis, where the axiom is a *theorem* |
| **input size / resolution** | best for vision — no training, `2.7x` of resolution is `7x` of resource |
| **sequence length / context** | the same, for text |
| **denoising steps, early exits** | good; wide spans come free |
| **epochs along one run, fixed LR** | a valid frontier, but in **training** cost — the price does not transfer to inference |
| **learning-rate sweep, retrained families** | a hyperparameter *envelope*, not one clock |
| **quantization levels, pruning without fine-tuning** | too few points, or the mechanism changes per level |
| **batch size** | `s = 0`, no price — and F18/F21 need none |

On the archive, training-resource axes pass Law I *more* often than inference
axes (`35` in class against `4` out, versus `51` against `20` plus `5`
envelopes), so an epoch-like axis is a fine frontier — the problem is never the
shape, it is the **coordinate**.

**The special case.** If your candidates differ *only* in weights — same
architecture, different epochs, different learning rate — their inference cost is
identical, so `x_i = x_j` and `sigma_i - sigma_j = ln(eps_j/eps_i)` for every
price. There is no trade-off to price: take the most accurate, certified by
`PRICE-FREE`, and no sweep, grain or price enters anywhere.

## What it costs to run, in full

| | Level 0 | Level 2 (derived price) |
|---|---|---|
| models to train | **0** | **0** |
| inference passes over the test set | 1 per candidate | 1 per candidate **+ 5** (the sweep) |
| numbers to declare by hand | **0** | **1** — the grain `t` |
| price set by hand | **none** | **none** |

At Level 0 nothing at all is declared *about the problem*: the envelope, the
ladder, the price-free test and the ranking are read off the measurements, with
no price, grain, chance level, floor or chart. One statistical convention does
remain, and the table's `0` is about the problem and not about it: `--alpha`,
the coverage level, defaults to `0.10` and every certified interval and tier
moves with it (the first rung's interval is `[0.2516, 0.3423]` at `0.10` and
`[0.2670, 0.3261]` at `0.5`). It is printed in the run header for that reason. The grain enters only at
Level 2, where it fixes the operating point `N_eff = 1/t` and hence
`lambda* = sqrt(kappa/t)`; it is the one input no frontier can supply
(Thm. `noinvariant`). `q0` is never needed for a decision — it is a constant
shift of `ln eps` and cancels from every difference of `sigma`, verified to
`1e-14` on all 117 frontiers — and moves only the floor and depth readings.

## What each line is

| line | statement | where |
|---|---|---|
| `grain` | `t`, the number of independently drawn items a delivered unit must all fail. The one input no frontier can supply. | Thm. `noinvariant`, Prop. `retry` |
| `class gate` | `ln eps` convex: `l[-] - 2 l[0] + l[+] >= 0`. Exact, one-sided, any spacing. The verdict rests on the **global chord**. | Prop. `rungs` |
| `envelope test` | the share of jagged wide triples, against the 95th percentile of that share under the frontier's **own** convex-minorant-plus-noise null. | new, A7 |
| `divisibility` | that second difference never grows (Widder, exact at every spacing). A rise refutes; no rise proves nothing. | Prop. `rungs`, `wallivp` |
| `precision gate` | `sd(ln eps) = sqrt(Q/((1-Q) n_test))`, propagated into `N_eff` and the price. | F3 |
| `paired` | `Var(l_i - l_j) = V1/n` with the covariance term the shared test set creates. | new, F11 |
| `design rule` | `h >= sqrt(sqrt(6) sd(l) / (tol kappa))` with `kappa = s^2 t` at the operating point: the grid width, before the runs. | new, F13 |
| `bias` | `(4 A_h - A_2h)/3`: the `O(h^2)` plug-in bias removed, when it exceeds the price's own noise. | new, F16 |
| `ENVELOPE` | the upper envelope of `sigma_i(lambda) = S_i - lambda x_i`: convex, piecewise linear, one sort and one stack pass. Its pieces name the winner at each price; absence from it is proof of never being optimal. | new, F22 |
| `BREAK-EVEN` | `lambda_be = ln(eps_C/eps_D)/(x_D - x_C)` at each envelope breakpoint, with F12's certificate inverted in `lambda` by bisection. No sweep, no grain, no price. | new, F21 |
| `RANK` | the ranking as one table: certified tier, `sigma` at the price in force, and the envelope interval. Read down the envelope column and it is the deployment schedule. | new, F19/F22 |
| `CERTIFIED SET` | `sigma_i > sigma_j <=> e_j > e^{c_i - c_j} e_i`, certified by a one-sided empirical-Bernstein bound on a bounded per-item variable, at `alpha/(K-1)`. | new, F12 |
| `PRICE-FREE` | no dearer on any axis, and certified no worse on accuracy: domination at every non-negative price, with no grain and no sweep. | new, F18 |
| `RANKING` | certified tiers: the partial order the certificates establish, laid out in layers. Within a tier, nothing is separable. | new, F19 |
| `PRICE SHARPNESS` | the derived price's interval against a candidate's reign over prices. `lambda* = sqrt(kappa/t)`, so `sd(lambda)/lambda ~ sd(l)/h^2`: a factor `F` of sharpening is `sqrt(F)` in sweep width, `F^2` in items. | new, F20 |
| `SCHEDULE` | which network at which price, each cell certified; the number of distinct entries is the number of networks the deployment needs. | new, F12/F14 |
| `CANDIDATES` | the concave upper envelope of Law I: the most an unseen network could still be worth, and where it would sit. | new, F14/F15 |
| `under-sizing` | `1/N^x = 1/N^k + s_m/s`, so a compute-axis reading is a lower bound. | Prop. `neff` |
| `count axis` | complete monotonicity on integers (Hausdorff); the floor as `P(ceil L) <= eps_inf <= P(floor L)`; the retry discount `D(t)`. | Prop. `countaxis` |
| `plane` / `operate at` | every constant-`sigma` chart is a straight line on the plane: slope `sigma`, `S`-intercept `-ln eps_inf`. | Prop. `odepencil`, `planeprice` |
| `walls` | `eps_inf <= eps/(1+N_eff)` (axiom) and `<= eps e^{-N_eff}` (divisible). | Prop. `wallivp`, Thm. `budget` |
| `arc` | the Fisher-Rao arc *length*, `dL = |dS|/sqrt(N_eff)` -- not the channel angle `phi = 2 arctan sqrt(N_eff)` in `(0,pi)`; `K = l/(2 sqrt(2 rho))` checkpoints cover it. | Thm. `covering` |
| `price` | `lambda* = s/sqrt(t N_eff)`, hence `lambda* = s` at the operating point. | Prop. `planeprice` |
| `BAND (v2)` | `sigma = S - sum_i lambda_i x_i`; differences below the band are not gaps. Kept for comparison; it is not the verdict. | Thm. `covering` |
| `INPUTS` | the strongest statement the supplied inputs support. | new, F17 |

`N_eff` costs three accuracies and no spacing: it is a square over a second
difference, so `h` cancels identically.

```
    N_eff = (l[+] - l[-])^2 / (4 (l[-] - 2 l[0] + l[+])),      l = ln eps
```

## Inputs

### Option 1 — from a file

`measurements.csv` — **the one file Level 0 needs.** Row 1 names the candidates;
exactly one further row has a **non-numeric first cell** and carries the
resource, its first cell naming the axis; every other row is one test item,
`1` = correct. Accuracy and `n_test` are **counted** from the item rows:

```
net_a_small,net_d_int8,net_c_mobile,net_b@224
inference_time_ms,4,9,18,24
0,1,1,1
1,1,1,1
```

```bash
python3 select_model.py measurements.csv
```

`pairs.csv` — the same information as the pairwise `2x2` tables, which is all
the certificate ever reads. `K(K-1)/2` rows instead of `n_test`:

```
i,j,both_fail,i_only,j_only,neither
net_a_small,net_d_int8,3326,4124,386,42164
```

```bash
python3 select_model.py candidates.csv --pairs pairs.csv
```

`candidates.csv` — the aggregates: one row per candidate, any number of resource
columns. `n_test` is required for any statement with a number attached:

```
name,accuracy,inference_time_ms,n_test
net_c_mobile,0.948000,18.0,50000
```

```bash
python3 select_model.py candidates.csv            # unpaired, conservative
python3 select_model.py candidates.csv --rho 0.5  # paired, with a declared rho
```

`items.csv` — the item matrix beside those aggregates: one column per candidate,
one row per test item. Equivalent to `measurements.csv`:

```
net_a_small,net_d_int8,net_c_mobile,net_b@224
0,1,1,1
0,0,0,0
```

```bash
python3 select_model.py candidates.csv --items items.csv
```

Without per-item outcomes **and** without `--pairs`, pass `--rho` (a declared
failure correlation; the archive measures `0.50` over all pairs and `0.77` over
adjacent ones) for the paired normal certificate, or nothing at all for the
unpaired one, which is conservative and says so.

### Option 2 — from the command line, no file at all

One `--model` per candidate, `name,accuracy,resource[,n_test]`, with `--axis`
naming the resource:

```bash
python3 select_model.py --axis inference_time_ms \
    --model net_a_small,0.851000,4,50000 \
    --model net_d_int8,0.925760,9,50000 \
    --model net_b@224,0.955320,24,50000 \
    --model net_f_large,0.969800,90,50000
```

That is the unpaired certificate. For the **exact** one, add one `--pair` per
pair, `i,j,both_fail,i_only,j_only,neither` — the same six numbers as a row of
`pairs.csv`:

```bash
python3 select_model.py --axis inference_time_ms \
    --model net_a_small,0.851000,4,50000 \
    --model net_d_int8,0.925760,9,50000 \
    --model net_b@224,0.955320,24,50000 \
    --model net_f_large,0.969800,90,50000 \
    --pair net_a_small,net_d_int8,3326,4124,386,42164 \
    --pair net_a_small,net_b@224,2118,5332,116,42434 \
    --pair net_a_small,net_f_large,1463,5987,47,42503 \
    --pair net_d_int8,net_b@224,1775,1937,459,45829 \
    --pair net_d_int8,net_f_large,1318,2394,192,46096 \
    --pair net_b@224,net_f_large,1108,1126,402,47364
```

`--price-interval LO:HI` works the same way from the command line, so Level 1
needs no file either.

> **The two commands above are abridged.** They carry four of this example's
> seven candidates, which is enough to show the syntax and reproduces the
> envelope and the ladder of those four exactly (`0.2963`, `0.5177`, `0.8591`
> with the same certified intervals). They do **not** reproduce the
> seven-candidate run, and they should not: with `net_c_mobile`,
> `net_e_hybrid` and `net_g_dominated` left out there is nothing for the
> price-free test to certify, so that block disappears and the envelope reads
> "4 of 4" rather than "4 of 7". Pass all seven `--model` arguments and all
> twenty-one `--pair` arguments to reproduce it.

### The two options give the same answer, and that is tested

`scripts/157_input_routes.py` runs all four routes on the **same** candidate
list and the **same** pair list — `measurements.csv`;
`candidates.csv --items items.csv`; `candidates.csv --pairs pairs.csv`; and
`--model`/`--pair` built from all seven candidates and all twenty-one pairs of
those files — and diffs the decision
blocks. They are **identical** (the `--pairs` route differs only in the
`inputs used` line, which names `2x2 tables` instead of `per-item outcomes`).
What is tested is that the four *routes* agree on the same inputs, not that a
shorter candidate list gives the same answer as a longer one. Writing that test found a real defect: the tier
order was being read from set iteration, so the reported ranking depended on the
process's hash seed. It is now deterministic.

### The consistency gate (F24)

The certified interval of a rung is read off a `2x2` table while the point
break-even is read off the two accuracies, so a **stale aggregate beside a fresh
table** would print a point outside its own interval. A table fixes both error
rates and the item count exactly, and an item vector fixes the accuracy exactly,
so these are compared at the door and a disagreement is refused:

```
INPUTS DISAGREE (F24) -- refusing to decide:
  net_a_small: accuracy implies a failure rate of 0.147000 but the
               net_a_small/net_d_int8 table implies 0.149000 (100 items apart)
```

`--force` proceeds anyway, and says what that costs.

### For Levels 1 and 2

`sweep.csv` — **one** model, fixed weights, at 5–7 budgets of **one** axis
(five is the protocol's figure; three is the algebraic floor for a second
difference), on a **geometric** grid spanning what F20 asks for. Input size is
the cheapest such sweep; a width or depth multiplier, a token budget or a
number of diffusion steps all work the same way. Without it **no price is
derived, and none is needed** — the envelope and the ladder are read as
functions of `lambda` (F21, F22).

A pass-at-`k` file — `k,P`, the probability an item is still failed after `k`
independent attempts, passed with `--passk`. This is the axis on which the
theory is a theorem rather than an assumption, and the only one that brackets
the floor. Two are shipped: `passk_demo.csv` and the measured
`passk_fmnist_nested.csv`; there is no file called `passk.csv`.

### All flags

`--axis NAME` (which resource the envelope is read on), `--model`, `--pair`,
`--items`, `--pairs`, `--rho`, `--alpha`, `--price-interval LO:HI`,
`--price axis=lambda`, `--calibration`, `--calibration-axis`, `--resources`,
`--grain t` or `t1:t2` (**only** with `--calibration`), `--passk`,
`--attempt-budget`, `--retry-budget R`, `--theta`, `--floor`,
`--no-bias-correction`, `--q0`, `--q0-convention`, `--tie-band`, `--json`,
`--force`.

Three of these sit outside the three levels and are listed here only because
they exist. `--price axis=lambda` declares a price, for a deployment that
already knows one and wants the single-price answer; the run then labels the
price `[declared]` rather than `[derived]` and prints a certified set at that
price. `--floor`, `--q0` and `--tie-band` are read at Level 2 only and are
echoed back under `NOT READ AT THIS LEVEL` when a Level-0 run supplies them.
`--alpha`, `--rho` and `--q0-convention` are conventions rather than inputs of
the problem and are always in force; `--alpha` is in the run header.

## Honest limits

All of these are theorems of the thing being used, not defects of the
implementation. Section 2 of
[`../../PRACTICAL_PIPELINE.md`](../../PRACTICAL_PIPELINE.md) has the full list;
the five that bite hardest here:

1. **`N_eff` is a ratio whose denominator is a second difference, so it is
   noisy.** On the paper's own corpus of 117 published frontiers the curve is
   noise-dominated — median straightness `R^2 = 0.22` — and against a
   design-matched null it has **no power at all** (`scripts/129`–`131`). F13 now
   says exactly what grid would change that, and on **not one** of the 36
   frontiers with a public item count is the derived price identified at the
   grid and items it was measured with.
2. **A point winner is a fiction near the optimum.** `13.8` consecutive models
   fit inside one indifference band on a median published frontier (Thm.
   `covering`), which is why the output is a set — and why the set now carries a
   coverage level instead of an adjective.
3. **The certificate is about the price you declare.** It guarantees coverage of
   the model optimal at the realised price, over the interval you give it. Give
   it six e-folds and it will correctly refuse to discard almost anything;
   that is a statement about your price, not about the method.
4. **Independence between candidates is refuted**, so there is no theory here
   for ensembling, routing or cascades. F11 turns the same correlation into an
   asset for *selection* — it is why the paired band is half the width — but it
   does not give a theory of combining models.
5. **A tier is not a tie.** Candidates inside one tier are not equivalent; they
   are *unseparated at this item count and this price interval*. The output says
   what it would take to separate them, per pair, and that number is often
   large: `200,555` items for `net_c_mobile` against `net_b@224` in the run
   above. A tier that will not shrink is the data telling you the choice does
   not matter.

The demo data is **generated**, so the agreements above are checks of the
implementation against a known truth. They are not evidence for any law.

## Files

| file | what |
|---|---|
| `select_model.py` | the CLI, version 3. The label names the interface -- the three levels, the four input routes and the certified set -- and has not moved since revision `e`; what changed inside it at each revision is recorded in `../../PRACTICAL_PIPELINE.md`, not here |
| `pipeline_core.py` | the reading layer: gates, count axis, price, paired variance, the certificate, the design rule, the stopping rule. Stdlib only, and the same module `scripts/139`–`156` import (`157` drives this CLI as a subprocess instead, and `158`–`159` are independent), so the archive's numbers and this example's cannot drift |
| `make_protocol_data.py` | regenerates `measurements.csv`, `pairs.csv`, `candidates.csv`, `items.csv` and `sweep.csv` from the cure chart — **the protocol** |
| `measurements.csv` | **the one file Level 0 needs**: per-item outcomes with the resource on its own row |
| `pairs.csv` | the same information as `K(K-1)/2` pairwise `2x2` tables — 21 lines, identical certificate |
| `sweep.csv` | the five-point sweep, for Level 2 |
| `candidates.csv`, `items.csv` | the aggregate + item-matrix form, equivalent to `measurements.csv` |
| `make_demo_data.py` | the earlier five-candidate demo, kept so versions 2 and 3 can be compared on the same numbers |
| `models.csv`, `items_demo.csv`, `calibration.csv`, `calibration_wide.csv`, `passk_demo.csv` | that demo's inputs |
| `passk_fmnist_nested.csv` | a **measured** count axis, from the archive's per-item panel |
| `models_v1_backup.csv`, `calibration_v1_backup.csv` | the version-1 demo inputs, kept so the versions can be compared on the same numbers |
