# Making the theory work in a prediction pipeline

Revision of **2026-10-08e**.  This note is the audit of what stands between the
theory as published and a selector that can be pointed at an arbitrary set of
neural networks and trusted, together with the twenty-four fixes that were made, the
measurements that check them, and the limitations that are **theorems and
therefore stay**.

The target was stated as: work for absolutely different cases, with the
**minimum number of networks** and the **maximum correctness** of picking the
best one, from the **minimum number of input parameters**.

Revision `c` made ten fixes, `F1`-`F10`.  An audit of that audit found that the
central claim -- maximum correctness -- was still unquantified: the verdict was
an *indifference band*, and a band is not a confidence statement about
anything.  Revision `d` added seven more, `F11`-`F17`, of which `F11` and `F12`
carry the weight: **the decision became a set with a coverage guarantee,
computed from per-item outcomes, valid with no appeal to Law I.**

Revision `e` finishes the job the whole exercise was for: **a protocol a
deployment can actually run, in which no price, grain or chance level has to be
set by hand at all.**  (`--price` still exists, for a deployment that does know
its price and wants the single-price answer; supplying it is a declaration and
the tool labels the price `[declared]` rather than `[derived]`.  Nothing in
levels 0-2 needs it, and a run that uses it is outside the three levels.)  Six more fixes, `F18`-`F23`, a consistency gate `F24`, and one
worked example that is the protocol itself.  Everything runs offline from the archive as shipped.

---

## The protocol, in three levels

> **Level 0 -- the whole answer from ONE file.**
> `measurements.csv`: row 1 names the candidates, one row carries the resource
> (its first cell names the axis), every other row is one test item.  Accuracy
> and `n_test` are **counted** from the item rows, so there is nothing to keep
> in step.  **Nothing else is supplied:** no price, no calibration sweep, no
> grain, no chance level.
>
> `select_model.py measurements.csv`
>
> Two other shapes carry the same information.  `--pairs pairs.csv` gives the
> `K(K-1)/2` pairwise `2x2` tables instead of the item matrix -- all the
> certificate ever reads -- so a 50,000-item evaluation of seven candidates is
> **21 lines** with an identical result; and `candidates.csv` alone, with
> aggregates only, falls back to the **unpaired normal** certificate.  Against the
> **paired normal** band that is conservative by `1.36x` overall and `2.1x`
> between adjacent models (`results/table_paired_band_summary.csv`); against
> the **exact** certificate the two routes above deliver it is not uniformly
> conservative, and on the shipped demo it is slightly narrower, so the
> aggregate route is a convenience and not a safe fallback.
>
> **Level 1 -- if you can bound your own price.**
> `select_model.py measurements.csv --price-interval LO:HI` -- still one file,
> still no grain and no price set by hand, only a *bound*.  The certified SET
> over that interval comes with it, and tiers that use it.
>
> **Level 2 -- if you will name no price at all.**  Add a five-point sweep of
> ONE model (fixed weights, five input sizes, geometric, at or below the
> training resolution) and the grain `t`; then `lambda* = sqrt(kappa/t)` is
> derived, and `N_eff`, the walls and the floor become readable too.

Level 0 is complete, not a fallback: the ladder's rungs **tile the whole
positive price axis**, so whatever the deployment's price turns out to be,
exactly one candidate is named -- and the candidates named by no price at all
are retired with a proof.  Only `F20`'s sharpness question and the second-order
readings need more.

**Why none of price, grain or `q0` is an input of the decision** (`F23`,
script `156`).  The envelope is `max_i (S_i - lambda x_i)` read as a *function*
of `lambda`, so it is computed without knowing `lambda`; its breakpoints are the
break-evens and the rung at `lambda = 0` is the price-free test.  The grain
enters only through `lambda* = sqrt(kappa/t)`, the operating point and the
wall and depth bounds.  And `q0` enters as
`ln eps = ln(1-Q) - ln(1-q0)`, a constant shift of `l`, so it cancels
identically: measured over `q0` in `{0, 0.001, 0.01, 0.1, 0.5}` on every
archived frontier, the largest deviation in any `sigma` difference is
`1.8e-15`, in any break-even `5.0e-14`, and **no** envelope winner set changes.

**Measured in Level 0 alone**, on all 117 archived frontiers: median `11`
candidates of which `5` can ever be optimal (**62.5% retired outright**) and a
ladder of `4` rungs -- and, on the 36 of them with a public item count, a median
undecided interval of `0.43` doublings with `100%` of rungs decided on at least
one side.

What comes back, ordered by how much of the price axis each statement commits
to -- which is also the order of the levels that deliver them:

1. **price-free dominations** -- a candidate no dearer on any axis and
   certified no worse on accuracy loses at *every* non-negative price (`F18`);
   the **price envelope** -- which candidates a linear price can ever select,
   computed without identifying any frontier (`F22`) -- and, at each of its
   breakpoints, the **break-even price** with its certified interval (`F21`),
   neither of which needs a sweep.  **Level 0.**
2. **a certified set** -- covering the candidate that is optimal at the
   realised price with probability at least `1 - alpha` (`F12`).  **Level 1**:
   it consumes the price interval.
3. **a certified ranking** -- the whole list in tiers over that interval, within
   which nothing is separable at the items in hand and between which everything
   is (`F19`).  Also **level 1**, and *not* price-free: tier membership moves
   with the interval.  What level 0 delivers of the ranking is the coarser
   **price-free partial order**, which is the `tier` column of the `RANK` table
   a bare `select_model.py <file>` prints; the `RANKING` block appears only once
   a price or an interval is supplied, and the archive's "median four tiers,
   median top tier 19 of 32" are measured at a derived price (script `151`).
4. **and what to do next** -- whether the price is sharp enough to name one
   winner, by what factor to widen the sweep if not (`F20`), which networks
   serve which prices if you would rather not (`F12`), and where an unseen
   network could still help (`F14`, `F15`).  These read the derived price and
   are **level 2**.

The **first** is what `select_model.py <file>` prints with nothing declared,
together with the envelope, the ladder and the price-free partial order; items
2 and 3 need `--price-interval`; item 4 needs `--calibration` and `--grain`.
Do not read items 2-4 as available at level 0 -- earlier
revisions of this file listed the set, and then the tiers, as if they were.

Two scenarios, from the same files.

**To pick one network**, stopping at the first that answers: `PRICE-FREE` (no
extra input at all); `ENVELOPE`, where the price falls in exactly one piece and
`BREAK-EVEN` certifies the boundary (still no extra input); the `CERTIFIED SET`
at a price interval you can bound; or, if no price will be named at all, the
certified set at the *derived* price, which is the only route that needs the
sweep and the grain.

**To rank all of them by the trade-off**, read the `RANK` table, which carries
the two honest layers and nothing else: the **tier** (certified dominance at the
price in force -- within a tier nothing is separable, between tiers everything
is, and the `sigma` order inside a tier is presentation rather than evidence),
and the **envelope interval** (the price range over which each candidate is the
best choice, independent of any declared price, where `never optimal` is a proof
and not a statement about noise).  Read down the envelope column and the same
table is the deployment schedule.

The worked example is `examples/model_selection/`, which runs the protocol end
to end against a known truth and recovers the canonical price to `-5%`, in the
direction the plug-in bias predicts.  Its README is the line-by-line
description.

---

## 0. The three answers, as they now stand

### Minimum number of networks

Three, placed by design -- not four, placed evenly.

Script `148` measures the smallest K whose worst case over a price range is
within one doubling (`rho_2 = 0.0589` nats), on all 117 frontiers:

| placement | median K | frontiers reaching `rho_2` within 8 | median worst case at K = 4 |
|---|---|---|---|
| equal arc (Thm *covering*) | 4 | 73 of 117 | `0.0967` nats |
| equal log-resource | 4 | 87 of 117 | `0.0949` nats |
| **greedy sequential (F15)** | **3** | **117 of 117** | **`0.0023` nats** |

All three rows are script `148` on all **117** frontiers.  Script `143`, which
thins each frontier rather than placing fresh points and keeps **111**, gets
`72` and `85` instead of `73` and `87`, and at `K = 4` reverses the first two
rows (`0.083` for arc against `0.095` for log-resource).  The two scripts
measure different things on different populations and both are archived; what
follows from having both is that the arc-against-log-resource comparison is
inside the noise of the plug-in arc, which is why the paper withdraws it and
keeps only the count.

Equal spacing is not a design.  The quantity to be covered is the regret over
the price range, so the next network belongs where that regret is, and placing
it there is a forty-fold improvement at K = 4 and the difference between
reaching the tolerance on 62% of frontiers and on all of them.  This also
settles the question revision `c` reported as a negative: neither equal-arc nor
equal-log-resource is right, because neither is a design.

And the count depends on something revision `c` never separated: **how well the
price is known.**  Over the whole range a frontier supports -- a median `6.71`
e-folds of price -- a median of 5 different models are optimal somewhere, and no
single model can be within `0.637` nats of the best choice everywhere.  Told
the price to within a doubling either way, the same frontiers need a median of
**2** networks, and one compromise network costs only `0.074` nats worst case
instead of `1.085`.  Revision `c`'s four came from thinning a frontier at *equal
spacing* (script `143`), so the count it reported was a property of the
placement rule as much as of pricing over six e-folds.  Placed by design the
count is three (script `148`), and priced to within a doubling it is two.

Add the calibration sweep the price needs -- one model at 5 input sizes, no
retraining -- and the honest total is

> **one model swept at 5 input sizes, plus 3 candidates**, or **2** if the
> price is known to within a doubling.

The sweep is not an extra model.  It is the same model five times, with fixed
weights and a different input size, which is why `F20`'s design rule is
affordable at all.

Proving that no unseen network would help is a stronger demand than deciding
among the ones in hand: the stopping rule of `F14` bounds what an unseen model
could be worth by the concave upper envelope Law I provides, and a *certified*
stop needs a median of 5 networks.  Three to decide, five to prove.  The rule
is never wrong in the other direction: **0% false stops** at every K on the
whole corpus.

### Minimum input parameters

**Zero declared numbers for the decision.**  Level 0 declares nothing at all;
Level 1 adds a price *bound*, not a price; Level 2 declares exactly one number,
the grain `t`, and only because `lambda* = sqrt(kappa/t)`.  `q0` is never an
input of a decision -- it is a constant shift of `ln eps` and cancels from every
difference of `sigma` (`F23`, script `156`).  Per candidate: the **per-item
outcome vector** and one figure per priced resource; `accuracy` and `n_test` are
*counted* from the vector, not declared.

One honest qualification.  "Nothing declared" means nothing about the *frontier*
or the *market*: no price, no grain, no chance level, no floor, no chart.  It
does not mean no statistical convention.  The coverage level `--alpha` defaults
to `0.10`, and it is a declared number in the ordinary sense -- every certified
interval and every tier moves with it.  On the worked example the first rung's
certified interval is `[0.2516, 0.3423]` (`0.44` doublings undecided) at
`alpha = 0.10` and `[0.2670, 0.3261]` (`0.29`) at `alpha = 0.5`.  It is reported
in the run header for that reason.  What the protocol claims is that no quantity
*of the problem* has to be guessed, not that a confidence level is free.
That last one is not a new measurement: it is what was summed to get the
accuracy.  Supplying it is what turns the verdict from a band into a
certificate (`F11`, `F12`), and it is the only input in the whole pipeline that
buys a guarantee for free.

Globally: **the grain `t`** -- the only input no frontier can supply, and an
input of the *second-order* reading only (the operating point, the walls, the
floor and the derived price), never of the envelope, the ladder, the tiers or
the certified set.  The chance level `q0` is recorded for reporting and is an
input of no decision at all: it is a constant shift of `ln eps` and cancels
from every difference of `sigma` (`F23`, `scripts/156`), and since revision
2026-10-08m it also cancels from the `F24` consistency gate, which compares raw
failure rates on both sides (`scripts/157`, section C).  Nothing smaller than
three points yields `N_eff`, because `N_eff` is a second derivative.

### Maximum correctness

Now a number rather than a claim.  The certified set of `F12` satisfies

> Pr[ the model that is optimal at the realised price is in the returned set ]
> `>= 1 - alpha`,

from a one-sided distribution-free finite-sample bound on a bounded per-item
variable, at `alpha/(K-1)` per comparison.  No distributional form, no
asymptotics, no convexity, no `N_eff`, no floor.  Measured on the co-failure
panel by resampling (script `146`: 300 replicates per cell, 2 families, 21
candidates, item counts 200 / 1,000 / 5,000), the certificate holds in every
cell.  Two things the same measurement shows, and both change what matters:

* the version-2 band **over-covers at a known price**: its `rho_2` floor is
  wider than the test-set noise at these item counts.  Its measured 45% miss
  rate (revision `c`, script `143`) was therefore never a noise failure -- it
  was **price uncertainty**, which a band says nothing about.
* tightening the band with the paired variance alone (`F11`, no certificate)
  drops coverage to `0.947` at 200 items.  Tightening without certifying costs
  correctness; that is why `F11` and `F12` ship together.

On the corpus, at a known price, the certified set is a median of **2 of 32**
candidates and reaches a decision on **100%** of frontiers; with the price
known only to a doubling either side it is 7 of 32 and 97%.  And of the **24
frontiers the class gate refuses outright**, where version 2 named no winner
at all, **9 have a public item count and it returns a set on all 9** -- median
size 1.  The other 15 want item counts their benchmarks do not publish, which
is a missing input and not a failure of the certificate.

---

## 1. The fixes

`F1`-`F10` are revision `c` and unchanged; `F11`-`F17` are revision `d`;
`F18`-`F24` are revision `e`.  Each is tagged
at its definition in `examples/model_selection/pipeline_core.py` and in the
selector's output.

### F1 - the grain is declared where it is used, and never defaulted
`--grain` is required **only with `--calibration`**, where the price is derived
at the operating point; it is not read by the envelope, the ladder, the
price-free test or a declared price interval (`F23`).  Theorem *noinvariant*
says an operating rule must take one input from outside the frontier, and
equation `(retry)` names it: the grain, the number of independently drawn items
a delivered unit must all fail.  Given an interval `--grain t1:t2` the tool operates at
`1/sqrt(t1 t2)` and prices the worst case at exactly
`ln cosh((1/4) ln(t2/t1))`.

### F2 - Law I is a gate, not a footnote
Law I is tested **before** anything is priced: convexity of `ln eps`,
one-sided, exact at every spacing, on wide triples with an absolute floor on
the arm length.  Four verdicts: `IN_CLASS`; `ENVELOPE` (the chord is convex
but the interior is jagged -- a Pareto envelope of several models, so a price
may be quoted but `N_eff`, the walls and the floor may not); `OUT_OF_CLASS`
(the global chord is concave); and `UNDECIDABLE` (too few admissible triples to
test, which is a refusal to decide and not a pass).

**Measured** (script `139`, 117 frontiers): `24` refused outright, `5` flagged
as envelopes under the original threshold.  The discriminating statistic is the
*share* of wide triples that violate convexity: distribution shift `0.748`
against `0.168` elsewhere, and 6 of 9 shift frontiers refused against 18 of
108.

**Revision d** replaces the threshold itself -- see `A7` below.

### F3 - test-set noise is propagated, and the cheap remedy is named
`sd(ln eps) = sqrt(Q/((1-Q) n_test))`, propagated into `N_eff`, into the
derived price (robust quantile scale; the draws are heavy-tailed), into a
pairwise band, and into a `price not identified` verdict.  Because `kappa` is a
second difference whose signal is `h^2 kappa` and whose noise is not,
**widening the geometric grid is quadratically cheaper than buying items**: a
factor `f` in width is worth `f^4` in items.

**Measured** (script `140`, the 36 frontiers whose evaluation-set size is
public): median `sd(ln eps)` is `11.5` milli-nats against the `0.73` the
paper's power calculation requires; `N_eff` is resolved at the items they have
on **8 of 36**; the median remedy is a `1.9x` wider grid.

### F4 - both objectives, always
The accuracy-only canonical value `J` and the resource-aware `sigma` agree on
only `40.7%` of the paper's panel cells, with a ninetieth-percentile regret of
`0.65` nats.  Both are printed and the disagreement is priced.  `J` is reported
only when a floor is **measured** (F6) or declared.

### F5 - n resources, one price per axis
Any number of resource columns; each priced from its own marginal curve where
one exists and by declaration otherwise, with the provenance of every price
printed.  The interaction is **bounded, not fitted**.

**Measured** (script `144`, 48 multi-resource blocks, 114 axis prices):
dropping one axis changes the decision on `64%` of (block, axis) pairs --
`79%` for axes carrying an above-median share of total cost.  Pricing every
axis is not optional.

### F6 - the count axis, where the theory is a theorem
Given `--passk` the tool reads the frontier where the axiom is a **theorem of
exchangeability**, complete monotonicity is exact and signed at every spacing
(Hausdorff), and the floor is **bracketed by two measured numbers**.

**Measured** (script `141`), in closed form from the archive's own per-item
panel (10,000 items x 21 models x 2 families): complete monotonicity holds on
orders 1-3 in both families; floor `0.0518` (nested) and `0.0342`
(independent), i.e. `18.9%` and `12.5%` of the one-attempt error; `D(2) =
0.709` and `0.727`.

### F7 - the under-sizing is corrected, and the correction is a test
`1/N^x = 1/N^k + s_m/s` identically, so a compute-axis channel count is a
**lower bound** and the rule fires too early.

**Measured** (script `142`): on the independent family `N^x = 0.228 <= N^k =
0.691`, `s_m/s = 2.94`.  On the **nested** family the inequality **fails**, and
that failure is the useful result: comparing the two readings is a cheap,
bandwidth-free **test of whether a pipeline's retries are exchangeable at
all**.

### F8 - the band's tiebreak
`13.8` consecutive models fit inside one indifference band on a median frontier
of the archive's 12-frontier vision panel, whose frontiers are its longest.
Inside the band the tie is broken on **worst-case regret over the price range
the frontier supports**, which cuts the median excess over the
minimax-optimal model from `0.075` to `0.029` nats, measured frontier by
frontier on the 91 the class gate admits.  (The *difference of the medians* is
the larger `0.329` against `0.269`.  Those are two different statistics and
earlier revisions of this file paired one statistic's "before" with the other's
"after"; the per-frontier pair is the one to quote.  The rule's own single pick
has median excess `0.088`.)  Retained, but it is no
longer the verdict: see `F12`.

### F9 - the candidates need not be one family
The price comes from the calibration frontier; the score comes from the
candidate.  A heterogeneous zoo -- architectures, quantizations, distillations,
hosted APIs -- is scored on a common declared price with no requirement that it
lie on one curve.

### F10 - a retry budget is converted to the grain it delivers
`S(t)/(t S(1)) = D(t)` in `[1/t, 1]` and non-increasing, so a budget of `t_r`
retries delivers grain `S(t_r)/S(1) < t_r`.  Two readings of `D(2)` are
archived and they are different constructions, so each is labelled: the
exchangeable census `P(k) = mean C(f_i,k)/C(M,k)` over the panel's models
(`scripts/141`, `results/table_count_axis_panel.csv`) gives `D(2) = 0.709` and
`0.727`, so two tries deliver grain `1.42` and `1.45`; the nested retry chain measured on
the co-failure panel (`scripts/114`, `results/table_attempt_budget.csv`) gives
`D(2) = 0.547` and `0.522`, so two tries deliver grain `2 D(2)` = `1.09` and
`1.04` respectively -- the second attempt buying nine and four per cent of what
the first did, which is the paper's phrasing.  The
paper quotes the second, because it is the one that measures a *retry* on the
same item.  Either way `D(2) < 1` and the budget buys less grain than it
spends.

---

### F11 - the gap is PAIRED  *(new in d)*

Candidates are evaluated on the **same items**, so the quantity a decision
turns on is a difference of two means taken on shared data:

```
Var(l_i - l_j) = V1/n,
V1 = (1-e_i)/e_i + (1-e_j)/e_j - 2 (p_ij - e_i e_j)/(e_i e_j),     l = ln eps
```

Version 2 used the unpaired sum `sd_i^2 + sd_j^2` -- the first two terms -- and
the matching factor 2 in the required item count.  The third term is strictly
negative whenever failures are positively associated, which the archive's own
panel measures: `Pr[both fail] = kappa min(h_i,h_j)` with `kappa = 0.835`.  So
the unpaired band is too **wide** and the `n_test` it demands too **large**.

**Measured** (script `145`, every pair of the co-failure panel -- 420 pairs,
10,000 items, both families, so every `p_ij` is a count and nothing is
assumed):

| | all pairs | **adjacent pairs** |
|---|---|---|
| median failure correlation `phi` | `0.496` | **`0.774`** |
| co-failure excess over independence | `+162%` | `+288%` |
| band shrinkage | `x0.738` | **`x0.475`** |
| item count needed | `x0.544` | **`x0.226`** |
| comparisons resolved at the items in hand | `96.2% -> 99.5%` | **`62.5% -> 95.0%`** |

The adjacent column is the one that matters, because near-neighbour ranking is
the hard case by theorem (*covering*).  There the unpaired band was **2.1x too
wide** and demanded **4.4x too many items**, and pairing decides 13 of 40
comparisons that were ties -- *at no extra cost and with no new input
parameter*.  This is the cheapest correctness gain in the pipeline.

### F12 - the verdict is a CERTIFIED set  *(new, and the load-bearing one)*

Write `sigma_i = -l_i - c_i` with `c_i = sum_a lambda_a x_{ia}`.  Then

```
sigma_i > sigma_j   <=>   e_j > R0 e_i ,     R0 = exp(c_i - c_j),
```

so domination is a statement about the mean of the **bounded per-item
variable**

```
v_k = 1{j fails item k} - R0 1{i fails item k}   in  [-R0, 1],
E v = e_j - R0 e_i ,
```

whose four values -- and therefore whose mean and variance -- are closed forms
of the pair's 2x2 table.  A one-sided **empirical-Bernstein** lower bound on
`E v` (Maurer and Pontil) therefore *certifies* domination from finitely many
items with no distributional assumption.  Candidate `j` is excluded only when
one rival certifies domination at **every price in the price-uncertainty
interval**, each pair tested at `alpha/(K-1)`.  The model that is optimal at
the realised price survives unless one of those `K-1` certificates is wrong, so
by a union bound

```
Pr[ the price-optimal model is in the returned set ] >= 1 - alpha.
```

Three properties matter for "absolutely different cases":

1. **It needs no Law I**, no `N_eff`, no floor and no derived price.  It is the
   only statement in the pipeline that survives `OUT_OF_CLASS` -- and it does:
   of the 24 refused frontiers, the 9 with a public item count all get a set
   (median size 1) where version 2 named no winner; the other 15 lack the item
   counts the certificate needs.
2. **The minimax-optimal model can never be excluded by a correct
   certificate.**  If one rival beat it at every price it would itself have a
   smaller worst case, a contradiction.  So the set contains the robust choice
   by construction, and the pick (minimax inside the set) attains it.
3. **The interval it ranges over must be the price UNCERTAINTY, not the whole
   range the frontier supports.**  Over six e-folds of price no single model is
   optimal throughout and the honest object is the *schedule* below.  Revision
   `c` conflated the two, which is what produced its 45% figure.

**Measured** (script `146`).  Coverage by resampling the panel, nominal 90%:
the exact certificate holds in all six cells (its worst is `100%`); the
version-2 band over-covers at a known price (its worst is `99.8%`); the paired
band without a certificate falls to `94.7%`.  At 200 items the certificate
excludes nothing from a 21-model zoo, which is the correct answer at 200
items.  On the
corpus: median certified set `2 of 32` at a known price and `7 of 32` with the
price known to a doubling, a decision on `100%` and `97%` of frontiers.

A certified set is **larger** than a band wherever the band was not a
confidence statement.  That is the honest trade, and it is the only version of
"maximum correctness" that can be written as an inequality.

### F13 - the calibration grid is DESIGNED, not diagnosed  *(new in d)*

Version 2 computed, after the runs were spent, the items `N_eff` would have
needed and the factor the grid should have been stretched by.  The grid width
is a design variable and the inequality that fixes it needs nothing known only
afterwards:

```
h >= sqrt( sqrt(6) sd(l) / (tol kappa) ),     sd(l) = sqrt(Q/((1-Q) n)),
kappa = s^2 t    at the operating point, where N_eff = 1/t = s^2/kappa,
```

so items, accuracy, an expected slope and the grain are enough -- three of them
counted from an evaluation already run, and the grain declared only because
this is the level that derives a price.  A **price** is a weaker target
than `N_eff`: a relative error `tol` in `lambda` needs only `2 tol` in `kappa`,
so the grid may be `sqrt(2)` narrower.  Both are reported.

**Measured** (script `147`).  The prospective verdict agrees with the
after-the-fact one on `83%` of the frontiers with a public item count, with the
error that matters -- declaring a grid feasible that the data will not support
-- at `16.7%`.  The corpus's real defect is now visible: the median published
grid is `h = 0.227` in log-resource on the 36 frontiers with a public item
count (`0.411` over all 117) where a price needs `1.17` and a channel
count `1.65`, both medians over all 117 -- factors of `5.8x` and `8.3x` against
the public-item-count grid.  Stated as a design, five
calibration points must span a resource ratio of about **110x** for a price and
**777x** for `N_eff`, cheapest to dearest, again over all 117.  (`F20` below
quotes `850x` for the same span: that is script `152`'s median over the 36
frontiers with a public item count and for `kappa` to a quarter, a different
target and a different population from this `777x`.)  And the sharpest number in this
revision: on **not one** of those 36 frontiers is the derived price identified
at the grid and items it was measured with.  The demo curves confirm the rule
end to end against a known truth -- the `x2` grid is refused for a price and
the `x4` grid accepted, which is exactly what the tool reports after the fact.

### F14 - an instance-level stopping rule for the number of networks  *(new in d)*

"Four networks" is a corpus median; a deployment needs a rule for its own
frontier.  Under Law I, `S = -ln eps` is concave in log-resource, so the curve
lies below the extension of every secant: between two adjacent candidates the
upper envelope is the minimum of two lines, and the most an **unseen** network
could be worth is the maximum of `S - lambda x` on that envelope -- a closed
form, no search, no fit.  Below `rho_2` the search is finished.

**Measured** (script `148`): **0% false stops** at every K on all 117
frontiers, and a certified stop needs a median of **5** networks where the
measured truth needs 3.  The gap is the price of proving that nothing unseen
helps.  With several resource axes the candidates do not lie on one curve, the
envelope is unavailable, and the rule says so and falls back to the
assumption-free neighbour gap.

### F15 - where the next network goes  *(new in d)*

The same bound names the resource: the next candidate belongs at the
log-resource where the worst case is attained.  Greedy sequential placement
beats both equal-arc and equal-log-resource at every K (table in section 0),
and the selector prints the resource to try next whenever the verdict is KEEP
GOING.

### F16 - the O(h^2) slope bias is corrected  *(new in d)*

The plug-in slope on the plane is biased low in magnitude by `O(h^2)`, so the
price reads low and the operating depth deep.  Version 2 printed the direction.
A bias of known order can be removed: read the plane at step `h` and at step
`2h` and combine, `(4 A_h - A_2h)/3`.  This needs at least seven readings (four
on the coarse grid); with fewer the correction is refused rather than
attempted.

**Measured** (script `149`).  Noise-free, against the known canonical price of
the cure chart, the error falls by a factor **8.8** (at `h = 0.693`, from
`-0.0036` to `-0.0001`).  With binomial noise the correction always lowers bias
and always raises variance, so it lowers RMSE in only 1 of 6 (grid, item-count)
cells -- which is why the tool applies it **adaptively**, only when the
estimated bias exceeds the price's own noise.  That rule lands within `17%` of
the better of the two estimators in every cell, knowing nothing about the
truth.  On the archive the correction raises the price on `73%` of frontiers
with seven or more readings, the direction the bias predicts, by a median
`+19%`.

### F17 - nothing degrades silently  *(new in d)*

One line per input, each saying what its absence costs, and a `level` that is
the strongest statement the inputs support: `CERTIFIED` with per-item outcomes,
`CERTIFIED (normal)` with only `n_test`, `CERTIFIED (price range only)` when
the price is not identified, `UNCERTIFIED` with no `n_test` at all.  A missing
`n_test` used to fall back to `rho_2` without a word.  `q0` is recorded with
its convention, and the output states that it cancels from every ranking and
every derivative while moving the floor and the depth.

### A7 - the ENVELOPE threshold is calibrated, not guessed  *(new in d)*

`OUT_OF_CLASS` rests on the global chord and needs no threshold.  `ENVELOPE`
rested on a constant: more than **half** the wide triples violating convexity.
One half was a guess.  The share of violating triples is not a fixed quantity
-- a dense grid on few items violates convexity from noise alone -- so the
threshold belongs to the instance and can be computed: simulate the frontier's
own **convex minorant** (a curve obeying Law I exactly, as close to the
readings as a convex curve can be from below) with the frontier's own binomial
noise, and take the 95th percentile of the violating share.

**Measured** (script `150`): the calibrated threshold is a median `0.183`, not
`0.5`, so the constant was far too permissive.  `15 of 36` verdicts change, all
from `IN_CLASS` to `ENVELOPE`, and every distribution-shift frontier is now
flagged (`100%` against `59%` elsewhere).  The statistic to report is the
**excess over the instance's own null** (`+0.097` for shift, `+0.034`
elsewhere), not the raw share.  The cost is honest: second-order readings are
withheld on `69%` of frontiers instead of `28%`.

### F18 - price-free domination, tried first  *(new in e)*

If candidate `i` is no dearer than `j` on every priced axis and is certified no
worse on accuracy, then `sigma_i >= sigma_j` for **every** non-negative price
vector.  The comparison needs no grain, no sweep and no `lambda*`: it is
`F12`'s certificate with the cost difference set to zero, so `R0 = 1` and the
per-item variable is the McNemar one, `v = 1{j fails} - 1{i fails}` in
`{-1,0,1}`.

This is the strongest statement the pipeline can make and the cheapest, so it
runs first.  **Measured** (script `151`): it fires on only 4 of 36 archived
frontiers -- because an archived frontier has already been Pareto-filtered, so
the dominated points were removed before publication.  On a raw candidate list
it is often the only test needed, and the worked example shows it retiring a
submission with no price at all.  This is also the whole of the batch-size
question: an axis that moves cost without moving accuracy gives `s = 0` and no
price, but it does not need one.

### F19 - the ranking is a sequence of TIERS  *(new in e)*

"Rank them all by effectiveness" has a dishonest answer and an honest one.  The
dishonest one is a total order on `sigma`, which invents distinctions the data
cannot support -- Thm. *covering* says `13.8` consecutive models fit inside one
indifference band on a median published frontier.  The honest one is the
partial order the certificates establish, laid out in layers: tier 1 is what
nothing is certified to beat, tier 2 the same after removing tier 1, and so on.
Within a tier nothing is separable at the items in hand; between tiers
everything is.  Pairs certified in both directions are discarded and reported,
each being an error of probability at most `alpha/(K-1)`.

**Measured** (script `151`, 36 frontiers with a public item count, treating
each frontier's Pareto points as the candidate list): a median of **4 tiers**
with a top tier of **19 of 32** candidates, so **44%** of a published frontier
is certifiably out of contention, and **0 of 36** frontiers produced a
contradictory pair.  A tier is not a tie: the output states, per pair, the
items that would separate it, and that number is often large.

### F20 - selection-grade precision: how sharp the automatic price must be  *(new in e)*

`F13` asks what grid `kappa` needs to be measured to a relative tolerance.
That is the wrong target for a selector and a far stricter one.  What a
selector needs is that the derived price's own interval be narrower than the
interval of prices over which the winner does not change -- a candidate's
**reign** -- and reigns are a property of the candidates in hand, computable at
no cost.  Since `lambda* = sqrt(kappa/t)`,

```
sd(lambda)/lambda = (1/2) sd(kappa)/kappa,   sd(kappa) = sqrt(6) sd(l)/h^2,
```

so a factor `F` of sharpening is `sqrt(F)` in sweep width and `F^2` in items:
**the sweep is the cheap route by a wide margin.**

**Measured** (script `152`, the same 36 frontiers): the derived price's 90%
interval is a median **4.92 doublings** against a median reign of **1.00**, so
the median sharpening factor is `F = 4.0` counting the **7** frontiers already
sharp enough at `F = 1`, and `F = 6.5` over the **29** that are not.  The sweep
those 29 need is `sqrt(6.5) = `**2.5x wider** -- five points spanning about
**5x** in resource -- against the **850x** that `kappa` to a quarter would
demand.  (`sqrt(4.0)` is `2.0`; the `2.5x` belongs to the 29 and not to the
median over all 36, and earlier revisions of this file paired the two.)  Seven
of the 36 frontiers already pass.  **Nothing in the theory blocks the automatic
price; what blocks it is that nobody designs a calibration sweep for a second
difference.**

### Where the sweep comes from: input size  *(new in e)*

The sweep has to come from somewhere that costs no training.  Feeding **one
trained model** inputs of several sizes is such a place: fixed weights, one
inference pass per size, and because compute grows as the pixel count a `2.7x`
range of resolution is a `7x` range of resource -- more than `F20` asks for.

Whether such a sweep is a frontier in the theory's sense is empirical, because
the axiom is about one mechanism with one clock and a larger input is not
obviously the same move as more channels.  **Measured** (script `153`): the
archive's own input-pixels frontiers separate the two ways of building one --

| how the points were made | Law I |
|---|---|
| **retrained** at each input size | **OUT_OF_CLASS 3 of 3** |
| **fixed backbone**, only the input varies | IN_CLASS 2, ENVELOPE 1 |

-- which is what "one clock" predicts: retraining changes the mechanism between
points.  Two rules follow from the geometry: **stay at or below the training
resolution** (above it accuracy falls with resource, so those points are not
Pareto and no frontier passes through them), and **record the resource you will
pay for** -- measured time on the target device, not the pixel count, because
pixels-to-time differs per architecture and the calibration axis must be the
axis the candidates are scored on.  Caveat, stated because it matters: `n = 6`,
and `linearProbing` is a frozen backbone re-probed per resolution rather than
one deployed model at several input sizes.  The verdicts point the right way;
they are not a proof.  A deployment runs the gate on its own sweep and finds
out, which is what the gate is for.

---

### F21 - the break-even ladder: the simplest statement, and the sharpest  *(new in e)*

Two candidates, one evaluation each, and the per-item outcomes behind them.
Then

```
sigma_D > sigma_C  <=>  lambda < lambda_be := ln(eps_C/eps_D) / (x_D - x_C),
```

so a single **break-even price** decides, and certifying it is `F12` run as a
function of `lambda`: the certificate for "the dearer wins" falls monotonically
in `lambda` and the one for "the cheaper wins" rises, so a bisection returns an
interval -- below it the dearer candidate is certified better, above it the
cheaper one is, and inside it the items in hand cannot separate them.  **No
sweep, no grain, no derived price, no Law I.**  For K candidates the adjacent
pairs form a price ladder, and each rung is the price at which the next model
up stops paying for itself.

**Measured** (script `154`, 1,309 adjacent Pareto pairs on the frontiers with a
public item count):

| | |
|---|---|
| undecided interval, per pair | median **0.87** doublings (p90 `3.12`) |
| the **derived** price's own interval | median **4.86** doublings |
| the ladder is sharper than the derived price | on **96%** of pairs |

So the smallest statement is also the sharpest available one.  For a deployment
that can say *anything* about its own price -- even only "below 0.5 nats per
e-fold" -- the ladder decides from one evaluation per candidate, with no sweep.
The sweep earns its keep only when nobody will name a price: then
`lambda* = sqrt(kappa/t)` supplies one, and `F20` says how wide the sweep must
be to beat these rungs.

### F22 - the price envelope: who can be optimal, without knowing the frontier  *(new in e)*

`F21`'s identity is algebra on two measured points, so it holds whether or not
either point lies on a frontier: `sigma = S - lambda x` is a function of
`(eps, x)` alone, with no appeal to Law I, convexity or an envelope.  What does
**not** survive off-frontier candidates is chaining break-evens between *cost
neighbours*: a dominated candidate, or one merely sitting in a dent of the
measured staircase, breaks the chain.

The repair needs no knowledge of which candidates are efficient.  Each
candidate is a straight line in the price,

```
sigma_i(lambda) = S_i - lambda x_i ,
```

so the best attainable score is the **upper envelope of K lines** -- convex,
piecewise linear, one sort and one stack pass.  Its pieces name the winner at
each price, its breakpoints are the break-evens that matter, and any candidate
absent from it is provably never the answer at any price.

That set is **stricter than the Pareto set**, and that is the measurement.  A
Pareto-efficient candidate in a dent maximises `sigma` for no `lambda` at all.

**Measured** (script `155`, all 117 archived frontiers, every one of them
already Pareto-filtered):

| | |
|---|---|
| median Pareto points | **11** |
| median points a linear price can ever select | **5** |
| median share never optimal at any price | **62.5%** (mean 62.3%, max 95.0%) |
| of those, median sitting in a *dent* | **6** per frontier |
| rungs per frontier, built on the envelope | **5** on the 36 frontiers with a public item count (**4** over all 117, script `156`), median undecided interval **0.43** doublings |
| rungs decided on at least one side | **100%** |

So `"is this model on the frontier?"` is the wrong question, and a Pareto
frontier over-states the viable set by a factor of two or more.  The envelope
answers the right question -- *can any price select this model?* -- and answers
it without identifying any frontier.  Building the ladder on its breakpoints
rather than on cost neighbours also halves the undecided interval, from `0.87`
doublings to `0.43`.

### Which axes can be a sweep, and which cannot

The sweep must be **one mechanism**, must **move accuracy**, and must be
measured in the **resource you will pay for**.  Three requirements, and most
candidate axes fail one of them.

| axis | one mechanism? | moves accuracy? | priced at inference? | verdict |
|---|---|---|---|---|
| **k crops / TTA samples** | yes | yes | yes | **best footing**: a count axis, where the axiom is a *theorem* of exchangeability and the floor is two measured numbers |
| **input size / resolution** | yes (fixed weights) | yes | yes | **best for vision**: no training, `2.7x` of resolution is `7x` of resource |
| **sequence length / context** | yes | yes | yes | the same, for text |
| **denoising / diffusion steps** | yes | yes | yes | good, and a wide span comes free (4 to 100 steps) |
| **early exits** (multi-exit net) | yes | yes | yes | good where the network has them |
| **epochs along ONE run, fixed LR** | yes | yes | **no** | a valid frontier in *training* cost; the price it yields is nats per e-fold of **training** time and does **not** transfer to inference |
| **learning-rate sweep** | **no** | yes | no | a hyperparameter envelope, not one clock |
| **retrained width/depth family** | **no** | yes | yes | an envelope; this is what most `OUT_OF_CLASS` corpus cases are |
| **quantization levels** | yes | barely, then a cliff | yes | too few points and not convex |
| **pruning without fine-tuning** | yes | yes | yes | the mechanism changes at each level; risky |
| **batch size** | yes | **no** | yes | `s = 0`, so no price -- but `F18`/`F21` need none |

Two measurements from the archive settle the first column.  Training-resource
axes pass Law I *more* often than inference axes (`35` in class against `4` out,
versus `51` against `20` plus `5` envelopes), so an epoch-like axis is a fine
frontier -- the problem is never the shape, it is the **coordinate**.  And the
input-pixels frontiers separate "one mechanism" from "many": retrained at each
size, out of class on `3` of `3`; a fixed backbone with only the input varying,
in class or envelope on `3` of `3` (script `153`).

**The special case worth naming.**  If your candidates differ *only* in weights
-- same architecture, different epochs, different learning rate -- then their
inference cost is **identical**, so `x_i = x_j` and

```
sigma_i - sigma_j = ln(eps_j / eps_i)    for every lambda.
```

There is no trade-off to price.  Take the most accurate, certified by the
price-free test of `F18`, and no sweep, grain or price enters anywhere.  The
theory is not needed for that comparison; it is needed only when the candidates
differ in what they cost to run.

### F23 - none of price, grain or `q0` is an input of the decision  *(new in e)*

Three quantities the selector appeared to require are inputs of neither the
envelope, the ladder nor the price-free test.

* **The price.** The envelope is `max_i (S_i - lambda x_i)` read as a *function*
  of `lambda`, so it is computed before any price is known; its breakpoints are
  the break-evens and the rung at `lambda = 0` is the price-free test.  Because
  its pieces partition `(0, infinity)`, the answer is **complete** without a
  price: whatever the realised price, exactly one candidate is named.
* **The grain.** It enters only through `lambda* = sqrt(kappa/t)`, the operating
  point `N_eff = 1/t` and the wall and depth bounds.
* **The chance level.** `ln eps = ln(1-Q) - ln(1-q0)`, a constant shift of `l`,
  so `q0` cancels from every difference of `sigma`.  **Measured** over `q0` in
  `{0, 0.001, 0.01, 0.1, 0.5}` on all 117 archived frontiers: the largest
  deviation in any `sigma` difference is `1.8e-15`, in any break-even
  `5.0e-14`, and **no** envelope winner set changes (script `156`).

So the minimal invocation is one file, or none at all:

```bash
select_model.py measurements.csv
select_model.py --axis inference_time_ms --model name,acc,resource,n_test ...
```

and `--grain` is required only with `--calibration`.

### F24 - the input consistency gate  *(new in e)*

The certified interval of `F21` is read off a pair's 2x2 table; the point
break-even is read off the two accuracies.  If those come from different places
-- a stale aggregate file beside a fresh table, a mistyped accuracy on the
command line -- the tool would print a point outside its own certified
interval.  Nothing downstream can detect that, so it is checked at the door: a
2x2 table fixes both marginal error rates and the item count exactly, and an
item vector fixes the accuracy exactly, so the declared values must match to
within the rounding of the accuracy as written.  They are compared, a
disagreement is **refused** with the offending counts named, and `--force`
overrides.

**Measured** (script `157`): the four ways of handing the selector one
evaluation -- `measurements.csv`; `candidates.csv --items items.csv`;
`candidates.csv --pairs pairs.csv`; `--model`/`--pair` on the command line --
produce **identical** decision blocks, and perturbing one accuracy by 100 items
while leaving its table alone makes the gate fire.  Writing that test found a
real defect: the tier order was taken from set iteration, so the reported
ranking depended on the process's hash seed.  It is now read in input order and
broken on the envelope position, the error and the name.

---

## 2. What is NOT fixed, because it cannot be

These are theorems about the object, not defects of the implementation.  The
selector states each one where it bites instead of silently assuming it away.

1. **The grain must come from outside.** Theorem *noinvariant*.  `F1` removes
   the silent default; it cannot remove the requirement.
2. **Near-neighbour ranking is delicate everywhere a frontier has been
   measured.** Theorem *covering*.  The honest output is a set, which is why
   `F12` exists; `F11` makes it as small as the data allow.
3. **The two walls are tangent at the floor**, so no test built on their gap
   has power there -- exactly the regime of a saturated production model.
4. **Divisibility is undecidable at compute-axis precision.**  The remedy is
   the count axis (`F6`), not a deeper corpus.
5. **`alpha` is the unit of its axis**, so ratios across axes transfer best --
   but not only ratios.  On the 41-cell panel (`scripts/159`), `ln alpha` has
   `eta^2 = 0.63` by axis and `0.65` by task, and an additive fit in both
   reaches `R^2 = 0.98`, so task identity adds `0.34`; borrowing an axis's
   median `alpha` from other tasks costs a median `24%` relative error and
   lands within a factor of two on `94%` of cells.  Every axis must be measured
   where it is deployed, and so, to within a factor of two, must every task.
6. **The floor is classically non-identified** on a compute axis.  `F6`
   replaces it with a measured bracket; absent a count axis, `J` is not
   reported.
7. **Independence is refuted**: nested difficulty, `Pr[both fail] = kappa
   min(h_i,h_j)`, `kappa = 0.835`.  There is therefore no theory here for
   ensembling, routing or cascades, and `F7`'s inequality is the detector.
   (`F11` turns the same fact into an asset for *selection*: the correlation
   that breaks ensembling tightens every comparison.)
8. **No canonical point exists** unless the two slowest channels separate by
   `r >= 3 + 2 sqrt 2 = 5.83`, nor for a compositional frontier with
   `eps_inf > 1/e`.  The general instruction -- minimise `|ln N_eff|` -- is
   what the tool uses.
9. **Prices are per e-fold.**  Fixed costs, GPU-tier step costs, latency-SLO
   cliffs, queueing and batching are not representable in
   `sigma = S - lambda.x`.
10. **The theory is static**: no drift, no retraining cadence, no online
    update.
11. **`q0` must exist.**  `eps = (1-Q)/(1-q0)` needs a chance level, so
    perplexity, BLEU, mAP and RMSE are outside the coordinates as they stand.
    (`q0` shifts `ln eps` by a constant, so it cancels from every derivative
    and from every difference of `sigma`.)
12. **`N_eff` is a second difference**, so three readings are the floor:
    single-shot selection is impossible and the calibration re-runs are
    irreducible.  `F13` says what they must span; it cannot make them
    unnecessary.
13. **On published compute-axis windows the curve is noise-dominated** (median
    straightness `R^2 = 0.22`) and has no power against a design-matched null.
    `F13` quantifies the design that would change this; the count axis (`F6`)
    is the only axis where the object is detectable today.

### Still open, and not theorems

These were named in the audit of 2026-10-08d and are **not** fixed here.

* **Candidate generation.**  `F15` says where the next network should sit on the
  resource axis; it does not say how to build a network that sits there.
* **Eliciting the grain.**  `t` is mandatory and out-of-band, and nothing maps
  an SLA, a retry policy or a human-review rate onto it.  `--grain t1:t2` with
  the exact worst-case price is a bandage, not a measurement.
* **Held-out validation.**  Everything here is measured on the same 117
  frontiers and one per-item panel, 8 corpus labels -- seven benchmark
  families, with vision split into `vision` and `vision-shift` -- dominated by
  tabular (43) and
  HPO (29).  "Works for absolutely different cases" remains an in-sample
  statement until a prospective corpus is run.
* **A convention for `q0`.**  It is now recorded and its invariance is stated;
  which convention to use is still the user's choice.
* **A tighter certificate.**  The empirical-Bernstein bound is honest but not
  sharp: it needs several times the items an asymptotic normal test would for
  the same true margin, the factor depending on the margin and the variance
  (no corpus-wide figure is archived, so none is quoted here).  A betting-style confidence sequence would recover
  most of that gap and be anytime-valid as well, which would let a deployment
  stop collecting items adaptively.  No new data and no new assumptions; it is
  simply not done here.
* **Non-linear cost.**  `sigma = S - lambda.x` is linear per e-fold, so GPU
  tiers, fixed costs, SLO cliffs and batching are unrepresentable.  The
  *certificate* never uses linearity -- it needs only the number
  `c_i - c_j` -- so the certified set, the ranking and the schedule would carry
  over to an arbitrary deterministic cost function; only the derived price and
  the plane geometry need the linear form.  Not done here.

---

## 3. Running it

```bash
cd examples/model_selection
python3 make_protocol_data.py             # the protocol's five input files

# THE PROTOCOL: five measurements of one model, one of each candidate,
# no price set by hand
python3 select_model.py measurements.csv            # Level 0: one file, nothing else
python3 select_model.py candidates.csv --pairs pairs.csv   # the same, 21 lines
python3 select_model.py candidates.csv --grain 1 \
    --calibration sweep.csv --calibration-axis inference_time_ms \
    --items items.csv                              # Level 2: the derived price

# aggregates only: the rungs fall back to an unpaired normal interval, and say so
python3 select_model.py candidates.csv

# the earlier five-candidate demo, kept so versions 2 and 3 compare.  It passes
# --q0 0.1 because its generated data has that chance level and the FLOOR and
# DEPTH readings use it; no ranking anywhere depends on it (F23).  --price
# prices the second axis, which has no calibration curve of its own (F5).
python3 make_demo_data.py
python3 select_model.py models.csv --grain 1 \
    --calibration calibration_wide.csv --calibration-axis inference_time \
    --price param_count=0.08 --q0 0.1 \
    --items items_demo.csv --alpha 0.10

# the price known to a doubling instead of derived, and a count axis
python3 select_model.py models.csv --grain 1 \
    --calibration calibration_wide.csv --calibration-axis inference_time \
    --price param_count=0.08 --q0 0.1 --items items_demo.csv \
    --price-interval 0.35:0.55 --passk passk_demo.csv --retry-budget 3

# aggregates only, no per-item file: the certificate falls back to the normal
# unpaired one and says so
python3 select_model.py candidates.csv --rho 0.5
```

```bash
cd scripts                   # none of these needs the network
python3 139_class_gate_corpus.py      # F2: the gate on all 117 frontiers
python3 140_precision_gate_corpus.py  # F3: what the precision allows
python3 141_count_axis_panel.py       # F6, F10: the count axis
python3 142_undersizing_measured.py   # F7: the under-sizing, measured
python3 143_selector_audit.py         # F8: policy regret, and K from thinning
python3 144_multiresource_price.py    # F5: one price per axis
python3 145_paired_band_panel.py      # F11: the paired band, on 420 pairs
python3 146_certified_set.py          # F12: coverage, measured
python3 147_design_rule.py            # F13: the grid, designed in advance
python3 148_sequential_design.py      # F14, F15: how few, where, when to stop
python3 149_slope_bias_corrected.py   # F16: the h^2 bias, removed
python3 150_envelope_threshold.py     # A7: the threshold, calibrated
python3 151_certified_ranking.py      # F18, F19: price-free domination, and the tiers
python3 152_selection_grade_price.py  # F20: how sharp the automatic price must be
python3 153_input_size_axis.py        # where the sweep comes from
python3 154_break_even_ladder.py      # F21: the price ladder, and how sharp it is
python3 155_price_envelope.py         # F22: who can be optimal at all
python3 156_measurement_only.py       # F23: q0 cancels exactly; Level 0 on the corpus
python3 157_input_routes.py           # F24: the four input routes agree; the gate fires
python3 158_floor_share_null.py       # the zero-floor null for the fitted floor share
python3 159_exponent_decomposition.py # does the exponent transfer? axis against task
python3 160_arc_deficit_gauge.py      # the arc deficit: the shares read at the crossover
python3 161_d1hat_spacing.py          # what node spacing does to the plug-in rung D1hat
```

`141` must run before `142`, which reads the count axis it writes; `158` needs
`81` and `133`, and `159` needs `47`.  A transcript
of `139`--`160` is in `results/practical_pipeline_log.txt`, and `161`'s own run
in `results/log_161.txt`; twenty-three scripts in all.

### Files added in revision e

| path | what |
|---|---|
| `examples/model_selection/make_protocol_data.py` | generates the protocol's five files from the cure chart |
| `examples/model_selection/sweep.csv` | one model, fixed weights, five geometric input sizes, 7.2x in time |
| `examples/model_selection/measurements.csv` | **the one file Level 0 needs**: 50,000 per-item outcomes with the resource on its own row; accuracy and `n_test` derived |
| `examples/model_selection/pairs.csv` | the same information as 21 pairwise `2x2` tables, identical certificate |
| `examples/model_selection/candidates.csv` | seven candidates, aggregates only, one deliberately dominated |
| `examples/model_selection/items.csv` | the item matrix in the two-file form |
| `scripts/151_certified_ranking.py` | F18, F19 on the corpus |
| `scripts/152_selection_grade_price.py` | F20: the reign-width design rule |
| `scripts/153_input_size_axis.py` | the input-pixels evidence |
| `scripts/154_break_even_ladder.py` | F21: the ladder, on 1,309 adjacent pairs |
| `scripts/155_price_envelope.py` | F22: the envelope, on all 117 frontiers |
| `scripts/156_measurement_only.py` | F23: the q0-invariance check and Level 0 measured |
| `scripts/157_input_routes.py` | F24: the four input routes agree, and the consistency gate fires |
| `results/table_certified_ranking*.csv` | F18, F19 |
| `results/table_selection_grade*.csv` | F20 |
| `results/table_input_size_axis.csv` | the input-size axis |
| `results/table_break_even*.csv` | F21 |
| `results/table_price_envelope*.csv` | F22 |
| `results/table_measurement_only*.csv` | F23 |

New in `pipeline_core.py`: `certified_pareto` (F18); `certified_ranking` (F19);
`reign_widths`, `design_for_selection` (F20); `certified_break_even`,
`break_even_chain` (F21); `price_envelope`, `break_even_envelope` (F22); `tiers_from_relation`,
`decide_from_measurements` (F23); `check_consistency` (F24).

### Files added in revision d

| path | what |
|---|---|
| `examples/model_selection/items_demo.csv` | per-item outcomes for the five candidates on one shared test set, from which their accuracies are counted |
| `scripts/145_paired_band_panel.py` | F11, on all 420 pairs of the co-failure panel |
| `scripts/146_certified_set.py` | F12: coverage by resampling, and the corpus |
| `scripts/147_design_rule.py` | F13: the prospective grid rule, validated |
| `scripts/148_sequential_design.py` | F14, F15: minimum networks, placement, stopping |
| `scripts/149_slope_bias_corrected.py` | F16: Richardson, against a known truth |
| `scripts/150_envelope_threshold.py` | A7: the calibrated ENVELOPE threshold |
| `results/table_paired_band_*.csv` | F11 |
| `results/table_certified_*.csv` | F12 |
| `results/table_design_rule*.csv` | F13 |
| `results/table_min_networks*.csv`, `table_schedule.csv`, `table_stopping_rule.csv` | F14, F15 |
| `results/table_slope_bias_*.csv` | F16 |
| `results/table_envelope_threshold*.csv` | A7 |

New in `pipeline_core.py`: `failure_stats`, `phi_correlation`, `sd_of_gap`,
`required_n_gap` (F11); `contingency`, `certify_domination_counts`,
`certify_domination`, `eb_lower`, `certified_set`, `certified_schedule` (F12);
`design_calibration` (F13); `envelope_upper`, `stopping_rule`,
`worst_case_regret`, `greedy_subset` (F14, F15); `richardson_price` (F16);
`inputs_audit` (F17); `convex_minorant`, `envelope_share_null` (A7).  Still
stdlib only, still the same module scripts `139`-`156` import -- `157` drives the
CLI as a subprocess and `158`-`159` are independent -- so the archive's
numbers and the example's cannot drift.
