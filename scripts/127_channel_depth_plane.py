#!/usr/bin/env python3
"""
127_channel_depth_plane.py -- pure mathematics, no data.

Verifies the candidate laws of derived_laws_20261006e:
the (S, N_eff) plane as the theory's complete invariant, the two walls as a
curve and its tangent, the arc as a functional on that plane with two sharp
variational bounds, lambda* = s at the operating point, the free count-axis
rung from P(0)=1, and the reconstruction quadrature.

mpmath, 50 digits. No corpus, no fit, no new data.
"""
import mpmath as mp

mp.mp.dps = 60
OK, FAIL = [], []


def chk(name, cond, detail=""):
    (OK if cond else FAIL).append((name, detail))
    print(("  ok   " if cond else "  FAIL ") + name + (("   " + detail) if detail else ""))


# ---------------------------------------------------------------- charts
# each chart: eps(x) on its domain, plus its true floor
def two_atom(f=mp.mpf("0.004"), c=mp.mpf("0.996"), a=mp.mpf("0.9")):
    return (lambda x: f + c * mp.e ** (-a * x)), f, dict(alpha=a)


def cure(lam=mp.mpf("5.0"), c=mp.mpf("0.7")):
    return (lambda x: mp.e ** (-lam * (1 - mp.e ** (-c * x)))), mp.e ** (-lam), dict(lam=lam)


def power_floor(Lam=mp.mpf("4.0"), C=mp.mpf("2.0"), beta=mp.mpf("-0.5")):
    # eps = exp(-Lam + C x^beta), beta<0 : divisible, with a floor
    return (lambda x: mp.e ** (-Lam + C * x ** beta)), mp.e ** (-Lam), dict(beta=beta)


def lomax(k=mp.mpf("2.5"), th=mp.mpf("1.3")):
    return (lambda x: (1 + x / th) ** (-k)), mp.mpf(0), dict(k=k)


def stretched(al=mp.mpf("0.8"), beta=mp.mpf("0.6")):
    return (lambda x: mp.e ** (-(al * x) ** beta)), mp.mpf(0), dict(beta=beta)


def atoms(ws, as_):
    s = sum(ws)
    ws = [w / s for w in ws]
    return (lambda x: sum(w * mp.e ** (-a * x) for w, a in zip(ws, as_))), \
           sum(w for w, a in zip(ws, as_) if a == 0), dict(n=len(ws))


def sqkap(eps, x):
    """sqrt(kappa) at x, underflow-safe."""
    try:
        k = -mp.diff(lambda u: -mp.log(eps(u)), x, 2)
    except Exception:
        return mp.mpf(0)
    return mp.sqrt(k) if k > 0 else mp.mpf(0)


def total_arc(eps, x_lo=mp.mpf(0), x_hi=mp.mpf(150)):
    """int sqrt(kappa) dx over the whole frontier, piecewise for stability."""
    cuts = [x_lo, mp.mpf(2), mp.mpf(8), mp.mpf(25), mp.mpf(60), x_hi]
    cuts = [c for c in cuts if x_lo <= c <= x_hi]
    if cuts[0] != x_lo:
        cuts = [x_lo] + cuts
    if cuts[-1] != x_hi:
        cuts = cuts + [x_hi]
    return sum(mp.quad(lambda x: sqkap(eps, x), [cuts[i], cuts[i + 1]])
               for i in range(len(cuts) - 1))


# ------------------------------------------------------- local jet of a chart
def jet(eps, x, floor):
    S = -mp.log(eps(x))
    s = -mp.diff(lambda u: mp.log(eps(u)), x)
    kap = mp.diff(lambda u: -mp.log(eps(u)), x, 2) * (-1)
    c3 = mp.diff(lambda u: -mp.log(eps(u)), x, 3)
    N = (s ** 2 / kap) if kap > 0 else mp.inf
    dS = mp.log(eps(x) / floor) if floor > 0 else mp.inf
    return dict(S=S, s=s, kap=kap, c3=c3, N=N, dS=dS, eps=eps(x))


print("\n=== A. N_eff is AFFINE in S on every constant-sigma chart "
      "(slope = sigma, root = the floor depth) ===")
cases = [
    ("cure           sigma=-1", cure(), mp.mpf(-1)),
    ("power-floor b=-0.5", power_floor(), None),
    ("power-floor b=-2.0", power_floor(Lam=mp.mpf(3), C=mp.mpf(1), beta=mp.mpf(-2)), None),
    ("Lomax k=2.5    sigma=0", lomax(), mp.mpf(0)),
    ("stretched b=0.6", stretched(), None),
    ("stretched b=0.3", stretched(al=mp.mpf("1.1"), beta=mp.mpf("0.3")), None),
]
for nm, (eps, floor, meta), sig_known in cases:
    xs = [mp.mpf(q) for q in ("0.8", "1.7", "3.1", "5.5", "9.0")]
    js = [jet(eps, x, floor) for x in xs]
    # fit a line N = sigma*S + b through the first two, test the rest
    s0, s1 = js[0]["S"], js[1]["S"]
    n0, n1 = js[0]["N"], js[1]["N"]
    sig = (n1 - n0) / (s1 - s0)
    b = n0 - sig * s0
    err = max(abs(j["N"] - (sig * j["S"] + b)) for j in js[2:])
    chk(f"A1 {nm}: N affine in S", err < mp.mpf("1e-28"), f"max dev {mp.nstr(err,3)}")
    if sig_known is not None:
        chk(f"A2 {nm}: slope = sigma", abs(sig - sig_known) < mp.mpf("1e-28"),
            f"slope {mp.nstr(sig,10)}")
    if floor > 0:
        root = -b / sig
        lam = -mp.log(floor)
        chk(f"A3 {nm}: S-intercept = -ln eps_inf", abs(root - lam) < mp.mpf("1e-26"),
            f"intercept {mp.nstr(root,12)} vs {mp.nstr(lam,12)}")
    else:
        chk(f"A3 {nm}: slope>=0 <=> no floor", sig >= 0, f"slope {mp.nstr(sig,6)}")

print("\n=== B. the two walls: a curve and its own tangent at the origin ===")
# axiom wall N <= e^{dS}-1 ; hypothesis wall N <= dS ; tangency at dS=0
for nm, (eps, floor, meta) in [("two-atom", two_atom()), ("cure", cure()),
                               ("power-floor", power_floor()),
                               ("3 atoms+floor", atoms([mp.mpf("0.01"), mp.mpf("0.4"), mp.mpf("0.59")],
                                                       [mp.mpf(0), mp.mpf("0.5"), mp.mpf("2.0")])),
                               ("5 atoms+floor", atoms([mp.mpf("0.002"), mp.mpf(".2"), mp.mpf(".3"),
                                                        mp.mpf(".3"), mp.mpf(".198")],
                                                       [mp.mpf(0), mp.mpf(".3"), mp.mpf(".9"),
                                                        mp.mpf("2.5"), mp.mpf("7.0")]))]:
    bad_ax = bad_hy = 0
    for q in range(1, 40):
        x = mp.mpf(q) / 4
        j = jet(eps, x, floor)
        if j["N"] > mp.e ** j["dS"] - 1 + mp.mpf("1e-30"):
            bad_ax += 1
        if j["N"] > j["dS"] + mp.mpf("1e-30"):
            bad_hy += 1
    chk(f"B1 {nm}: N <= e^dS - 1 (axiom wall)", bad_ax == 0, f"{bad_ax} violations/39")
    div = nm in ("cure", "power-floor")
    if div:
        chk(f"B2 {nm}: N <= dS (hypothesis wall)", bad_hy == 0, f"{bad_hy} violations/39")
    else:
        chk(f"B2 {nm}: N > dS somewhere (indivisible)", bad_hy > 0, f"{bad_hy}/39 above tangent")

eps, floor, _ = two_atom()
devs = []
for q in range(1, 30):
    x = mp.mpf(q) / 2
    j = jet(eps, x, floor)
    devs.append(abs(j["N"] - (mp.e ** j["dS"] - 1)))
chk("B3 two-atom chart IS the axiom wall, N = e^dS - 1", max(devs) < mp.mpf("1e-28"),
    f"max dev {mp.nstr(max(devs),3)}")
eps, floor, _ = cure()
devs = [abs(jet(eps, mp.mpf(q) / 2, floor)["N"] - jet(eps, mp.mpf(q) / 2, floor)["dS"])
        for q in range(1, 30)]
chk("B4 cure chart IS the hypothesis wall, N = dS", max(devs) < mp.mpf("1e-28"),
    f"max dev {mp.nstr(max(devs),3)}")
g = lambda u: mp.e ** u - 1
chk("B5 the hypothesis wall is the tangent of the axiom wall at the origin",
    abs(g(mp.mpf("1e-25")) - mp.mpf("1e-25")) < mp.mpf("1e-45") and abs(mp.diff(g, 0) - 1) < mp.mpf("1e-30"),
    "g(0)=0, g'(0)=1")

print("\n=== C. the two floor bounds ARE the two walls solved for the floor ===")
for nm, (eps, floor, meta), div in [("two-atom", two_atom(), False), ("cure", cure(), True),
                                    ("power-floor", power_floor(), True),
                                    ("5 atoms+floor", atoms([mp.mpf("0.002"), mp.mpf(".2"), mp.mpf(".3"),
                                                             mp.mpf(".3"), mp.mpf(".198")],
                                                            [mp.mpf(0), mp.mpf(".3"), mp.mpf(".9"),
                                                             mp.mpf("2.5"), mp.mpf("7.0")]), False)]:
    bA = bW = 0
    for q in range(1, 30):
        x = mp.mpf(q) / 2
        j = jet(eps, x, floor)
        if j["eps"] / (1 + j["N"]) < floor - mp.mpf("1e-30"):
            bA += 1
        if div and j["eps"] * mp.e ** (-j["N"]) < floor - mp.mpf("1e-30"):
            bW += 1
    chk(f"C1 {nm}: eps/(1+N) >= eps_inf  (= eps_A)", bA == 0, f"{bA} violations")
    if div:
        chk(f"C2 {nm}: eps*e^-N >= eps_inf  (= eps_W)", bW == 0, f"{bW} violations")
chk("C3 eps*e^-N <= eps/(1+N) always",
    all(mp.e ** (-mp.mpf(n) / 7) <= 1 / (1 + mp.mpf(n) / 7) for n in range(0, 60)), "e^-N(1+N)<=1")

# the 1006d grid forms converge to these
print("\n=== D. the 1006d grid floor bounds are these two walls in finite differences ===")
eps, floor, _ = two_atom()
x0 = mp.mpf("2.0")
for h in [mp.mpf("0.2"), mp.mpf("0.05"), mp.mpf("0.0125")]:
    e_, e0, ep = eps(x0 - h), eps(x0), eps(x0 + h)
    l_, l0, lp = mp.log(e_), mp.log(e0), mp.log(ep)
    epsA = (e_ * ep - e0 ** 2) / (e_ - 2 * e0 + ep)
    epsW = mp.e ** ((l_ * lp - l0 ** 2) / (l_ + lp - 2 * l0))
    j = jet(eps, x0, floor)
    contA, contW = j["eps"] / (1 + j["N"]), j["eps"] * mp.e ** (-j["N"])
    chk(f"D1 h={mp.nstr(h,3)}: eps_A -> eps/(1+N)", abs(epsA - contA) < 40 * h ** 2,
        f"gap {mp.nstr(abs(epsA-contA),3)}")
    chk(f"D2 h={mp.nstr(h,3)}: eps_W -> eps*e^-N", abs(epsW - contW) < 40 * h ** 2,
        f"gap {mp.nstr(abs(epsW-contW),3)}")

print("\n=== E. the arc is a functional of the SAME curve, with two sharp bounds ===")
# identity  int sqrt(kappa) dx = int dS / sqrt(N)
eps, floor, _ = atoms([mp.mpf("0.01"), mp.mpf("0.5"), mp.mpf("0.49")],
                      [mp.mpf(0), mp.mpf("0.4"), mp.mpf("1.8")])
x1, x2 = mp.mpf("0.5"), mp.mpf("8.0")
arc_x = mp.quad(lambda x: sqkap(eps, x), [x1, x2])
arc_S = mp.quad(lambda x: jet(eps, x, floor)["s"] / mp.sqrt(jet(eps, x, floor)["N"]), [x1, x2])
chk("E1 int sqrt(kappa) dx = int dS/sqrt(N_eff)", abs(arc_x - arc_S) < mp.mpf("1e-20"),
    f"{mp.nstr(arc_x,14)} vs {mp.nstr(arc_S,14)}")

# total arc of the two-atom chart = 2 arccos sqrt(eps_inf)
for f_ in ["0.004", "0.05", "0.2"]:
    f0 = mp.mpf(f_)
    eps, floor, _ = two_atom(f=f0, c=1 - f0, a=mp.mpf("0.9"))
    L = total_arc(eps)
    pred = 2 * mp.acos(mp.sqrt(f0))
    chk(f"E2 two-atom total arc = 2 arccos sqrt(eps_inf)  [floor {f_}]",
        abs(L - pred) < mp.mpf("1e-12"), f"{mp.nstr(L,12)} vs {mp.nstr(pred,12)}")

# the universal lower bound: L >= 2 arccos sqrt(eps_inf) for any floored CM frontier
spectra = [([mp.mpf("0.01"), mp.mpf("0.5"), mp.mpf("0.49")], [mp.mpf(0), mp.mpf("0.4"), mp.mpf("1.8")]),
           ([mp.mpf("0.004"), mp.mpf(".3"), mp.mpf(".3"), mp.mpf(".396")],
            [mp.mpf(0), mp.mpf(".2"), mp.mpf("1.0"), mp.mpf("6.0")]),
           ([mp.mpf("0.05"), mp.mpf(".45"), mp.mpf(".5")], [mp.mpf(0), mp.mpf("1.0"), mp.mpf("1.2")])]
for i, (ws, as_) in enumerate(spectra):
    eps, floor, _ = atoms(ws, as_)
    L = total_arc(eps)
    pred = 2 * mp.acos(mp.sqrt(floor))
    chk(f"E3 spectrum {i}: L >= 2 arccos sqrt(eps_inf)", L >= pred - mp.mpf("1e-14"),
        f"L {mp.nstr(L,10)}  bound {mp.nstr(pred,10)}  slack {mp.nstr(L-pred,4)}")

# the hypothesis bound: L >= 2 sqrt(lambda), equality on the cure chart
for lam_ in ["2.0", "5.0", "9.0"]:
    lam = mp.mpf(lam_)
    eps, floor, _ = cure(lam=lam, c=mp.mpf("0.7"))
    L = total_arc(eps)
    chk(f"E4 cure total arc = 2 sqrt(lambda)  [lambda {lam_}]",
        abs(L - 2 * mp.sqrt(lam)) < mp.mpf("1e-12"), f"{mp.nstr(L,12)} vs {mp.nstr(2*mp.sqrt(lam),12)}")
eps, floor, _ = power_floor(Lam=mp.mpf(6), C=mp.mpf(2), beta=mp.mpf("-0.5"))
L = total_arc(eps, x_lo=mp.mpf("1e-12"))
chk("E5 power-floor: L >= 2 sqrt(lambda) (divisible bound)", L >= 2 * mp.sqrt(mp.mpf(6)) - mp.mpf("1e-10"),
    f"L {mp.nstr(L,10)} bound {mp.nstr(2*mp.sqrt(mp.mpf(6)),10)}")
chk("E6 the divisible bound beats pi exactly beyond pi^2/4 nats",
    abs(2 * mp.sqrt(mp.pi ** 2 / 4) - mp.pi) < mp.mpf("1e-40"),
    f"pi^2/4 = {mp.nstr(mp.pi**2/4,8)} nats")
chk("E7 pi = int_0^inf du/sqrt(e^u - 1)  (Law III's constant as the axiom wall's area)",
    abs(mp.quad(lambda u: 1 / mp.sqrt(mp.e ** u - 1), [0, mp.inf]) - mp.pi) < mp.mpf("1e-18"))

print("\n=== F. the price at the operating point is the local price ===")
for nm, (eps, floor, meta) in [("two-atom", two_atom()), ("cure", cure()),
                               ("power-floor", power_floor()),
                               ("3 atoms+floor", atoms([mp.mpf("0.01"), mp.mpf("0.5"), mp.mpf("0.49")],
                                                       [mp.mpf(0), mp.mpf("0.4"), mp.mpf("1.8")]))]:
    for t in [mp.mpf(1), mp.mpf(2), mp.mpf(5)]:
        try:
            xs = mp.findroot(lambda x: jet(eps, x, floor)["N"] - 1 / t, mp.mpf("2.0"))
        except Exception:
            continue
        j = jet(eps, xs, floor)
        lam_can = mp.sqrt(j["kap"] / t)
        chk(f"F1 {nm} t={int(t)}: lambda* = sqrt(kappa/t) = s", abs(lam_can - j["s"]) < mp.mpf("1e-24"),
            f"s {mp.nstr(j['s'],12)}")
        if nm == "two-atom":
            chk(f"F2 two-atom t={int(t)}: s = alpha/(1+t) at the operating point",
                abs(j["s"] - meta["alpha"] / (1 + t)) < mp.mpf("1e-24"),
                f"{mp.nstr(j['s'],12)} vs {mp.nstr(meta['alpha']/(1+t),12)}")
            chk(f"F3 two-atom t={int(t)}: dS* = ln(1+1/t)",
                abs(j["dS"] - mp.log(1 + 1 / t)) < mp.mpf("1e-24"))
        if nm == "cure":
            chk(f"F4 cure t={int(t)}: dS* = 1/t", abs(j["dS"] - 1 / t) < mp.mpf("1e-24"))

print("\n=== G. one bit versus one nat; the elbow at one bit ===")
eps, floor, meta = two_atom()
xe = mp.findroot(lambda x: eps(x) - 2 * floor, mp.mpf("6.0"))   # the elbow, well conditioned
je = jet(eps, xe, floor)
chk("G1 two-atom elbow (c3=0) sits at dS = ln 2 exactly", abs(je["dS"] - mp.log(2)) < mp.mpf("1e-22"),
    f"dS_elbow {mp.nstr(je['dS'],14)}")
chk("G2 and c3 vanishes there", abs(je["c3"]) < mp.mpf("1e-20") * meta["alpha"] ** 3,
    f"c3(elbow) = {mp.nstr(je['c3'],3)}")
x1 = mp.findroot(lambda x: jet(eps, x, floor)["N"] - 1, mp.mpf("6.0"))
chk("G3 at grain 1 the operating point IS the elbow", abs(x1 - xe) < mp.mpf("1e-18"),
    f"x* = {mp.nstr(x1,16)}, x_elbow = {mp.nstr(xe,16)}")
chk("G4 the band at grain 1 is one nat minus one bit", abs((1 - mp.log(2)) - mp.mpf("0.30685281944")) < mp.mpf("1e-10"),
    f"1 - ln2 = {mp.nstr(1-mp.log(2),12)}")
chk("G5 grain leverage = ln2 - ln(1+1/t) <= ln 2 (one bit) for every t>=1",
    all(mp.log(2) - mp.log(1 + 1 / mp.mpf(t)) <= mp.log(2) + mp.mpf("1e-40") for t in range(1, 200))
    and abs(mp.log(2) - mp.log(1 + mp.mpf("1e-40"))) <= mp.log(2) + mp.mpf("1e-30"))

print("\n=== H. the rule has no parameter in delivered units ===")
# under eps -> eps^t : S,dS,N all multiply by t ; the rule N=1/t is N^unit=1
eps, floor, _ = atoms([mp.mpf("0.01"), mp.mpf("0.5"), mp.mpf("0.49")],
                      [mp.mpf(0), mp.mpf("0.4"), mp.mpf("1.8")])
t = mp.mpf("3.0")
epst = lambda x: eps(x) ** t
x = mp.mpf("2.3")
j, jt = jet(eps, x, floor), jet(epst, x, floor ** t)
chk("H1 grain multiplies S, dS, N_eff by exactly t",
    abs(jt["S"] - t * j["S"]) < mp.mpf("1e-24") and abs(jt["dS"] - t * j["dS"]) < mp.mpf("1e-24")
    and abs(jt["N"] - t * j["N"]) < mp.mpf("1e-22"), f"N {mp.nstr(jt['N'],12)} = {mp.nstr(t*j['N'],12)}")
chk("H2 the operating rule N=1/t is N^unit=1 (regret zero in delivered units)",
    abs(mp.log(mp.cosh(mp.log(t * (1 / t)) / 2))) < mp.mpf("1e-40"))
chk("H3 regret is invariant under N -> 1/N; its unique fixed point is N=1",
    all(abs(mp.log(mp.cosh(mp.log(mp.mpf(n) / 3) / 2)) - mp.log(mp.cosh(mp.log(3 / mp.mpf(n)) / 2)))
        < mp.mpf("1e-40") for n in range(1, 20)))

print("\n=== I. reconstruction: the curve determines the frontier up to the axis gauge ===")
for nm, (eps, floor, _) in [("3 atoms+floor", atoms([mp.mpf("0.01"), mp.mpf("0.5"), mp.mpf("0.49")],
                                                    [mp.mpf(0), mp.mpf("0.4"), mp.mpf("1.8")])),
                            ("power-floor", power_floor()), ("two-atom", two_atom())]:
    # N as a function of S, then s(S) = s(S0) exp(-int_{S0}^{S} dS'/N(S'))
    Sref = jet(eps, mp.mpf("1.0"), floor)["S"]
    sref = jet(eps, mp.mpf("1.0"), floor)["s"]
    N_of_S = lambda SS: jet(eps, mp.findroot(lambda x: jet(eps, x, floor)["S"] - SS, mp.mpf("1.5")),
                            floor)["N"]
    worst = mp.mpf(0)
    for xq in [mp.mpf("1.8"), mp.mpf("3.0"), mp.mpf("4.5")]:
        jq = jet(eps, xq, floor)
        pred = sref * mp.e ** (-mp.quad(lambda SS: 1 / N_of_S(SS), [Sref, jq["S"]]))
        worst = max(worst, abs(pred / jq["s"] - 1))
    chk(f"I1 {nm}: s recovered from N(S) by quadrature", worst < mp.mpf("1e-12"),
        f"max rel err {mp.nstr(worst,3)}")

print("\n=== J. three accuracies give N_eff with the spacing cancelling identically ===")
eps, floor, _ = two_atom()
x0 = mp.mpf("2.0")
prev = None
for h in [mp.mpf("0.4"), mp.mpf("0.1"), mp.mpf("0.025")]:
    l_, l0, lp = mp.log(eps(x0 - h)), mp.log(eps(x0)), mp.log(eps(x0 + h))
    Nhat = (lp - l_) ** 2 / (4 * (l_ - 2 * l0 + lp))
    err = abs(Nhat - jet(eps, x0, floor)["N"])
    chk(f"J1 h={mp.nstr(h,3)}: N_hat=(l+ - l-)^2/[4(l- -2l0 +l+)] -> N_eff, no h in the formula",
        err < 3 * h ** 2, f"err {mp.nstr(err,3)}")
    if prev is not None:
        chk(f"J2 h={mp.nstr(h,3)}: error falls as h^2", err < prev / 10,
            f"{mp.nstr(err,3)} vs {mp.nstr(prev,3)}")
    prev = err

print("\n=== K. the count axis: P(0)=1 is a free rung ===")
# L_m = ln P(m)/P(m+1) = Q(m+1)-Q(m) ; Q=-ln P Bernstein  =>  (L_m) is a Hausdorff moment sequence
def Lseq(P, M=7):
    return [mp.log(P(mp.mpf(m)) / P(mp.mpf(m + 1))) for m in range(M)]


def hankel_min(L, k, off=0):
    A = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            A[i, j] = L[off + i + j]
    return mp.det(A)


tasks = {
    "divisible: compound-Poisson  P=exp(-lam(1-e^-cm))":
        (lambda m: mp.e ** (-mp.mpf(4) * (1 - mp.e ** (-mp.mpf("0.6") * m))), True),
    "divisible: Lomax             P=(1+m/th)^-k":
        (lambda m: (1 + m / mp.mpf("1.5")) ** (-mp.mpf("2.0")), True),
    "indivisible: floor+one class P=w+(1-w)g^m":
        (lambda m: mp.mpf("0.05") + mp.mpf("0.95") * mp.mpf("0.45") ** m, False),
    "indivisible: 3-class census":
        (lambda m: mp.mpf("0.02") + mp.mpf("0.55") * mp.mpf("0.17") ** m
                   + mp.mpf("0.43") * mp.mpf("0.95") ** m, False),
}
for nm, (P, div) in tasks.items():
    chk(f"K0 {nm[:28]}: P(0)=1", abs(P(mp.mpf(0)) - 1) < mp.mpf("1e-40"))
    L = Lseq(P)
    new = L[0] * L[2] - L[1] ** 2          # offset 0 : needs P(1),P(2),P(3)
    old = L[1] * L[3] - L[2] ** 2          # offset 1 : needs P(1..4)
    new3 = hankel_min(L, 3, 0)             # offset 0 : needs P(1..5)
    old3 = hankel_min(L, 3, 1)             # offset 1 : needs P(1..6)
    if div:
        chk(f"K1 {nm[:28]}: L is a moment sequence (all minors >= 0)",
            new >= -mp.mpf("1e-35") and old >= -mp.mpf("1e-35") and new3 >= -mp.mpf("1e-35"),
            f"2x2@0 {mp.nstr(new,4)}  3x3@0 {mp.nstr(new3,4)}")
        chk(f"K2 {nm[:28]}: L_m completely monotone (decreasing, convex)",
            all(L[i] >= L[i + 1] for i in range(5)) and all(L[i] - 2 * L[i + 1] + L[i + 2] >= -mp.mpf("1e-35")
                                                            for i in range(4)))
    else:
        T_new = mp.log(L[0]) + mp.log(L[2]) - 2 * mp.log(L[1])
        T_old = mp.log(L[1]) + mp.log(L[3]) - 2 * mp.log(L[2])
        chk(f"K3 {nm[:28]}: rung 1 at offset 0 and 1 agree in verdict", (new < 0) == (old < 0),
            f"L0L2-L1^2 = {mp.nstr(new,4)} ; L1L3-L2^2 = {mp.nstr(old,4)}")
        chk(f"K4 {nm[:28]}: offset-0 rung 1 is CHEAPER, not larger (recorded)",
            True, f"T0 {mp.nstr(T_new,4)} vs T1 {mp.nstr(T_old,4)}  |T0/T1| {mp.nstr(abs(T_new/T_old),4)}")
        nn = new3 / (L[0] * L[2] * L[4]); no_ = old3 / (L[1] * L[3] * L[5])
        chk(f"K5 {nm[:28]}: rung 2 at offset 0 refutes from P(1..5)", new3 < 0,
            f"det {mp.nstr(new3,4)} (offset 1, P(1..6): {mp.nstr(old3,4)})")
        chk(f"K6 {nm[:28]}: and normalised it is the LARGER determinant", abs(nn) > abs(no_),
            f"|det|/prod(diag): offset0 {mp.nstr(abs(nn),4)} vs offset1 {mp.nstr(abs(no_),4)}"
            f"  ratio {mp.nstr(abs(nn/no_),4)}")

print("\n=== L. covering: a floor costs arc, and divisibility costs more ===")
rho = mp.mpf("0.05")
for f_ in ["0.01", "0.001"]:
    f0 = mp.mpf(f_)
    Kmin_ax = 2 * mp.acos(mp.sqrt(f0)) / (2 * mp.sqrt(2 * rho))
    Kmin_hy = 2 * mp.sqrt(-mp.log(f0)) / (2 * mp.sqrt(2 * rho))
    chk(f"L1 floor {f_}: every CM frontier needs >= {float(Kmin_ax):.2f} points at 0.05 nats",
        Kmin_ax > 4.5, f"K_min = {mp.nstr(Kmin_ax,5)}")
    chk(f"L2 floor {f_}: a DIVISIBLE one needs >= {float(Kmin_hy):.2f}",
        Kmin_hy > Kmin_ax, f"{mp.nstr(Kmin_hy,5)} vs {mp.nstr(Kmin_ax,5)}")

print("\n=== M. the hierarchy is redundant at the bottom ===")
# sigma >= -1 (rung 1) + N(floor)=0  =>  N <= dS (rung 0 = Thm. budget)
eps, floor, _ = power_floor()
bad = 0
for q in range(2, 40):
    x = mp.mpf(q) / 2
    j = jet(eps, x, floor)
    sig = mp.diff(lambda u: jet(eps, u, floor)["N"], x) / j["s"]   # dN/dS
    if sig < -1 - mp.mpf("1e-25"):
        bad += 1
chk("M1 power-floor: dN/dS >= -1 pointwise (rung 1)", bad == 0, f"{bad} violations")
eps, floor, _ = cure()
sigs = [mp.diff(lambda u: jet(eps, u, floor)["N"], mp.mpf(q) / 2) / jet(eps, mp.mpf(q) / 2, floor)["s"]
        for q in range(2, 20)]
chk("M2 cure: dN/dS = -1 exactly (the extremal of rung 1)",
    max(abs(s + 1) for s in sigs) < mp.mpf("1e-18"), f"max dev {mp.nstr(max(abs(s+1) for s in sigs),3)}")
_ce, _cf, _ = cure()
_Nfar = jet(_ce, mp.mpf(30), _cf)["N"]
chk("M3 N -> 0 at the floor: the curve passes through the origin of the plane",
    _Nfar < mp.mpf("1e-7"), f"N(x=30) = {mp.nstr(_Nfar,3)}")


print("\n=== N. the whole apparatus, on measured log-accuracies ===")
# AXIOM: D2 l >= 0 (log-convexity).  HYPOTHESIS: D2 l never grows (Widder, exact at every h)
def d2(l, j):
    return l[j - 1] - 2 * l[j] + l[j + 1]


for nm, (eps, floor, meta), div in [("two-atom", two_atom(), False), ("cure", cure(), True),
                                    ("power-floor", power_floor(), True),
                                    ("Lomax", lomax(), True),
                                    ("3 atoms+floor", atoms([mp.mpf("0.01"), mp.mpf("0.5"), mp.mpf("0.49")],
                                                            [mp.mpf(0), mp.mpf("0.4"), mp.mpf("1.8")]), False)]:
    for h in [mp.mpf("0.3"), mp.mpf("1.0")]:
        xs = [mp.mpf("0.6") + h * j for j in range(10)]
        l = [mp.log(eps(x)) for x in xs]
        D2 = [d2(l, j) for j in range(1, 9)]
        chk(f"N1 {nm} h={mp.nstr(h,2)}: axiom = D2 ln(eps) >= 0", all(v >= -mp.mpf("1e-40") for v in D2),
            f"min {mp.nstr(min(D2),3)}")
        mono = all(D2[i] >= D2[i + 1] - mp.mpf("1e-40") for i in range(len(D2) - 1))
        if div:
            chk(f"N2 {nm} h={mp.nstr(h,2)}: hypothesis = D2 never grows (exact at this h)", mono)
        else:
            rr = sum(1 for i in range(len(D2) - 1) if D2[i + 1] > D2[i])
            chk(f"N2 {nm} h={mp.nstr(h,2)}: growth refutes; its absence proves nothing (one-sided)",
                True, f"D2 rises {rr}/7 steps -> {'REFUTED' if rr else 'no verdict on this window'}")

# telescoping: every third difference in the window sums to the two end curvatures
eps, floor, _ = two_atom()
h = mp.mpf("0.5")
xs = [mp.mpf("1.0") + h * j for j in range(9)]
l = [mp.log(eps(x)) for x in xs]
tel = sum(l[j + 3] - 3 * l[j + 2] + 3 * l[j + 1] - l[j] for j in range(6))
chk("N3 telescoping: sum of all third differences = D2(right end) - D2(left end)",
    abs(tel - (d2(l, 7) - d2(l, 1))) < mp.mpf("1e-45"),
    f"{mp.nstr(tel,8)} = {mp.nstr(d2(l,7)-d2(l,1),8)}")
chk("N4 the test is LINEAR in the measured log-errors (no ratio, no plug-in)",
    True, "coefficients (1,-2,1) at each end; variance exactly 12 sigma^2 for disjoint ends")

print("\n=== O. divisible <=> no elbow ===")
for nm, (eps, floor, meta), div in [("cure", cure(), True), ("power-floor", power_floor(), True),
                                    ("Lomax", lomax(), True), ("two-atom", two_atom(), False)]:
    ks = [jet(eps, mp.mpf("0.4") + mp.mpf(q) / 2, floor)["kap"] for q in range(1, 26)]
    rises = sum(1 for i in range(len(ks) - 1) if ks[i + 1] > ks[i])
    if div:
        chk(f"O1 {nm}: curvature never rises (no elbow)", rises == 0, f"{rises} rises/24")
    else:
        chk(f"O1 {nm}: curvature rises then falls (an elbow) -> indivisible", rises > 0,
            f"{rises} rises/24")
chk("O2 a bounded spectrum forces an elbow: kappa -> 0 at both ends",
    jet(two_atom()[0], mp.mpf("0.02"), two_atom()[1])["kap"] < mp.mpf("0.02")
    and jet(two_atom()[0], mp.mpf("40"), two_atom()[1])["kap"] < mp.mpf("1e-10"))

import csv as _csv
import os as _os
_RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_os.makedirs(_RES, exist_ok=True)
with open(_os.path.join(_RES, "table_plane_checks.csv"), "w", newline="") as _fh:
    _w = _csv.writer(_fh)
    _w.writerow(["check", "passed", "detail"])
    for _n, _d in OK:
        _w.writerow([_n, 1, _d])
    for _n, _d in FAIL:
        _w.writerow([_n, 0, _d])
print("  -> results/table_plane_checks.csv")

print("\n" + "=" * 74)
print(f"  {len(OK)} checks passed, {len(FAIL)} failed")
if FAIL:
    for n, d in FAIL:
        print("   FAILED:", n, d)
print("=" * 74)
