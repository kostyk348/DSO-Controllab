#!/usr/bin/env python3
"""
On-target Bayesian Optimization
================================
Optimize a fixed-structure linear controller DIRECTLY on the target
(QEMU Cortex-M4, closed loop) — no Python plant model used for the objective.

Structure:  u = a0*e + a1*ie + a2*de + a3*r + a4*y   (5 params)
Surrogate : Gaussian Process (RBF) + Expected Improvement (numpy only)
Objective : IAE measured by hardware-in-the-loop
"""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hil import run_hil


# ──────────────────────────────────────────────
#  GP surrogate (numpy only)
# ──────────────────────────────────────────────

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def rbf(X1, X2, ls=0.4, var=1.0):
    d2 = (np.sum(X1**2, 1)[:, None] + np.sum(X2**2, 1)[None, :]
          - 2.0 * X1 @ X2.T)
    return var * np.exp(-0.5 * np.maximum(d2, 0.0) / (ls * ls))


class GP:
    def __init__(self, ls=0.4, var=1.0, noise=1e-4):
        self.ls, self.var, self.noise = ls, var, noise
        self.X = None
        self.alpha = None
        self.L = None

    def fit(self, X, y):
        self.X = np.atleast_2d(X)
        K = rbf(self.X, self.X, self.ls, self.var) + self.noise * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        self.ymean = float(np.mean(y))

    def predict(self, Xs):
        Xs = np.atleast_2d(Xs)
        Ks = rbf(self.X, Xs, self.ls, self.var)
        mu = Ks.T @ self.alpha
        v = np.linalg.solve(self.L, Ks)
        var = self.var - np.sum(v**2, 0)
        return mu, np.maximum(var, 1e-10)


def expected_improvement(mu, var, best):
    sigma = np.sqrt(var)
    z = (best - mu) / np.maximum(sigma, 1e-9)
    ei = (best - mu) * np.array([_norm_cdf(float(zi)) for zi in z]) \
         + sigma * np.array([_norm_pdf(float(zi)) for zi in z])
    return np.maximum(ei, 0.0)


# ──────────────────────────────────────────────
#  Fixed-structure linear controller
# ──────────────────────────────────────────────

def make_linear_tree(params):
    """u = a0*e + a1*ie + a2*de + a3*r + a4*y  (5-param linear)"""
    from racs2 import Node
    terms = ['e', 'ie', 'de', 'r', 'y']
    node = None
    for a, t in zip(params, terms):
        term = Node('mul', left=Node('const', val=float(a)), right=Node('var', val=t))
        node = term if node is None else Node('add', left=node, right=term)
    return node


def make_pid_tree(params):
    """u = Kp*e + Ki*ie + Kd*de  (3-param PID)"""
    from racs2 import Node
    Kp, Ki, Kd = params
    terms = [('e', Kp), ('ie', Ki), ('de', Kd)]
    node = None
    for t, a in terms:
        term = Node('mul', left=Node('const', val=float(a)), right=Node('var', val=t))
        node = term if node is None else Node('add', left=node, right=term)
    return node


# ──────────────────────────────────────────────
#  On-target objective
# ──────────────────────────────────────────────

class OnTargetObjective:
    """Evaluate a parameter vector by running the closed loop on QEMU."""

    def __init__(self, plant_name, n_steps=3000, penalty=30.0, log=False,
                 structure='pid'):
        self.plant_name = plant_name
        self.n_steps = n_steps
        self.penalty = penalty
        self.log = log
        self.structure = structure
        self.cache = {}
        self.n_evals = 0

    def __call__(self, params):
        key = tuple(np.round(params, 6))
        if key in self.cache:
            return self.cache[key]
        tree = make_pid_tree(params) if self.structure == 'pid' else make_linear_tree(params)
        res = run_hil(tree, self.plant_name, n_steps=self.n_steps, timeout=20)
        self.n_evals += 1
        if res.get("error") or res.get("iae") is None:
            raw = self.penalty
        else:
            iae = res["iae"]
            raw = iae if (np.isfinite(iae) and iae < self.penalty) else self.penalty
        val = math.log10(raw + 1.0) if self.log else raw
        self.cache[key] = val
        return val


def structured_init(bounds):
    """PID-like seeds: (Kp, Ki, Kd)."""
    seeds = [
        np.array([3.0, 1.0, 0.5]),
        np.array([5.0, 2.0, 0.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([8.0, 3.0, 0.5]),
        np.array([1.0, 0.5, 0.2]),
    ]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    out = [np.clip(s, lo, hi) for s in seeds]
    # pad/truncate to match dimension
    d = len(bounds)
    out = [s[:d] for s in out]
    return out


def bayes_opt(objective, bounds, n_init=8, n_iter=22, seed=0, init_points=None):
    """GP-EI Bayesian optimization. bounds: list of (lo, hi)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    d = len(bounds)

    if init_points is not None:
        X = [np.clip(np.asarray(p, float), lo, hi) for p in init_points]
        while len(X) < n_init:
            X.append(rng.uniform(lo, hi))
    else:
        X = [rng.uniform(lo, hi) for _ in range(n_init)]
    y = [objective(x) for x in X]
    X = np.array(X)
    y = np.array(y)
    history = [float(np.min(y))]

    for it in range(n_iter):
        gp = GP(ls=0.35, var=float(np.var(y) + 1e-6), noise=1e-4)
        gp.fit(X, y)
        # candidate pool, excluding points too close to already-evaluated ones
        cand = rng.uniform(lo, hi, size=(2000, d))
        if len(X) > 0:
            d2 = np.sum((cand[:, None, :] - X[None, :, :])**2, axis=2)
            cand = cand[np.min(d2, axis=1) > 1e-4]
        if len(cand) == 0:
            cand = rng.uniform(lo, hi, size=(200, d))
        mu, var = gp.predict(cand)
        ei = expected_improvement(mu, var, float(np.min(y)))
        x_next = cand[int(np.argmax(ei))]
        y_next = objective(x_next)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        history.append(float(np.min(y)))
        print(f"    BO iter {it+1:2d}/{n_iter}: IAE={y_next:.4f}  best={np.min(y):.4f}")

    best_i = int(np.argmin(y))
    return X[best_i], float(y[best_i]), history


def random_search(objective, bounds, n_evals, seed=99):
    """Random-search baseline with the same evaluation budget."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    y = []
    hist = []
    best = float('inf')
    for _ in range(n_evals):
        x = rng.uniform(lo, hi)
        v = objective(x)
        y.append(v)
        best = min(best, v)
        hist.append(best)
    return best, hist


if __name__ == "__main__":
    import pickle

    PLANTS = {
        'Underdamped': 'Underdamped',
        'NonMinPhase': 'NonMinPhase',
        'FourthOrder': 'FourthOrder',
    }

    print("=" * 70)
    print("  On-Target Bayesian Optimization (closed loop on QEMU Cortex-M4)")
    print("  structure: PID  u = Kp*e + Ki*ie + Kd*de")
    print("=" * 70)

    bounds = [(0.0, 12.0), (0.0, 6.0), (0.0, 6.0)]   # Kp, Ki, Kd
    N_INIT, N_ITER = 12, 18
    BUDGET = N_INIT + N_ITER

    # NSGA-II reference
    try:
        with open('/home/lain/racs_nsga_results.pkl', 'rb') as f:
            nsga = pickle.load(f)
    except FileNotFoundError:
        nsga = {}

    NSGA_KEY = {
        'Underdamped': 'Underdamped (ζ=0.15,ωₙ=1.5)',
        'NonMinPhase': 'Non-Min Phase (1-2s)/(s+1)³',
        'FourthOrder': '4th-Order 1/(s+1)⁴',
    }

    summary = []
    for label, cname in PLANTS.items():
        print(f"\n{'='*70}\n  {label}\n{'='*70}")
        obj = OnTargetObjective(cname, n_steps=3000, penalty=30.0, log=True)
        best_x, best_y_log, hist = bayes_opt(
            obj, bounds, n_init=N_INIT, n_iter=N_ITER, seed=0,
            init_points=structured_init(bounds))
        best_iae = 10 ** best_y_log - 1.0
        print(f"\n  BO result: IAE={best_iae:.4f}  ({obj.n_evals} on-target evals)")
        print(f"  params: Kp,Ki,Kd = {np.round(best_x, 3).tolist()}")

        # random-search baseline, same budget
        obj2 = OnTargetObjective(cname, n_steps=3000, penalty=30.0, log=True)
        rs_best_log, rs_hist = random_search(obj2, bounds, BUDGET, seed=99)
        rs_iae = 10 ** rs_best_log - 1.0
        print(f"  random search (same budget): IAE={rs_iae:.4f}")

        key = NSGA_KEY.get(label)
        ns = nsga[key][0]['iae'] if (key in nsga and nsga[key]) else None
        summary.append((label, best_iae, rs_iae, ns))

    print("\n" + "=" * 70)
    print("  ON-TARGET BO vs RANDOM SEARCH vs NSGA-II")
    print("=" * 70)
    print(f"  {'plant':<14} {'BO(30)':>9} {'random(30)':>11} {'NSGA-II':>9}  BO gain")
    for label, bo, rs, ns in summary:
        gain = f"{100*(rs-bo)/max(1e-9,rs):+.0f}%" if rs > 0 else "-"
        ns_s = f"{ns:.3f}" if ns is not None else "n/a"
        print(f"  {label:<14} {bo:9.4f} {rs:11.4f} {ns_s:>9}  {gain}")
    print("  (BO uses the same 30 on-target evaluations as random search)")

    print("\nDone.")
