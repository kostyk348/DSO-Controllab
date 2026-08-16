#!/usr/bin/env python3
"""
RACS + DSO — NSGA-II Multi-Objective Controller Synthesis
==========================================================
Replaces scalarization with true multi-objective evolution:
  objectives (ALL MINIMIZED):
    f1 = variance proxy (determinism deficit)   ← DSO PRIMARY
    f2 = IAE (control quality)                  ← secondary
    f3 = resource cost (cycles + RAM)           ← tertiary
  hard constraint: ResourceContract (≤80cyc, ≤48B RAM)

NSGA-II: fast non-dominated sorting + crowding distance + μ+λ elitism.
"""
import sys, os, time, pickle, math
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from racs2 import *
from dso import *


# ──────────────────────────────────────────────
#  NSGA-II Core
# ──────────────────────────────────────────────

def fast_non_dominated_sort(objectives):
    """objectives: np.ndarray (n_pop, n_obj) — all minimized.
    Returns list of fronts, each a list of indices."""
    n = len(objectives)
    dom_count = np.zeros(n, dtype=int)
    dominates = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            # does p dominate q?
            p_better = objectives[p] <= objectives[q]
            q_better = objectives[q] <= objectives[p]
            if np.all(p_better) and not np.all(q_better):
                dominates[p].append(q)
            elif np.all(q_better) and not np.all(p_better):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    # remove trailing empty front
    while fronts and not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(front_indices, objectives):
    """Assign crowding distance to each individual in a front."""
    n = len(front_indices)
    if n == 0:
        return {}
    dist = {idx: 0.0 for idx in front_indices}
    n_obj = objectives.shape[1]

    for obj in range(n_obj):
        # sort front by this objective
        ordered = sorted(front_indices, key=lambda idx: objectives[idx, obj])
        obj_vals = [objectives[idx, obj] for idx in ordered]
        fmin, fmax = min(obj_vals), max(obj_vals)
        spread = fmax - fmin
        if spread == 0:
            continue
        dist[ordered[0]] = float('inf')
        dist[ordered[-1]] = float('inf')
        for j in range(1, n - 1):
            dist[ordered[j]] += (obj_vals[j + 1] - obj_vals[j - 1]) / spread

    return dist


class NSGA2:
    """Multi-objective GP evolution (NSGA-II)."""

    def __init__(self, plant_class, Ts, t_end, contract: ResourceContract,
                 pop_size=60, generations=20, max_depth=6,
                 iae_ref: float = 1.0):
        self.plant_class = plant_class
        self.Ts = Ts
        self.t_end = t_end
        self.contract = contract
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.da = DeterminismAnalyzer()
        self.iae_ref = iae_ref      # normalization baseline (PID IAE)

        self.pop = []          # list of Node trees
        self.obj = None        # np.ndarray (pop_size, 3) all minimized
        self.fronts = []
        self.history = {'f1': [], 'f2': [], 'f3': []}  # best per gen
        self.seen_formulas = {}  # formula → count (duplicate suppression)

    # ── objectives ──
    def _objectives(self, tree):
        """Return (f1, f2, f3) all to minimize. None if contract hard-violates."""
        # f2: IAE (quality), normalized by PID baseline
        res = run_sim(self.plant_class, self._ctrl(tree), self.Ts, self.t_end)
        f2 = res['iae'] / max(self.iae_ref, 1e-9)

        # f3: resources (calibrated cycles incl. ABI overhead)
        mp = MemoryPlanner()
        mp.analyze_tree(tree)
        cyc = tree.cycles_total()
        ram = mp.total_ram()
        f3 = cyc / 80.0 + ram / 48.0   # normalized to contract budget

        # f1: variance proxy (static determinism deficit) — DSO PRIMARY
        det = self.da.analyze(tree, mp)
        f1 = (100.0 - det['determinism_score']) / 100.0

        # hard contract check
        passed, _, _ = self.contract.check(
            cycles=cyc, ram=ram, latency_us=det['wcet_us'], jitter_ns=det['jitter_ns'])
        if not passed:
            return None  # infeasible

        return (f1, f2, f3)

    def _ctrl(self, tree):
        return lambda r, y, e, ie, de: tree.evaluate({'e': e, 'ie': ie, 'de': de, 'r': r, 'y': y})

    # ── evolution ──
    def _init_pop(self):
        self.pop = []
        for i in range(self.pop_size):
            self.pop.append(random_tree(2 + i % (self.max_depth - 1), self.max_depth, VARS))

    def _evaluate(self, pop):
        """Compute objectives for pop; infeasible → dominated far point."""
        obj = np.full((len(pop), 3), np.inf)
        for i, tree in enumerate(pop):
            o = self._objectives(tree)
            if o is not None:
                obj[i] = o
        return obj

    def _select_parent(self):
        """Binary tournament: better front rank, then crowding distance."""
        while True:
            i1, i2 = np.random.randint(0, len(self.pop), 2)
            # find ranks
            r1 = r2 = None
            for fi, front in enumerate(self.fronts):
                if i1 in front: r1 = fi
                if i2 in front: r2 = fi
                if r1 is not None and r2 is not None: break
            if r1 is None: r1 = len(self.fronts)
            if r2 is None: r2 = len(self.fronts)
            if r1 < r2:
                return i1
            if r2 < r1:
                return i2
            # same rank → crowding distance
            d1 = self.crowd.get(i1, 0)
            d2 = self.crowd.get(i2, 0)
            return i1 if d1 > d2 else i2

    def _mutate_cx(self, parent):
        """Crossover + mutation producing one child."""
        other = self.pop[self._select_parent()]
        if np.random.random() < 0.75:
            c1, _ = subtree_cx(parent, other)
        else:
            c1 = parent.clone()
        if np.random.random() < 0.20:
            c1 = mutate(c1, VARS)
        if c1.depth() > self.max_depth:
            c1 = parent.clone()
        return c1

    def run(self):
        self._init_pop()
        self.obj = self._evaluate(self.pop)
        self.fronts = fast_non_dominated_sort(self.obj)

        for gen in range(self.generations):
            # compute crowding for all fronts (for parent selection)
            self.crowd = {}
            for front in self.fronts:
                self.crowd.update(crowding_distance(front, self.obj))

            # generate offspring (μ+λ)
            offspring = []
            for _ in range(self.pop_size):
                parent = self.pop[np.random.randint(0, len(self.pop))]
                offspring.append(self._mutate_cx(parent))

            # combine μ+λ
            combined_pop = self.pop + offspring
            combined_obj = self._evaluate(combined_pop)

            # DUPLICATE SUPPRESSION: identical formulas are dominated
            # by their first occurrence (prevents trivial-solution flooding)
            self.seen_formulas = {}
            dup_mask = np.zeros(len(combined_pop), dtype=bool)
            for i, tree in enumerate(combined_pop):
                f = str(tree)
                if f in self.seen_formulas:
                    dup_mask[i] = True
                else:
                    self.seen_formulas[f] = i
            if np.any(dup_mask):
                # duplicates get inflated f1 (worse rank → dropped unless needed)
                combined_obj[dup_mask, :] = np.inf

            # non-dominated sort on combined
            fronts = fast_non_dominated_sort(combined_obj)

            # fill next generation by fronts
            new_pop = []
            new_obj = np.full((self.pop_size, 3), np.inf)
            idx = 0
            for front in fronts:
                if idx >= self.pop_size:
                    break
                if idx + len(front) <= self.pop_size:
                    for fi in front:
                        new_pop.append(combined_pop[fi])
                        new_obj[idx] = combined_obj[fi]
                        idx += 1
                else:
                    # partial front: keep by crowding distance
                    crowd = crowding_distance(front, combined_obj)
                    ordered = sorted(front, key=lambda fi: crowd.get(fi, 0), reverse=True)
                    for fi in ordered[:self.pop_size - idx]:
                        new_pop.append(combined_pop[fi])
                        new_obj[idx] = combined_obj[fi]
                        idx += 1

            # if all infeasible/duplicates, force re-init diversity
            if all(np.isinf(new_obj[:, 0])):
                self._init_pop()
                new_obj = self._evaluate(self.pop)
                new_pop = list(self.pop)

            self.pop = new_pop
            self.obj = new_obj
            self.fronts = fast_non_dominated_sort(self.obj)

            # track best feasible per objective
            feasible = self.obj[:, 0] != np.inf
            if np.any(feasible):
                self.history['f1'].append(np.min(self.obj[feasible, 0]))
                self.history['f2'].append(np.min(self.obj[feasible, 1]))
                self.history['f3'].append(np.min(self.obj[feasible, 2]))
            else:
                for k in self.history: self.history[k].append(float('inf'))

            if gen % 5 == 0 or gen == self.generations - 1:
                f2 = min(self.history['f2']) if self.history['f2'] else float('inf')
                print(f"  Gen {gen:3d}: best-IAE={f2:.4f} "
                      f"best-f1={min(self.history['f1']):.4f} "
                      f"front_size={len(self.fronts[0]) if self.fronts else 0}")

        return self.pop, self.obj, self.fronts, self.history


# ──────────────────────────────────────────────
#  Front Analysis + Variance Measurement
# ──────────────────────────────────────────────

def analyze_front(pop, obj, fronts, plant_class, Ts, t_end, contract, n_var_runs=8):
    """Measure ACTUAL variance on the best feasible front. Returns report rows."""
    da = DeterminismAnalyzer()
    rows = []
    feasible_idx = [i for i in range(len(obj)) if obj[i, 0] != np.inf]

    for i in feasible_idx:
        tree = pop[i]
        res = run_sim(plant_class, lambda r, y, e, ie, de: tree.evaluate(
            {'e': e, 'ie': ie, 'de': de, 'r': r, 'y': y}), Ts, t_end)

        mp = MemoryPlanner(); mp.analyze_tree(tree)
        det = da.analyze(tree, mp)

        # actual variance measurement
        ctrl_fn = lambda r, y, e, ie, de: tree.evaluate({'e': e, 'ie': ie, 'de': de, 'r': r, 'y': y})
        var = da.measure_variance(tree, plant_class, Ts, t_end, n_runs=n_var_runs, ctrl_fn=ctrl_fn)

        passed, _, _ = contract.check(det['cycles'], det['ram_bytes'], det['wcet_us'], det['jitter_ns'])
        rows.append({
            'tree': tree,
            'formula': tree_formula(tree),
            'iae': res['iae'],
            'cycles': det['cycles'],  # calibrated (incl. ABI overhead)
            'cycles_model': tree.cycles(),
            'ram': det['ram_bytes'],
            'var_out': var['output_variance'],
            'var_iae': var['iae_variance'],
            'detvar': var['determinism_variance_score'],
            'det_static': det['determinism_score'],
            'contract': passed,
            'f1': float(obj[i, 0]),
            'f2': float(obj[i, 1]),
            'f3': float(obj[i, 2]),
        })
    return rows


# ──────────────────────────────────────────────
#  Benchmark across plants
# ──────────────────────────────────────────────

def run_nsga_benchmark(plants, contract, pop_size=60, generations=20):
    all_rows = {}
    for PlantCls in plants:
        plant = PlantCls(Ts=cfg.Ts)
        print(f"\n{'='*70}\n  NSGA-II: {plant.name}\n{'='*70}")

        # PID baseline for IAE normalization
        pid = optimize_pid(PlantCls, cfg.Ts, cfg.t_end)
        pid_res = run_sim(PlantCls, lambda r, y, e, ie, de: pid.compute(r, y, e, ie, de),
                          cfg.Ts, cfg.t_end)
        iae_ref = pid_res['iae']
        print(f"  [PID baseline: IAE={iae_ref:.4f}]")

        nsga = NSGA2(PlantCls, cfg.Ts, cfg.t_end, contract,
                     pop_size=pop_size, generations=generations, iae_ref=iae_ref)
        t0 = time.time()
        pop, obj, fronts, hist = nsga.run()
        print(f"  elapsed: {time.time()-t0:.0f}s")

        rows = analyze_front(pop, obj, fronts, PlantCls, cfg.Ts, cfg.t_end, contract)
        # sort by IAE
        rows.sort(key=lambda r: r['iae'])
        all_rows[plant.name] = rows

        # print top-5 by IAE + top-3 by variance
        print(f"\n  TOP-5 by IAE (feasible, contract-OK):")
        for r in rows[:5]:
            print(f"    IAE={r['iae']:8.4f}  cyc={r['cycles']:3d}  RAM={r['ram']:3d}B  "
                  f"DetVar={r['detvar']:5.1f}  {r['formula']}")

        best_var = sorted(rows, key=lambda r: r['detvar'], reverse=True)
        print(f"\n  TOP-3 by Determinism (zero variance):")
        for r in best_var[:3]:
            print(f"    DetVar={r['detvar']:5.1f}  IAE={r['iae']:8.4f}  cyc={r['cycles']:3d}  "
                  f"{r['formula']}")

    return all_rows


# ──────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────

def plot_pareto3d(all_rows, save_path='racs_nsga_pareto3d.png'):
    n = len(all_rows)
    fig = plt.figure(figsize=(6*n, 6))
    for idx, (name, rows) in enumerate(all_rows.items()):
        ax = fig.add_subplot(1, n, idx+1, projection='3d')
        if rows:
            iae = [r['iae'] for r in rows]
            cyc = [r['cycles'] for r in rows]
            detvar = [r['detvar'] for r in rows]
            sc = ax.scatter(iae, cyc, detvar, c=detvar, cmap='viridis', s=30, alpha=0.8)
            ax.set_xlabel('IAE (↓ good)')
            ax.set_ylabel('Cycles (↓ cheap)')
            ax.set_zlabel('DetVar (↑ determ)')
            ax.set_title(name[:20], fontsize=9)
            fig.colorbar(sc, ax=ax, shrink=0.6, label='DetVar')
    fig.suptitle('NSGA-II Pareto Front: IAE × Cycles × Determinism', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_front2d(all_rows, save_path='racs_nsga_fronts.png'):
    """IAE vs Cycles and IAE vs DetVar scatter across all plants."""
    n = len(all_rows)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)
    for idx, (name, rows) in enumerate(all_rows.items()):
        if not rows:
            continue
        iae = [r['iae'] for r in rows]
        cyc = [r['cycles'] for r in rows]
        detvar = [r['detvar'] for r in rows]

        axes[0, idx].scatter(iae, cyc, c='steelblue', s=25, alpha=0.7)
        axes[0, idx].set_xlabel('IAE'); axes[0, idx].set_ylabel('Cycles')
        axes[0, idx].set_title(f'{name[:24]}', fontsize=8)
        axes[0, idx].grid(True, alpha=0.3)

        axes[1, idx].scatter(iae, detvar, c='green', s=25, alpha=0.7)
        axes[1, idx].set_xlabel('IAE'); axes[1, idx].set_ylabel('DetVar (variance det)')
        axes[1, idx].grid(True, alpha=0.3)

    fig.suptitle('NSGA-II Pareto Fronts (all feasible individuals)', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  RACS + DSO — NSGA-II Multi-Objective Controller Synthesis")
    print("  objectives: (variance-proxy, IAE, resources) — all minimized")
    print("  hard constraint: ResourceContract(≤80cyc, ≤48B RAM)")
    print("=" * 70)

    contract = ResourceContract(cycles_max=80, ram_bytes_max=48, latency_us_max=100)

    plants = [SecondOrderDelay, FourthOrder, Underdamped,
              NonMinPhase, IntegratingDelay]

    t0 = time.time()
    all_rows = run_nsga_benchmark(plants, contract, pop_size=60, generations=20)
    print(f"\n\nTotal elapsed: {time.time()-t0:.0f}s")

    plot_pareto3d(all_rows)
    plot_front2d(all_rows)

    # Save
    with open('/tmp/racs_nsga_results.pkl', 'wb') as f:
        pickle.dump(all_rows, f)

    print("\nDone. Results saved to /tmp/racs_nsga_results.pkl")
