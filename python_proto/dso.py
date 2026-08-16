#!/usr/bin/env python3
"""
DSO — Deterministic Systems Optimization
=========================================
Memory Planner + Resource Contracts + Determinism Analysis.
Compile-time computation for RACS controllers.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
import math
import numpy as np

# Cortex-M4 @ 48 MHz
CLOCK_NS = 20.8  # 48 MHz → 20.8 ns per cycle

# ──────────────────────────────────────────────
#  Arena Types
# ──────────────────────────────────────────────

class ArenaType:
    ACTIVATION = 'activation'  # per-call local vars
    WEIGHT = 'weight'          # persistent coefficients
    FRAME = 'frame'            # per-iteration buffers
    TEMP = 'temp'              # expression temporaries

# ──────────────────────────────────────────────
#  Memory Block
# ──────────────────────────────────────────────

@dataclass
class MemBlock:
    name: str
    size_bytes: int
    arena: str
    offset: int = 0
    lifetime: Tuple[int, int] = (0, 0)  # birth, death (instr index)
    alignment: int = 4

    @property
    def end(self) -> int:
        return self.offset + self.size_bytes


# ──────────────────────────────────────────────
#  Memory Planner
# ──────────────────────────────────────────────

class MemoryPlanner:
    """Compile-time memory layout for a controller expression tree.
    No malloc. No runtime allocation. Linear arenas."""

    def __init__(self, clock_ns: float = CLOCK_NS):
        self.clock_ns = clock_ns
        self.blocks: List[MemBlock] = []
        self.arenas: Dict[str, int] = {
            ArenaType.ACTIVATION: 0,
            ArenaType.WEIGHT: 0,
            ArenaType.FRAME: 0,
            ArenaType.TEMP: 0,
        }
        self._layout_valid = False

    def analyze_tree(self, tree) -> None:
        """Analyze expression tree → extract constants, variables, structure."""
        from racs2 import Node
        self.blocks.clear()

        # 1. Constants → Weight Arena (literal pool)
        consts = set()
        def collect_consts(n):
            if n.op == 'const':
                consts.add(round(n.val, 6))
            if n.left: collect_consts(n.left)
            if n.right: collect_consts(n.right)
        collect_consts(tree)

        for i, c in enumerate(sorted(consts)):
            self.blocks.append(MemBlock(
                name=f"const_{i}",
                size_bytes=4,
                arena=ArenaType.WEIGHT,
                lifetime=(0, 1000)  # persistent
            ))

        # 2. Variables → Activation Arena (read from stack frame)
        vars_used = set()
        def collect_vars(n):
            if n.op == 'var':
                vars_used.add(n.val)
            if n.left: collect_vars(n.left)
            if n.right: collect_vars(n.right)
        collect_vars(tree)

        for v in sorted(vars_used):
            self.blocks.append(MemBlock(
                name=f"var_{v}",
                size_bytes=4,
                arena=ArenaType.ACTIVATION,
                lifetime=(0, 1000)
            ))

        # 3. Temporaries → Temp Arena (stack-like)
        depth = tree.depth()
        for i in range(depth):
            self.blocks.append(MemBlock(
                name=f"tmp_{i}",
                size_bytes=4,
                arena=ArenaType.TEMP,
                lifetime=(i, i+1)  # short-lived
            ))

        # 4. Output → Frame Arena
        self.blocks.append(MemBlock(
            name="output",
            size_bytes=4,
            arena=ArenaType.FRAME,
            lifetime=(0, 1)
        ))

        self._compute_layout()

    def analyze_pid(self, pid) -> None:
        """Analyze PID controller for comparison."""
        self.blocks.clear()
        self.blocks.append(MemBlock("Kp", 4, ArenaType.WEIGHT))
        self.blocks.append(MemBlock("Ki", 4, ArenaType.WEIGHT))
        self.blocks.append(MemBlock("Kd", 4, ArenaType.WEIGHT))
        self.blocks.append(MemBlock("Ts", 4, ArenaType.WEIGHT))
        self.blocks.append(MemBlock("ie", 4, ArenaType.WEIGHT))  # integrator state
        self.blocks.append(MemBlock("e", 4, ArenaType.ACTIVATION))
        self.blocks.append(MemBlock("de", 4, ArenaType.TEMP))
        self.blocks.append(MemBlock("u", 4, ArenaType.FRAME))
        self._compute_layout()

    def _compute_layout(self) -> None:
        """Assign offsets within each arena. No fragmentation guarantee."""
        arenas = {}
        for block in self.blocks:
            if block.arena not in arenas:
                arenas[block.arena] = 0
            # Align
            misalign = arenas[block.arena] % block.alignment
            if misalign:
                arenas[block.arena] += block.alignment - misalign
            block.offset = arenas[block.arena]
            arenas[block.arena] += block.size_bytes

        self.arenas = arenas
        self._layout_valid = True

    def total_ram(self) -> int:
        """Total RAM used across all arenas."""
        return sum(self.arenas.values())

    def per_arena(self) -> Dict[str, int]:
        return dict(self.arenas)

    def report(self) -> str:
        lines = []
        lines.append("─" * 50)
        lines.append("DSO Memory Planner — Layout Report")
        lines.append("─" * 50)
        arena_blocks = {a: [] for a in self.arenas}
        for b in self.blocks:
            arena_blocks[b.arena].append(b)

        for arena, size in sorted(self.arenas.items(), key=lambda x: x[1], reverse=True):
            blocks = arena_blocks.get(arena, [])
            lines.append(f"\n  {arena.upper():12s}: {size:4d} bytes  ({len(blocks)} blocks)")
            for b in blocks:
                line = f"    @{b.offset:4d}  {b.name:12s}  {b.size_bytes:2d} B  "
                if b.lifetime != (0, 1000):
                    line += f"  live [{b.lifetime[0]}–{b.lifetime[1]}]"
                lines.append(line)

        lines.append(f"\n  TOTAL RAM: {self.total_ram()} bytes")
        lines.append("─" * 50)
        return "\n".join(lines)


# ──────────────────────────────────────────────
#  Resource Contract
# ──────────────────────────────────────────────

@dataclass
class ResourceContract:
    """Hard/soft constraints on computational resources.
    If budget is None → unlimited (soft only)."""
    cycles_max: Optional[int] = 500
    ram_bytes_max: Optional[int] = 128
    latency_us_max: Optional[int] = 100  # worst-case per iteration
    jitter_max_ns: Optional[float] = 100

    # Penalty weights for violations (fraction of fitness)
    penalty_cycles: float = 0.5
    penalty_ram: float = 0.3
    penalty_latency: float = 0.2

    def check(self, cycles: int, ram: int, latency_us: float,
              jitter_ns: float = 0) -> Tuple[bool, float, Dict]:
        """Check contract. Returns (pass, penalty, details)."""
        violations = {}
        penalty = 0.0

        if self.cycles_max and cycles > self.cycles_max:
            viol = (cycles - self.cycles_max) / self.cycles_max
            violations['cycles'] = viol
            penalty += self.penalty_cycles * viol

        if self.ram_bytes_max and ram > self.ram_bytes_max:
            viol = (ram - self.ram_bytes_max) / self.ram_bytes_max
            violations['ram'] = viol
            penalty += self.penalty_ram * viol

        if self.latency_us_max and latency_us > self.latency_us_max:
            viol = (latency_us - self.latency_us_max) / self.latency_us_max
            violations['latency'] = viol
            penalty += self.penalty_latency * viol

        if self.jitter_max_ns and jitter_ns > self.jitter_max_ns:
            viol = (jitter_ns - self.jitter_max_ns) / self.jitter_max_ns
            violations['jitter'] = viol
            # jitter adds extra on top

        passed = len(violations) == 0
        return passed, penalty, violations

    def __str__(self) -> str:
        parts = []
        if self.cycles_max: parts.append(f"≤{self.cycles_max} cyc")
        if self.ram_bytes_max: parts.append(f"≤{self.ram_bytes_max} B RAM")
        if self.latency_us_max: parts.append(f"≤{self.latency_us_max} μs")
        if self.jitter_max_ns is not None: parts.append(f"jitter<{self.jitter_max_ns}ns")
        return "Contract(" + ", ".join(parts) + ")"


# ──────────────────────────────────────────────
#  Determinism Analyzer
# ──────────────────────────────────────────────

class DeterminismAnalyzer:
    """Analyze determinism of a controller:
    WCET, jitter, branch predictability, cache behavior.
    Variance-based: primary criterion = variance(output, latency, IAE)."""

    def __init__(self, clock_ns: float = CLOCK_NS):
        self.clock_ns = clock_ns

    def analyze(self, tree, mem_planner: MemoryPlanner = None) -> Dict:
        """Compute determinism metrics for a tree controller."""
        from racs2 import Node

        cyc = tree.cycles_total()  # calibrated (incl. ABI overhead)
        depth = tree.depth()

        # Count branch-like operations (conditional behavior)
        branch_count = 0
        nonlinear_count = 0
        def count_ops(n):
            nonlocal branch_count, nonlinear_count
            if n.op in ('min', 'max'): branch_count += 1
            if n.op in ('sin', 'cos', 'sqrt'): nonlinear_count += 1
            if n.left: count_ops(n.left)
            if n.right: count_ops(n.right)
        count_ops(tree)

        # WCET: worst-case path (all branches taken)
        wcet_cycles = cyc  # our model already counts worst-case

        # Best-case: execute the minimum path
        bcet_cycles = cyc - branch_count * 2  # skip some branches

        # Jitter estimation: variance from unpredictable branches
        # Each branch adds uncertainty
        jitter_ns = branch_count * 3 * self.clock_ns  # branch misprediction penalty

        if nonlinear_count > 0:
            jitter_ns += nonlinear_count * 10 * self.clock_ns  # nonlin iteration variance

        # Memory footprint → cache pressure
        ram = mem_planner.total_ram() if mem_planner else 0
        # Cortex-M4: typically 8-64 KB cache
        cache_misses_est = max(0, (ram - 4096) / 4096) * 10 if ram > 4096 else 0.5

        # Determinism score: 0-100
        # Penalties for: branches, nonlinear ops, cache pressure, jitter
        det_score = 100.0
        det_score -= branch_count * 5.0          # -5% per branch
        det_score -= nonlinear_count * 3.0       # -3% per nonlinear op
        det_score -= min(50, cache_misses_est * 2)  # cache penalty
        det_score -= min(30, jitter_ns / 10)     # jitter penalty
        det_score = max(0, min(100, det_score))

        return {
            'cycles': cyc,
            'wcet_cycles': wcet_cycles,
            'bcet_cycles': bcet_cycles,
            'wcet_us': wcet_cycles * self.clock_ns / 1000,
            'bcet_us': bcet_cycles * self.clock_ns / 1000,
            'jitter_ns': jitter_ns,
            'branch_count': branch_count,
            'nonlinear_count': nonlinear_count,
            'code_size_bytes': cyc * 4,  # approx: 4 bytes per instruction
            'ram_bytes': ram,
            'cache_misses_est': cache_misses_est,
            'determinism_score': det_score,
            'clock_ns': self.clock_ns,
        }

    def analyze_pid(self, mem_planner: MemoryPlanner = None) -> Dict:
        """Determinism metrics for PID."""
        ram = mem_planner.total_ram() if mem_planner else 0
        det_score = 95.0  # PID is very deterministic (straight-line code)
        return {
            'cycles': 20,
            'wcet_cycles': 20,
            'bcet_cycles': 20,
            'wcet_us': 20 * self.clock_ns / 1000,
            'bcet_us': 20 * self.clock_ns / 1000,
            'jitter_ns': 0.0,
            'branch_count': 0,
            'nonlinear_count': 0,
            'code_size_bytes': 80,
            'ram_bytes': ram,
            'cache_misses_est': 0.2,
            'determinism_score': det_score,
        }

    def analyze_lqr(self, n: int, mem_planner: MemoryPlanner = None) -> Dict:
        """Determinism for LQR (matrix ops)."""
        ram = mem_planner.total_ram() if mem_planner else max(64, 8*n*n)
        cycles = 30 + 10*n
        det_score = 85.0  # matrix ops are deterministic but many loads/stores
        return {
            'cycles': cycles,
            'wcet_cycles': cycles,
            'bcet_cycles': cycles,
            'wcet_us': cycles * self.clock_ns / 1000,
            'bcet_us': cycles * self.clock_ns / 1000,
            'jitter_ns': 0.0,
            'branch_count': 0,
            'nonlinear_count': 0,
            'code_size_bytes': cycles * 4,
            'ram_bytes': ram,
            'cache_misses_est': min(20, ram / 512),
            'determinism_score': det_score,
        }

    def measure_variance(self, tree, plant_class, Ts, t_end,
                          n_runs: int = 8, ctrl_fn=None) -> Dict:
        """
        Measure controller VARIANCE across multiple runs with stochastic
        perturbation. In DSO, variance is the PRIMARY quality metric —
        lower variance = higher determinism = better controller.

        Runs n_runs simulations with increasing measurement noise,
        computes variance of u(t) and IAE across runs.
        """
        import numpy as np
        from racs2 import cfg

        all_u = []
        all_iae = []

        for run in range(n_runs):
            noise_scale = 0.0 if run == 0 else 0.001 * (run ** 0.5)
            res = run_sim_noisy(
                plant_class, ctrl_fn, Ts, t_end,
                noise_scale=noise_scale,
                dist_jitter=noise_scale * 0.5
            )
            all_u.append(res['u'])
            all_iae.append(res['iae'])

        all_u = np.array(all_u)
        u_var_per_step = np.var(all_u, axis=0)
        output_variance = float(np.mean(u_var_per_step))
        max_u_variance = float(np.max(u_var_per_step))

        iae_variance = float(np.var(all_iae))
        iae_std = float(np.std(all_iae))

        # Determinism FROM VARIANCE: if output_variance ≈ 0 → perfect
        if output_variance < 1e-14:
            determinism_variance_score = 100.0
        else:
            det_var_score = max(0, 100 - 20 * math.log10(1 + output_variance * 1e6))
            determinism_variance_score = min(100, det_var_score)

        return {
            'output_variance': output_variance,
            'max_u_variance': max_u_variance,
            'iae_variance': iae_variance,
            'iae_std': iae_std,
            'determinism_variance_score': determinism_variance_score,
            'n_runs': n_runs,
            'iae_values': all_iae,
        }


def run_sim_noisy(plant_class, ctrl_fn, Ts, t_end,
                  noise_scale=0.0, dist_jitter=0.0):
    """
    Like run_sim but with configurable measurement noise and disturbance jitter.
    Used by DeterminismAnalyzer.measure_variance() for variance measurement.
    """
    import numpy as np
    from racs2 import cfg
    np.random.seed(42)

    plant = plant_class(Ts=Ts)
    plant.reset()
    N = int(t_end / Ts)
    t = np.arange(N) * Ts
    y_arr = np.zeros(N)
    u_arr = np.zeros(N)
    iae = 0.0
    ie = 0.0
    e_prev = 0.0
    y = 0.0
    base_dist_time = t_end * 0.5
    dist_mag = -0.5

    for i in range(N):
        r = 1.0
        y_meas = y + np.random.randn() * noise_scale
        e = r - y_meas
        de = (e - e_prev) / Ts if Ts > 0 else 0.0
        u = ctrl_fn(r, y_meas, e, ie, de)
        u = max(cfg.u_min, min(cfg.u_max, u))

        dist_t = base_dist_time + np.random.randn() * dist_jitter * t_end
        u_plant = u + (dist_mag if t[i] >= max(0, dist_t) else 0.0)
        y = plant.update(u_plant)

        y_arr[i] = y
        u_arr[i] = u
        iae += abs(e) * Ts
        ie += e * Ts
        e_prev = e

    return {'t': t, 'y': y_arr, 'u': u_arr, 'iae': iae}


# ──────────────────────────────────────────────
#  DSO Controller Wrapper
# ──────────────────────────────────────────────

class DSOController:
    """Wraps any controller with DSO constraints.
    Provides compile-time guarantees on resources."""

    def __init__(self, inner, name: str = "DSO controller",
                 contract: Optional[ResourceContract] = None):
        self.inner = inner
        self.name = name
        self.contract = contract or ResourceContract()
        self.mem_planner = MemoryPlanner()
        self.det_analyzer = DeterminismAnalyzer()
        self.metrics: Dict = {}
        self._analyzed = False

    def analyze(self, tree=None, plant_class=None, Ts=0.01, t_end=30.0):
        """Run DSO analysis: memory plan + determinism + variance + contract check."""
        if tree is not None:
            self.mem_planner.analyze_tree(tree)
            self.metrics = self.det_analyzer.analyze(tree, self.mem_planner)
            var = self.det_analyzer.measure_variance(
                tree, plant_class, Ts, t_end,
                ctrl_fn=lambda r, y, e, ie, de: tree.evaluate({'e': e, 'ie': ie, 'de': de, 'r': r, 'y': y})
            )
            self.metrics.update(var)
        else:
            # For non-tree controllers (PID etc), use generic analysis
            self.mem_planner.analyze_pid(self.inner)
            self.metrics = self.det_analyzer.analyze_pid(self.mem_planner)
            self.metrics['output_variance'] = 0.0
            self.metrics['iae_variance'] = 0.0
            self.metrics['determinism_variance_score'] = 100.0

        # Check contract
        passed, penalty, violations = self.contract.check(
            cycles=self.metrics['cycles'],
            ram=self.metrics['ram_bytes'],
            latency_us=self.metrics['wcet_us'],
            jitter_ns=self.metrics.get('jitter_ns', 0)
        )
        self.metrics['contract_passed'] = passed
        self.metrics['contract_penalty'] = penalty
        self.metrics['contract_violations'] = violations

        # DSO score: variance-based primary, determinism secondary
        var_score = self.metrics.get('determinism_variance_score', 50) / 100
        static_det = self.metrics.get('determinism_score', 50) / 100
        self.metrics['dso_score'] = (
            0.6 * var_score +
            0.2 * static_det +
            0.1 * (1.0 if passed else 0.0) +
            0.1 * (1.0 / (1.0 + penalty))
        )
        self._analyzed = True
        return self.metrics

    def report(self) -> str:
        if not self._analyzed:
            return "Not analyzed yet. Call .analyze() first."
        m = self.metrics
        lines = []
        lines.append("=" * 56)
        lines.append(f"  DSO Report: {self.name}")
        lines.append(f"  Contract: {self.contract}")
        lines.append("=" * 56)
        lines.append(f"  Memory Plan:")
        lines.append(self.mem_planner.report().replace("─" * 50, "").strip())
        lines.append(f"\n  ── VARIANCE (PRIMARY DSO CRITERION) ──")
        lines.append(f"    Output variance:  {m.get('output_variance', 0):.3e}")
        lines.append(f"    IAE variance:     {m.get('iae_variance', 0):.3e}")
        lines.append(f"    IAE std:          {m.get('iae_std', 0):.3f}")
        lines.append(f"    Variance det:     {m.get('determinism_variance_score', 100):.1f}/100")
        lines.append(f"\n  ── STATIC ANALYSIS ──")
        lines.append(f"    WCET:             {m['wcet_us']:.1f} μs  ({m['wcet_cycles']} cycles)")
        lines.append(f"    BCET:             {m['bcet_us']:.1f} μs  ({m['bcet_cycles']} cycles)")
        lines.append(f"    Jitter:           {m['jitter_ns']:.1f} ns")
        lines.append(f"    Branches/NL ops:  {m['branch_count']}/{m['nonlinear_count']}")
        lines.append(f"    Code size:        {m['code_size_bytes']} B  (~{m['code_size_bytes']//4} instr)")
        lines.append(f"    Determinism:      {m['determinism_score']:.1f}/100")
        lines.append(f"    Cache misses:     ~{m['cache_misses_est']:.1f}")
        lines.append(f"\n  Contract: {'✓ PASS' if m['contract_passed'] else '✗ FAIL'}")
        if m['contract_violations']:
            for k, v in m['contract_violations'].items():
                lines.append(f"      {k}: violation {v:.2%}")
            lines.append(f"    Penalty: {m['contract_penalty']:.3f}")
        lines.append(f"  DSO Score (variance-weighted): {m['dso_score']:.3f}")
        lines.append("=" * 56)
        return "\n".join(lines)


# ──────────────────────────────────────────────
#  Benchmark Integration
# ──────────────────────────────────────────────

def dso_benchmark(results: List[dict], Ts=0.01, t_end=30.0) -> List[dict]:
    """Wrap RACS benchmark results with DSO analysis (variance-based)."""
    from racs2 import Node

    def _set_contract(ctrl, m):
        passed, penalty, violations = ctrl.contract.check(
            cycles=m.get('cycles', 0),
            ram=m.get('ram_bytes', 0),
            latency_us=m.get('wcet_us', 0),
            jitter_ns=m.get('jitter_ns', 0)
        )
        m['contract_passed'] = passed
        m['contract_penalty'] = penalty
        m['contract_violations'] = violations
        var_score = m.get('determinism_variance_score', 50) / 100
        det_score = m.get('determinism_score', 50) / 100
        m['dso_score'] = 0.6*var_score + 0.2*det_score + 0.2*(1.0 if passed else 0.0)
        return m

    # Map plant name → plant class
    from racs2 import SecondOrderDelay, FourthOrder, Underdamped, NonMinPhase, IntegratingDelay
    PLANT_MAP = {
        '2nd+Delay (τ₁=1.5,τ₂=0.8,L=2.0)': SecondOrderDelay,
        '4th-Order 1/(s+1)⁴': FourthOrder,
        'Underdamped (ζ=0.15,ωₙ=1.5)': Underdamped,
        'Non-Min Phase (1-2s)/(s+1)³': NonMinPhase,
        'Integrator+Delay (L=1.0)': IntegratingDelay,
    }

    dso_results = []
    for res in results:
        plant_name = res['plant']
        PlantCls = PLANT_MAP.get(plant_name, SecondOrderDelay)
        dso_entry = {'plant': plant_name}

        for tag, key, tree_key in [
            ('PID', 'pid', None),
            ('LQR', 'lqr', None),
            ('MPC', 'mpc', None),
            ('GP', 'gp', 'gp_tree'),
            ('GP-R', 'gpr', 'gpr_tree'),
        ]:
            if res.get(key) is None:
                continue

            ctrl = DSOController(res.get(key), name=f"{plant_name} {tag}")

            if tree_key and res.get(tree_key) is not None:
                tree = res[tree_key]
                ctrl.analyze(tree=tree, plant_class=PlantCls, Ts=Ts, t_end=t_end)
            elif tag == 'PID' and res.get('pid_ctrl'):
                ctrl.mem_planner.analyze_pid(res['pid_ctrl'])
                ctrl.metrics = ctrl.det_analyzer.analyze_pid(ctrl.mem_planner)
                ctrl.metrics['output_variance'] = 0.0
                ctrl.metrics['iae_variance'] = 0.0
                ctrl.metrics['determinism_variance_score'] = 100.0
                ctrl.metrics = _set_contract(ctrl, ctrl.metrics)
                ctrl._analyzed = True
            elif tag == 'LQR' and res.get('lqr_ctrl'):
                n = res['lqr_ctrl'].n
                ctrl.mem_planner.blocks = []
                ctrl.metrics = ctrl.det_analyzer.analyze_lqr(n)
                ctrl.metrics['ram_bytes'] = 8*n*n + 16
                ctrl.metrics['output_variance'] = 0.0
                ctrl.metrics['iae_variance'] = 0.0
                ctrl.metrics['determinism_variance_score'] = 100.0
                ctrl.metrics = _set_contract(ctrl, ctrl.metrics)
                ctrl._analyzed = True
            elif tag == 'MPC':
                ctrl.mem_planner.blocks = []
                ctrl.metrics = ctrl.det_analyzer.analyze_pid(ctrl.mem_planner)
                ctrl.metrics['cycles'] = res['mpc'].cycles if hasattr(res['mpc'], 'cycles') else 350
                ctrl.metrics['wcet_us'] = ctrl.metrics['cycles'] * CLOCK_NS / 1000
                ctrl.metrics['determinism_score'] = 70
                ctrl.metrics['output_variance'] = 0.0
                ctrl.metrics['iae_variance'] = 0.0
                ctrl.metrics['determinism_variance_score'] = 70.0
                ctrl.metrics = _set_contract(ctrl, ctrl.metrics)
                ctrl._analyzed = True

            dso_entry[tag] = ctrl

        dso_results.append(dso_entry)

    return dso_results


def dso_report_table(dso_results: List[dict]) -> str:
    """Print DSO analysis table with variance as primary metric."""
    lines = []
    lines.append("\n" + "=" * 130)
    lines.append("  DSO ANALYSIS — VARIANCE-CENTRIC (PRIMARY CRITERION)")
    lines.append("=" * 130)
    hdr = (f"{'Process':<28} {'Ctl':<6} {'Var_out':<11} {'Var_IAE':<11} "
           f"{'DetVar':<7} {'WCETμs':<8} {'RAM':<6} {'DetSt':<7} {'Score':<7} {'Contract':<8}")
    lines.append(f"  {hdr}")
    lines.append("  " + "-" * 104)
    for entry in dso_results:
        for tag in ['PID', 'LQR', 'MPC', 'GP', 'GP-R']:
            ctrl = entry.get(tag)
            if ctrl is None or not ctrl._analyzed:
                continue
            m = ctrl.metrics
            contract = '✓' if m.get('contract_passed', True) else '✗'
            var_out = f"{m.get('output_variance', 0):.2e}"
            var_iae = f"{m.get('iae_variance', 0):.2e}"
            det_var = f"{m.get('determinism_variance_score', 100):.0f}"
            row = (f"{entry['plant'][:26]:<28} {tag:<6} {var_out:<11} {var_iae:<11} "
                   f"{det_var:<7} {m['wcet_us']:<8.1f} {m['ram_bytes']:<6d} "
                   f"{m['determinism_score']:<7.1f} {m.get('dso_score', 0):<7.3f} {contract:<8}")
            lines.append(f"  {row}")
    lines.append("=" * 130)
    lines.append("  Var_out = output variance | Var_IAE = IAE variance across runs")
    lines.append("  DetVar  = determinism from variance (100 = zero variance)")
    lines.append("  DetSt   = static determinism score (branch/nonlinear analysis)")
    lines.append("  Score   = weighted DSO score (60% variance + 20% static + 20% contract)")
    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    from racs2 import FourthOrder, SecondOrderDelay, Underdamped, NonMinPhase, IntegratingDelay
    from racs2 import Node, random_tree, VARS, PID

    print("=" * 70)
    print("  DSO — Deterministic Systems Optimization")
    print("  Memory Planner + Resource Contracts + Determinism")
    print("=" * 70)

    # Demo: analyze a GP tree from RACS
    tree = random_tree(0, 5, VARS)

    print("\n  [GP Controller Analysis]")
    ctrl = DSOController(tree, name="GP random tree",
                         contract=ResourceContract(cycles_max=200, ram_bytes_max=64))
    ctrl.analyze(tree)
    print(ctrl.report())

    print("\n  [PID Controller Analysis]")
    pid = PID(1.0, 0.5, 0.1)
    ctrl_pid = DSOController(pid, name="PID baseline",
                             contract=ResourceContract(cycles_max=100, ram_bytes_max=128))
    ctrl_pid.analyze()
    print(ctrl_pid.report())

    # Run DSO benchmark over RACS results if available
    try:
        import json, pickle, os
        # Check if saved results exist
        if os.path.exists('/tmp/racs_results.pkl'):
            with open('/tmp/racs_results.pkl', 'rb') as f:
                rac_results = pickle.load(f)
            dso_res = dso_benchmark(rac_results)
            print(dso_report_table(dso_res))
    except:
        pass
