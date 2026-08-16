#!/usr/bin/env python3
"""
RACS + DSO — Full-Stak Benchmark
==================================
1. Run RACS controllers (PID, GP, GP-R, LQR, MPC) on all plants
2. Wrap each in DSO analysis (Memory Planner, Determinism, Contracts)
3. Compare conventional vs DSO-constrained search
4. Generate integrated report
"""

import sys, os, time, pickle, math
sys.path.insert(0, '.')

from racs2 import *
from dso import *

# ──────────────────────────────────────────────
#  DSO-Constrained GP
# ──────────────────────────────────────────────

class DSOGPController(GPController):
    """GP with DSO resource contracts as hard constraints.
    PRIMARY criterion: variance (determinism).
    Secondary: IAE (control quality)."""

    def __init__(self, plant_class, Ts, t_end, contract: ResourceContract):
        super().__init__(plant_class, Ts, t_end, resource_penalty=True)
        self.contract = contract
        self.dso_analyzer = DeterminismAnalyzer()
        self.mem_planner = MemoryPlanner()

    def _fit(self, tree):
        try:
            if tree.depth() > cfg.max_depth:
                return -float('inf')

            ctrl_fn = self._ctrl(tree)

            # 1. Simulate (single run for IAE)
            res = run_sim(self.plant_class, ctrl_fn, self.Ts, self.t_end)

            # 2. DSO static analysis: fast determinism proxy
            mp = MemoryPlanner()
            mp.analyze_tree(tree)
            da = DeterminismAnalyzer()
            det = da.analyze(tree, mp)

            # 3. Check contract
            passed, penalty, violations = self.contract.check(
                cycles=det['cycles'],
                ram=det['ram_bytes'],
                latency_us=det['wcet_us'],
                jitter_ns=det['jitter_ns']
            )

            # ── DSO COST: determinism PRIMARY (via static analysis) ──
            # Static determinism is a fast proxy for variance:
            # - branches (min/max) → jitter
            # - nonlinear ops (sin/cos/sqrt) → iteration variance
            # - memory pressure → cache miss variance
            det_deficit = (100.0 - det['determinism_score']) / 100.0  # 0..1, higher = worse

            # Control quality cost (SECONDARY)
            quality_cost = res['iae'] / 15.0

            # Resource cost (TERTIARY)
            resource_cost = 0.05 * det['cycles'] / 100 + 0.03 * det['ram_bytes'] / 20

            # Contract penalty
            contract_cost = penalty * 2.0

            total_cost = (
                4.0 * det_deficit +       # PRIMARY: determinism (variance proxy)
                1.0 * quality_cost +      # SECONDARY: IAE
                0.5 * resource_cost +     # TERTIARY: resources
                2.0 * contract_cost       # PENALTY: contract violations
            )

            # Store Pareto (IAE, cycles, determinism)
            self.pareto_front.append((
                res['iae'], det['cycles'], det['ram_bytes'],
                det['determinism_score'], tree.clone()
            ))

            return 1.0 / (1.0 + total_cost)
        except Exception as e:
            return -float('inf')


# ──────────────────────────────────────────────
#  Full Benchmark
# ──────────────────────────────────────────────

def run_full_benchmark():
    print("=" * 70)
    print("  RACS + DSO — Deterministic Controller Synthesis")
    print("  PID / LQR / MPC / GP / GP-R / DSO-GP")
    print("  + Memory Planner + Resource Contracts + Determinism")
    print("=" * 70)

    plants = [SecondOrderDelay, FourthOrder, Underdamped,
              NonMinPhase, IntegratingDelay]

    # Lightweight DSO contract
    dso_contract = ResourceContract(
        cycles_max=80,
        ram_bytes_max=48,
        latency_us_max=100,
    )

    rac_results = []  # for DSO analysis later

    for PlantCls in plants:
        plant = PlantCls(Ts=cfg.Ts)
        print(f"\n{'='*70}\n  {plant.name}\n{'='*70}")

        # ── PID ──
        pid = optimize_pid(PlantCls, cfg.Ts, cfg.t_end)
        pid_res = run_sim(PlantCls, lambda r,y,e,ie,de: pid.compute(r,y,e,ie,de), cfg.Ts, cfg.t_end)
        print(f"  PID:  IAE={pid_res['iae']:.4f}  Kp={pid.Kp:.3f} Ki={pid.Ki:.3f} Kd={pid.Kd:.3f}")

        # ── LQR (if SS available) ──
        lqr_res = None; lqr_ctrl = None
        ss = PlantCls(Ts=cfg.Ts).get_ss()
        if ss is not None:
            A,B,C,D = ss
            try:
                lqr_ctrl = LQR(A,B,C,D,cfg.Ts)
                lqr_res = run_sim(PlantCls, lambda r,y,e,ie,de: lqr_ctrl.compute(r,y,e,ie,de), cfg.Ts, cfg.t_end)
                print(f"  LQR:  IAE={lqr_res['iae']:.4f}")
            except: pass

        # ── MPC ──
        mpc_ctrl = MPC(PlantCls, cfg.Ts, N=5)
        mpc_res = run_sim(PlantCls, lambda r,y,e,ie,de: mpc_ctrl.compute(r,y,e,ie,de), cfg.Ts, cfg.t_end)
        print(f"  MPC:  IAE={mpc_res['iae']:.4f}")

        # ── GP ──
        gp = GPController(PlantCls, cfg.Ts, cfg.t_end, resource_penalty=False)
        gp_tree, gp_fit, gp_hist, gp_pareto = gp.run()
        gp_res = run_sim(PlantCls, gp._ctrl(gp_tree), cfg.Ts, cfg.t_end)
        print(f"  GP:   IAE={gp_res['iae']:.4f}  cycles={gp_tree.cycles()}")

        # ── GP-R ──
        gpr = GPController(PlantCls, cfg.Ts, cfg.t_end, resource_penalty=True)
        gpr_tree, gpr_fit, gpr_hist, gpr_pareto = gpr.run()
        gpr_res = run_sim(PlantCls, gpr._ctrl(gpr_tree), cfg.Ts, cfg.t_end)
        print(f"  GP-R: IAE={gpr_res['iae']:.4f}  cycles={gpr_tree.cycles()}")

        # ── DSO-GP ──
        print("  [DSO-GP]")
        dso_gp = DSOGPController(PlantCls, cfg.Ts, cfg.t_end, dso_contract)
        dso_tree, dso_fit, dso_hist, dso_pareto = dso_gp.run()
        dso_res = run_sim(PlantCls, dso_gp._ctrl(dso_tree), cfg.Ts, cfg.t_end)

        # DSO analysis (static + variance)
        mp = MemoryPlanner(); mp.analyze_tree(dso_tree)
        da = DeterminismAnalyzer()
        det = da.analyze(dso_tree, mp)
        var = da.measure_variance(
            dso_tree, PlantCls, cfg.Ts, cfg.t_end,
            n_runs=6, ctrl_fn=dso_gp._ctrl(dso_tree)
        )
        det.update(var)
        passed, penalty, viol = dso_contract.check(
            det['cycles'], det['ram_bytes'], det['wcet_us'], det['jitter_ns'])
        print(f"  DSO-GP: IAE={dso_res['iae']:.4f}  cycles={det['cycles']}  "
              f"RAM={det['ram_bytes']}B  WCET={det['wcet_us']:.1f}μs  "
              f"Det(static)={det['determinism_score']:.0f}/100  "
              f"Det(var)={det['determinism_variance_score']:.0f}/100  "
              f"Var_out={det['output_variance']:.2e}  "
              f"Contract={'✓' if passed else '✗'}")
        print(f"    Formula: {tree_formula(dso_tree)}")

        rac_results.append({
            'plant': plant.name,
            'pid': pid_res, 'pid_ctrl': pid,
            'lqr': lqr_res, 'lqr_ctrl': lqr_ctrl,
            'mpc': mpc_res, 'mpc_ctrl': mpc_ctrl,
            'gp': gp_res, 'gp_tree': gp_tree,
            'gpr': gpr_res, 'gpr_tree': gpr_tree,
            'dso': dso_res, 'dso_tree': dso_tree,
            'dso_metrics': det,
            'dso_contract_passed': passed,
        })

    return rac_results


# ──────────────────────────────────────────────
#  DSO Post-Processing
# ──────────────────────────────────────────────

def dso_post_process(rac_results):
    """Wrap all results in DSO analysis and print report (variance-centric)."""
    dso_results = dso_benchmark(rac_results)

    print("\n" + "=" * 110)
    print("  DSO DETAILED ANALYSIS (VARIANCE-CENTRIC)")
    print("=" * 110)

    for entry in dso_results:
        for tag in ['PID', 'LQR', 'MPC', 'GP', 'GP-R']:
            ctrl = entry.get(tag)
            if ctrl and ctrl._analyzed:
                print(f"\n  {entry['plant'][:26]} — {tag}:")
                print(ctrl.report())

    print(dso_report_table(dso_results))

    # DSO-GP specific variance report
    print("\n" + "=" * 110)
    print("  DSO-GP VARIANCE ANALYSIS (PRIMARY CRITERION)")
    print("=" * 110)
    for res in rac_results:
        m = res.get('dso_metrics', {})
        plant = res['plant'][:30]
        out_var = m.get('output_variance', 'N/A')
        iae_var = m.get('iae_variance', 'N/A')
        iae_std = m.get('iae_std', 'N/A')
        det_var = m.get('determinism_variance_score', 'N/A')
        passed = '✓' if res.get('dso_contract_passed', False) else '✗'
        print(f"\n  {plant}")
        print(f"    Output variance: {out_var}")
        print(f"    IAE variance:    {iae_var}")
        print(f"    IAE std:         {iae_std}")
        print(f"    Det from var:    {det_var}/100")
        print(f"    Contract:        {passed}")
        print(f"    Formula:         {tree_formula(res['dso_tree'])}" if 'dso_tree' in res else "")

    return dso_results


# ──────────────────────────────────────────────
#  Summary Table
# ──────────────────────────────────────────────

def print_summary(results):
    da = DeterminismAnalyzer()

    print("\n\n" + "=" * 160)
    print("  FULL BENCHMARK: RACS + DSO (VARIANCE-CENTRIC)")
    print("=" * 160)

    hdr = (f"{'Process':<26} {'PID-IAE':<9} {'GP-IAE':<9} {'DSO-IAE':<9} "
           f"{'DSOcyc':<7} {'DSOram':<7} {'WCETμs':<7} {'Var_out':<11} "
           f"{'Var_IAE':<11} {'DetVar':<7} {'Contr':<6}")
    print(f"  {hdr}")
    print("  " + "-" * 131)

    for res in results:
        plant = res['plant'][:24]
        pid_iae = f"{res['pid']['iae']:.3f}"
        gp_iae = f"{res['gp']['iae']:.3f}"
        dso_iae = f"{res['dso']['iae']:.3f}"
        dso_cyc = str(res['dso_metrics']['cycles'])
        dso_ram = str(res['dso_metrics']['ram_bytes'])
        wcet = f"{res['dso_metrics']['wcet_us']:.1f}"

        # Variance metrics for DSO-GP
        out_var = res['dso_metrics'].get('output_variance', 0)
        iae_var = res['dso_metrics'].get('iae_variance', 0)
        det_var = res['dso_metrics'].get('determinism_variance_score', 100)

        var_out_s = f"{out_var:.2e}"
        var_iae_s = f"{iae_var:.2e}"
        det_var_s = f"{det_var:.0f}"
        ctr = '✓' if res['dso_contract_passed'] else '✗'
        print(f"  {plant:<26} {pid_iae:<9} {gp_iae:<9} {dso_iae:<9} "
              f"{dso_cyc:<7} {dso_ram:<7} {wcet:<7} {var_out_s:<11} "
              f"{var_iae_s:<11} {det_var_s:<7} {ctr:<6}")

    print("=" * 160)
    print("  Var_out = output variance  |  Var_IAE = IAE variance across runs")
    print("  DetVar  = determinism-from-variance (100 = perfect zero variance)")


# ──────────────────────────────────────────────
#  Plot
# ──────────────────────────────────────────────

def plot_dso(results, save_dir='.'):
    """Generate DSO comparison plot."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import numpy as np

    n = len(results)
    fig = plt.figure(figsize=(18, 5*n))

    for idx, res in enumerate(results):
        gs = gridspec.GridSpecFromSubplotSpec(3, 3,
              subplot_spec=fig.add_gridspec(n, 1)[idx],
              width_ratios=[2, 1.5, 1], hspace=0.3, wspace=0.35)

        # Step response (all controllers)
        ax1 = fig.add_subplot(gs[:, 0])
        for tag, key in [('PID', 'pid'), ('GP', 'gp'), ('GP-R', 'gpr'),
                         ('DSO-GP', 'dso')]:
            if res.get(key):
                ax1.plot(res[key]['t'], res[key]['y'], lw=1.5, alpha=0.8, label=tag)
        ax1.axhline(1, color='gray', ls='--', alpha=0.5)
        ax1.axvline(15, color='gray', ls=':', alpha=0.3)
        ax1.set_ylabel('y(t)'); ax1.set_title(f"{res['plant']}")
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

        # Metrics bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        tags = ['PID', 'GP', 'GP-R', 'DSO-GP']
        iae_vals = [res[t]['iae'] if res.get(t) else 0 for t in ['pid', 'gp', 'gpr', 'dso']]
        x = np.arange(len(tags)); w = 0.35
        ax2.bar(x, iae_vals, w, color='steelblue', alpha=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(tags, fontsize=8)
        ax2.set_ylabel('IAE'); ax2.set_title('Control Quality'); ax2.grid(True, alpha=0.3, axis='y')

        # Resources: cycles + RAM
        ax3 = fig.add_subplot(gs[1, 1])
        cyc_vals = [20, res['gp_tree'].cycles(), res['gpr_tree'].cycles(),
                    res['dso_metrics']['cycles']]
        ax3.bar(x, cyc_vals, w, color='crimson', alpha=0.7)
        ax3.set_xticks(x); ax3.set_xticklabels(tags, fontsize=8)
        ax3.set_ylabel('Cycles'); ax3.set_title('Compute Cost'); ax3.grid(True, alpha=0.3, axis='y')

        # Determinism + Resources
        ax4 = fig.add_subplot(gs[2, 1])
        det_vals = [95, res['dso_metrics']['determinism_score'],
                    res['dso_metrics']['determinism_score'],
                    res['dso_metrics']['determinism_score']]
        # Actually compute for each controller
        da = DeterminismAnalyzer()
        det_vals = []
        for tree in [None, res['gp_tree'], res['gpr_tree'], res['dso_tree']]:
            if tree is None:
                mp = MemoryPlanner(); mp.analyze_pid(res['pid_ctrl'])
                det_vals.append(da.analyze_pid(mp)['determinism_score'])
            else:
                mp = MemoryPlanner(); mp.analyze_tree(tree)
                det_vals.append(da.analyze(tree, mp)['determinism_score'])
        ax4.bar(x, det_vals, w, color='green', alpha=0.7)
        ax4.set_xticks(x); ax4.set_xticklabels(tags, fontsize=8)
        ax4.set_ylabel('Score/100'); ax4.set_title('Determinism'); ax4.grid(True, alpha=0.3, axis='y')

        # Pareto: IAE vs Determinism
        ax5 = fig.add_subplot(gs[0, 2])
        ax5.scatter(iae_vals, det_vals, c=['blue', 'red', 'green', 'purple'],
                    s=100, alpha=0.8)
        ax5.set_xlabel('IAE (lower better)'); ax5.set_ylabel('Determinism (higher better)')
        ax5.set_title('Quality vs Determinism'); ax5.grid(True, alpha=0.3)

        # Pareto: IAE vs Cycles
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.scatter(iae_vals, cyc_vals, c=['blue', 'red', 'green', 'purple'],
                    s=100, alpha=0.8)
        ax6.set_xlabel('IAE'); ax6.set_ylabel('Cycles (lower better)')
        ax6.set_title('Quality vs Cost'); ax6.grid(True, alpha=0.3)

        # Formula + contract
        ax7 = fig.add_subplot(gs[2, 2]); ax7.axis('off')
        lines = [
            f"GP: {tree_formula(res['gp_tree'])}",
            f"GP-R: {tree_formula(res['gpr_tree'])}",
            f"DSO-GP: {tree_formula(res['dso_tree'])}",
            f"Contract: ≤80cyc ≤48B RAM",
            f"Pass: {'YES' if res['dso_contract_passed'] else 'NO'}"
        ]
        for i, line in enumerate(lines):
            ax7.text(0.05, 0.85 - i*0.2, line, fontsize=7, family='monospace',
                     bbox=dict(boxstyle='round', fc='lightyellow' if 'DSO' in line or 'Contract' in line or 'Pass' in line else 'white', alpha=0.8))

    fig.suptitle('RACS + DSO — Deterministic Controller Synthesis', fontsize=14, y=1.005)
    plt.tight_layout()
    path = os.path.join(save_dir, 'racs_dso_benchmark.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()
    results = run_full_benchmark()
    elapsed = time.time() - t0

    print(f"\n\nRACS runtime: {elapsed:.0f}s")

    print_summary(results)
    dso_post_process(results)

    try:
        plot_dso(results)
    except Exception as e:
        print(f"Plot error: {e}")

    # Save results
    with open('/tmp/racs_dso_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nDone. Results saved to /tmp/racs_dso_results.pkl")
