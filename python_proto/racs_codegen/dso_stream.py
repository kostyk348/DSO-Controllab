#!/usr/bin/env python3
"""
DSO Streaming Compiler — multi-rate controller pipelines
=========================================================
Compile-time scheduling of N controllers running at different rates.
Everything decided BEFORE runtime: static schedule, static memory, no malloc,
no dynamic dispatch. Feasibility = worst-case per-tick load <= budget.

Key DSO guarantee: the schedule table is computed offline, so execution
order is deterministic and WCET per tick is bounded by construction.
"""
import sys, os, math, subprocess, tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from functools import reduce

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CPU_HZ = 48_000_000          # Cortex-M4 @ 48 MHz
BASE_TS = 0.001              # scheduler base tick = 1 ms


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


@dataclass
class RateTask:
    """One control task: a controller tree + its period + resource contract."""
    name: str
    period_s: float
    tree: object                      # racs2.Node
    contract: object = None           # dso.ResourceContract
    reads: List[str] = field(default_factory=list)   # shared vars consumed
    writes: List[str] = field(default_factory=list)  # shared vars produced


class StreamCompiler:
    """Multi-rate compile-time scheduler for controller pipelines."""

    def __init__(self, tasks: List[RateTask], base_ts: float = BASE_TS,
                 cpu_hz: float = CPU_HZ):
        self.tasks = tasks
        self.base_ts = base_ts
        self.cpu_hz = cpu_hz
        self.result: Optional[Dict] = None

    # ── analysis ──
    def analyze(self) -> Dict:
        periods = [max(1, round(t.period_s / self.base_ts)) for t in self.tasks]
        H = reduce(lcm, periods, 1)

        # schedule table: tick -> tuple of task indices due at this tick
        schedule = []
        for tick in range(H):
            due = tuple(i for i, p in enumerate(periods) if tick % p == 0)
            schedule.append(due)

        # cycles per task
        cyc = [t.tree.cycles_total() for t in self.tasks]

        # per-tick load (cycles)
        tick_load = [sum(cyc[i] for i in due) for due in schedule]
        wcet_tick = max(tick_load) if tick_load else 0

        avail = int(self.base_ts * self.cpu_hz)   # cycles per base tick
        utilization = sum(cyc[i] / (periods[i] * avail) for i in range(len(self.tasks)))
        # average cycles per base tick (deterministic, from schedule)
        avg_load = sum(cyc[i] * (H // periods[i]) for i in range(len(self.tasks))) / H

        # memory: sum of task arenas + shared state
        from dso import MemoryPlanner
        total_ram = 0
        per_task_ram = []
        for t in self.tasks:
            mp = MemoryPlanner()
            mp.analyze_tree(t.tree)
            per_task_ram.append(mp.total_ram())
            total_ram += mp.total_ram()
        shared = sorted(set(v for t in self.tasks for v in (t.reads + t.writes)))
        total_ram += 4 * len(shared)

        feasible = wcet_tick <= avail and utilization < 1.0

        self.result = {
            "periods_ticks": periods,
            "hyperperiod_ticks": H,
            "hyperperiod_s": H * self.base_ts,
            "schedule": schedule,
            "cycles_per_task": cyc,
            "tick_load": tick_load,
            "wcet_tick": wcet_tick,
            "available_per_tick": avail,
            "utilization": utilization,
            "avg_load": avg_load,
            "per_task_ram": per_task_ram,
            "shared_vars": shared,
            "total_ram": total_ram,
            "feasible": feasible,
            "n_schedule_slots": sum(len(d) for d in schedule),
        }
        return self.result

    def report(self) -> str:
        if self.result is None:
            self.analyze()
        r = self.result
        L = []
        L.append("=" * 68)
        L.append("  DSO Streaming Compiler — Multi-Rate Schedule")
        L.append("=" * 68)
        L.append(f"  base tick: {self.base_ts*1000:.2f} ms  |  CPU: {self.cpu_hz/1e6:.0f} MHz")
        L.append(f"  tasks: {len(self.tasks)}  |  hyperperiod: {r['hyperperiod_ticks']} ticks "
                 f"({r['hyperperiod_s']*1000:.1f} ms)")
        L.append("")
        L.append(f"  {'task':<14} {'period':>8} {'cyc':>5} {'RAM':>5} {'util%':>7}")
        L.append("  " + "-" * 44)
        for i, t in enumerate(self.tasks):
            util = r['cycles_per_task'][i] / (r['periods_ticks'][i] * r['available_per_tick']) * 100
            L.append(f"  {t.name:<14} {t.period_s*1000:6.2f}ms {r['cycles_per_task'][i]:5d} "
                     f"{r['per_task_ram'][i]:5d} {util:6.2f}%")
        L.append("")
        L.append(f"  worst-case per tick : {r['wcet_tick']} cyc / {r['available_per_tick']} avail")
        L.append(f"  average per tick    : {r['avg_load']:.1f} cyc "
                 f"({r['avg_load']/r['available_per_tick']*100:.3f}% CPU)")
        L.append(f"  total utilization   : {r['utilization']*100:.2f}%")
        L.append(f"  total RAM           : {r['total_ram']} B  (shared: {', '.join(r['shared_vars']) or '-'})")
        L.append(f"  schedule slots      : {r['n_schedule_slots']}")
        L.append(f"  FEASIBLE            : {'✓ YES' if r['feasible'] else '✗ NO (over budget)'}")
        L.append("=" * 68)
        return "\n".join(L)

    # ── closed-loop multi-rate codegen ──
    def emit_closed_loop_c(self, plant_c: str, task_bodies, n_base_steps=3000,
                           ref=1.0, dist_frac=0.5, dist_mag=-0.3) -> str:
        r = self.analyze()
        H = r["hyperperiod_ticks"]
        masks = []
        for due in r["schedule"]:
            m = 0
            for i in due:
                m |= (1 << i)
            masks.append(m)
        mask_str = ", ".join(f"0x{m:02x}" for m in masks)
        dist_step = int(n_base_steps * dist_frac)
        tasks_c = [f"static void task{i}(void){{\n{b}\n}}" for i, b in enumerate(task_bodies)]
        calls = " ".join(f"if(m&(1<<{i})) task{i}();" for i in range(len(task_bodies)))
        return (
            "#include <stdint.h>\n"
            "void dbg_init(void); uint32_t dbg_cycles(void);\n"
            "void dbg_print_str(const char*); void dbg_print_u32(uint32_t);\n"
            + plant_c + "\n"
            + "\n".join(tasks_c) + "\n"
            f"static const uint8_t SCHED[{H}] = {{ {mask_str} }};\n"
            "static void exit_qemu(void){\n"
            '    register uint32_t r0 asm("r0")=0x18; register uint32_t r1 asm("r1")=0x20026;\n'
            '    asm volatile("bkpt 0xAB" : : "r"(r0),"r"(r1) : "memory");\n'
            "}\n"
            "int main(void){\n"
            "    dbg_init(); plant_reset();\n"
            "    float iae = 0.0f; uint32_t max_load = 0;\n"
            f"    for(int step=0; step<{n_base_steps}; step++){{\n"
            "        uint32_t c0 = dbg_cycles();\n"
            f"        uint8_t m = SCHED[step % {H}];\n"
            f"        {calls}\n"
            "        uint32_t c1 = dbg_cycles();\n"
            "        uint32_t load = c1 - c0; if(load > max_load) max_load = load;\n"
            "        float u = get_u();\n"
            f"        if(step >= {dist_step}) u += {dist_mag}f;\n"
            "        plant_step(u);\n"
            f"        float err = {ref}f - plant_pos();\n"
            "        iae += (err < 0 ? -err : err) * 0.001f;\n"
            "    }\n"
            '    dbg_print_str("iae="); dbg_print_u32((uint32_t)(iae*10000.0f));\n'
            '    dbg_print_str("max_tick_load="); dbg_print_u32(max_load);\n'
            '    dbg_print_str("\\n");\n'
            "    exit_qemu(); return 0;\n"
            "}\n"
        )

    # ── static-schedule codegen (task-load only) ──
    def emit_c(self, plant_c: str, ctrl_exprs: Dict[str, str],
               n_steps: int = 2000) -> str:
        """Generate C firmware with static multi-rate scheduler.

        ctrl_exprs: task name -> C expression body (already generated).
        plant_c: plant C source (plant_step / plant_reset).
        """
        r = self.analyze()
        periods = r["periods_ticks"]
        H = r["hyperperiod_ticks"]

        # schedule bitmask per phase
        masks = []
        for due in r["schedule"]:
            m = 0
            for i in due:
                m |= (1 << i)
            masks.append(m)

        tasks_c = []
        for i, t in enumerate(self.tasks):
            body = ctrl_exprs.get(t.name, "0.0f")
            tasks_c.append(f"static float t{i}_out;\nvoid task{i}(void){{ t{i}_out = {body}; }}")

        mask_str = ", ".join(f"0x{m:02x}" for m in masks)

        return f"""
#include <stdint.h>
void dbg_init(void); uint32_t dbg_cycles(void);
void dbg_print_str(const char*); void dbg_print_u32(uint32_t);
{plant_c}
/* --- generated control tasks --- */
{chr(10).join(tasks_c)}
/* --- static schedule (compile-time, {H} phases) --- */
static const uint8_t SCHED[{H}] = {{ {mask_str} }};
static const uint16_t PERIOD[{len(self.tasks)}] = {{ {', '.join(str(p) for p in periods)} }};
static void exit_qemu(void){{
    register uint32_t r0 asm("r0")=0x18; register uint32_t r1 asm("r1")=0x20026;
    asm volatile("bkpt 0xAB" : : "r"(r0),"r"(r1) : "memory");
}}
int main(void){{
    dbg_init(); plant_reset();
    uint32_t max_load = 0, sum_load = 0;
    for(int phase=0; phase<{H}; phase++){{
        uint32_t c0 = dbg_cycles();
        uint8_t m = SCHED[phase];
        {' '.join(f'if(m&(1<<{i})) task{i}();' for i in range(len(self.tasks)))}
        uint32_t c1 = dbg_cycles();
        uint32_t load = c1 - c0;
        if(load > max_load) max_load = load;
        sum_load += load;
    }}
    dbg_print_str("max_tick_load="); dbg_print_u32(max_load);
    dbg_print_str("avg_tick_load="); dbg_print_u32(sum_load/{H});
    dbg_print_str("\\n");
    exit_qemu(); return 0;
}}
"""
