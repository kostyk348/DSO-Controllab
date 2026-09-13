#!/usr/bin/env python3
"""
RACS Hardware-in-the-Loop (HIL) on QEMU Cortex-M4
==================================================
Plant + controller both run on the emulated Cortex-M4 (MPS2-AN386).
Measures: real IAE, per-step cycles (WCET), jitter — the final DSO validation.
"""
import os, subprocess, tempfile, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hil_plants import PLANTS_C, PLANT_NAMES

CODEGEN_DIR = os.path.dirname(os.path.abspath(__file__))

MAIN_TEMPLATE = r"""
#include <stdint.h>
extern float ctrl(float e, float ie, float de, float r, float y);
void dbg_init(void); uint32_t dbg_cycles(void);
void dbg_print_str(const char*); void dbg_print_u32(uint32_t);
%s
static void exit_qemu(void){
    register uint32_t r0 asm("r0") = 0x18;
    register uint32_t r1 asm("r1") = 0x20026;
    asm volatile("bkpt 0xAB" : : "r"(r0), "r"(r1) : "memory");
}
int main(void){
    dbg_init(); plant_reset();
    const float Ts = 0.01f;
    const float r  = 1.0f;
    float y=0.0f, u=0.0f, e=r, e_prev=r, ie=0.0f, de=0.0f, iae=0.0f;
    uint32_t cmin=0xFFFFFFFF, cmax=0, csum=0, csq=0;
    for(int i=0;i<%d;i++){
        y = plant_step(u);
        e = r - y;
        de = (e - e_prev)/Ts;
        iae += (e<0?-e:e)*Ts;
        uint32_t c0 = dbg_cycles();
        u = ctrl(e, ie, de, r, y);
        uint32_t c1 = dbg_cycles();
        uint32_t dc = c1 - c0;
        if(dc<cmin) cmin=dc;
        if(dc>cmax) cmax=dc;
        csum += dc; csq += dc*dc;
        if(u>10.0f) u=10.0f;
        if(u<-10.0f) u=-10.0f;
        if(i>=%d) u += %ff;
        ie += e*Ts; e_prev = e;
    }
    dbg_print_str("iae=");      dbg_print_u32((uint32_t)(iae*10000.0f));
    dbg_print_str("cyc_avg=");  dbg_print_u32(csum/%d);
    dbg_print_str("cyc_min=");  dbg_print_u32(cmin);
    dbg_print_str("cyc_max=");  dbg_print_u32(cmax);
    dbg_print_str("cyc_sum=");  dbg_print_u32(csum);
    dbg_print_str("cyc_sq=");   dbg_print_u32(csq);
    dbg_print_str("\n");
    exit_qemu();
    return 0;
}
"""


def make_hil_main(plant_name, n_steps=3000, dist_frac=0.5, dist_mag=-0.5):
    return MAIN_TEMPLATE % (PLANTS_C[plant_name], n_steps,
                            int(n_steps * dist_frac), dist_mag, n_steps)


def run_hil(tree, plant_name, n_steps=3000, timeout=25):
    """Build closed-loop firmware (plant + controller) and run on QEMU."""
    from codegen import tree_to_c
    out_dir = tempfile.mkdtemp(prefix="racs_hil_")
    ctrl_path = os.path.join(out_dir, "ctrl.c")
    main_path = os.path.join(out_dir, "hil_main.c")
    with open(ctrl_path, "w") as f:
        f.write(tree_to_c(tree))
    with open(main_path, "w") as f:
        f.write(make_hil_main(plant_name, n_steps))

    support = os.path.join(CODEGEN_DIR, "support.c")
    startup = os.path.join(CODEGEN_DIR, "startup.s")
    linker = os.path.join(CODEGEN_DIR, "mps2_an386.ld")

    cc = ["clang", "--target=armv7em-none-eabi", "-mcpu=cortex-m4",
          "-mfloat-abi=softfp", "-mfpu=fpv4-sp-d16",
          "-O2", "-ffreestanding", "-fno-builtin", "-c"]

    objs = []
    for name, path in [("startup", startup), ("ctrl", ctrl_path),
                       ("main", main_path), ("support", support)]:
        o = os.path.join(out_dir, name + ".o")
        cmd = (["clang", "--target=armv7em-none-eabi", "-mcpu=cortex-m4",
                "-c", path, "-o", o] if name == "startup" else cc + [path, "-o", o])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return {"error": f"compile {name}: {r.stderr[:300]}"}
        objs.append(o)

    elf = os.path.join(out_dir, "firmware.elf")
    r = subprocess.run(["ld.lld", "-T", linker] + objs + ["-o", elf],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": f"link: {r.stderr[:300]}"}

    try:
        q = subprocess.run(
            ["qemu-system-arm", "-machine", "mps2-an386", "-cpu", "cortex-m4",
             "-nographic", "-semihosting-config", "enable=on,target=native",
             "-kernel", elf, "-no-reboot"],
            capture_output=True, text=True, timeout=timeout)
        out = q.stdout + q.stderr
    except subprocess.TimeoutExpired as ex:
        raw = (ex.stdout or b"") + (ex.stderr or b"")
        out = raw.decode(errors="replace") if isinstance(raw, bytes) else raw

    def g(tag):
        for line in out.splitlines():
            if tag in line:
                try:
                    return int(line.split("=")[1].strip(), 16)
                except (ValueError, IndexError):
                    return None
        return None

    res: dict = {"iae": None, "cyc_avg": None, "cyc_min": None, "cyc_max": None}
    raw_iae = g("iae=")
    if raw_iae is not None:
        res["iae"] = raw_iae / 10000.0
    for k in ("cyc_avg", "cyc_min", "cyc_max"):
        res[k] = g(k + "=")
    res["raw"] = out[-200:]
    res["elf"] = elf
    # jitter from per-step min/max
    if res["cyc_min"] is not None and res["cyc_max"] is not None:
        res["jitter"] = res["cyc_max"] - res["cyc_min"]
    return res


if __name__ == "__main__":
    import pickle
    print("=" * 74)
    print("  RACS Hardware-in-the-Loop — closed loop on QEMU Cortex-M4")
    print("=" * 74)

    with open('/home/lain/racs_nsga_results.pkl', 'rb') as f:
        all_rows = pickle.load(f)

    from racs2 import (SecondOrderDelay, FourthOrder, Underdamped,
                       NonMinPhase, IntegratingDelay, cfg, run_sim)

    PMAP = {
        '2nd+Delay (τ₁=1.5,τ₂=0.8,L=2.0)': (SecondOrderDelay, 'SecondOrderDelay'),
        '4th-Order 1/(s+1)⁴': (FourthOrder, 'FourthOrder'),
        'Underdamped (ζ=0.15,ωₙ=1.5)': (Underdamped, 'Underdamped'),
        'Non-Min Phase (1-2s)/(s+1)³': (NonMinPhase, 'NonMinPhase'),
        'Integrator+Delay (L=1.0)': (IntegratingDelay, 'IntegratingDelay'),
    }

    print(f"\n  {'plant':<22} {'pyIAE':>8} {'hilIAE':>8} {'err%':>6} "
          f"{'WCET':>5} {'model':>6} {'ticks':>6}")
    print("  " + "-" * 72)
    for plant_name, rows in all_rows.items():
        if not rows or plant_name not in PMAP:
            continue
        PlantCls, cname = PMAP[plant_name]
        r = rows[0]
        tree = r['tree']

        # Python reference IAE (same conditions)
        ctrl_fn = lambda rr, y, e, ie, de: tree.evaluate(
            {'e': e, 'ie': ie, 'de': de, 'r': rr, 'y': y})
        py = run_sim(PlantCls, ctrl_fn, cfg.Ts, cfg.t_end)

        from codegen import measure_real_cycles
        wcet, _, _ = measure_real_cycles(tree)

        hil = run_hil(tree, cname, n_steps=3000)
        if hil.get("error"):
            print(f"  {plant_name[:20]:<22} ERROR {hil['error'][:40]}")
            continue
        err = abs(hil['iae'] - py['iae']) / max(1e-9, py['iae']) * 100
        print(f"  {plant_name[:20]:<22} {py['iae']:8.3f} {hil['iae']:8.3f} "
              f"{err:5.1f}% {wcet:5d} {r['cycles']:6d} {hil['cyc_avg']:6d}")

    print("\n  pyIAE=Python IAE | hilIAE=hardware-in-the-loop IAE (QEMU Cortex-M4)")
    print("  WCET=static worst-case cycles | ticks=SysTick/step (QEMU quantized)")
    print("\nDone.")
