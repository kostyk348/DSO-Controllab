#!/usr/bin/env python3
"""DSO streaming demo: 2-rate cascade (outer 100 Hz + inner 1 kHz) on QEMU Cortex-M4."""
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dso_stream import RateTask, StreamCompiler

HERE = os.path.dirname(os.path.abspath(__file__))

CASCADE_PLANT = r"""
/* Plant: 1/(s^2 + 0.5 s) — position from force */
static float p_pos, p_vel;
static void plant_reset(void){ p_pos=0; p_vel=0; }
static void plant_step(float u){
    float dvel = -0.5f*p_vel + u;
    p_vel += dvel*0.001f;      /* base Ts = 1 ms */
    p_pos += p_vel*0.001f;
}
static float plant_pos(void){ return p_pos; }
static float vref = 0.0f, u_cmd = 0.0f;
static float eo = 0.0f, ieo = 0.0f, ei = 0.0f, iei = 0.0f;
static float get_u(void){ return u_cmd; }
"""

OUTER_BODY = r"""
    eo = 1.0f - plant_pos();
    ieo += eo * 0.01f;
    vref = 4.0f*eo + 1.0f*ieo - 2.0f*p_vel;
"""

INNER_BODY = r"""
    ei = vref - p_vel;
    iei += ei * 0.001f;
    u_cmd = 3.0f*ei + 5.0f*iei;
"""


def build_run(src, tag, timeout=20):
    out_dir = tempfile.mkdtemp(prefix="racs_stream_")
    fw = os.path.join(out_dir, "stream.c")
    with open(fw, "w") as f:
        f.write(src)
    cc = ["clang", "--target=armv7em-none-eabi", "-mcpu=cortex-m4",
          "-mfloat-abi=softfp", "-mfpu=fpv4-sp-d16", "-O2",
          "-ffreestanding", "-fno-builtin", "-c"]
    objs = []
    for name, path in [("startup", os.path.join(HERE, "startup.s")),
                       ("stream", fw),
                       ("support", os.path.join(HERE, "support.c"))]:
        o = os.path.join(out_dir, name + ".o")
        cmd = (["clang", "--target=armv7em-none-eabi", "-mcpu=cortex-m4", "-c", path, "-o", o]
               if name == "startup" else cc + [path, "-o", o])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [{tag}] compile {name} FAILED: {r.stderr[:200]}")
            return None
        objs.append(o)
    elf = os.path.join(out_dir, "firmware.elf")
    r = subprocess.run(["ld.lld", "-T", os.path.join(HERE, "mps2_an386.ld")]
                       + objs + ["-o", elf], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [{tag}] link FAILED: {r.stderr[:200]}")
        return None
    try:
        q = subprocess.run(["qemu-system-arm", "-machine", "mps2-an386",
                            "-cpu", "cortex-m4", "-nographic",
                            "-semihosting-config", "enable=on,target=native",
                            "-kernel", elf, "-no-reboot"],
                           capture_output=True, text=True, timeout=timeout)
        out = q.stdout + q.stderr
    except subprocess.TimeoutExpired as e:
        raw = (e.stdout or b"") + (e.stderr or b"")
        out = raw.decode(errors="replace") if isinstance(raw, bytes) else raw

    res = {}
    for line in out.splitlines():
        for k in ("iae=", "max_tick_load="):
            if line.startswith(k):
                try:
                    res[k[:-1]] = int(line.split("=")[1].strip(), 16)
                except (ValueError, IndexError):
                    pass
    return res


def main():
    from racs2 import Node
    from dso import ResourceContract

    def mk(vals):
        n = Node('const', val=vals[0])
        for v in vals[1:]:
            n = Node('add', left=n, right=Node('const', val=v))
        return n

    contract = ResourceContract(cycles_max=80, ram_bytes_max=48)

    print("=" * 68)
    print("  DSO Streaming Compiler — 2-rate cascade demo (QEMU Cortex-M4)")
    print("=" * 68)

    # --- multi-rate: outer 100 Hz + inner 1 kHz ---
    multi = StreamCompiler([
        RateTask('outer_pos', 0.01, mk([4.0, 1.0, 2.0]), contract,
                 reads=['pos', 'vel'], writes=['vref']),
        RateTask('inner_vel', 0.001, mk([3.0, 5.0]), contract,
                 reads=['vref', 'vel'], writes=['u']),
    ])
    print(multi.report())
    src_multi = multi.emit_closed_loop_c(CASCADE_PLANT, [OUTER_BODY, INNER_BODY], 3000)
    res_multi = build_run(src_multi, "multi")
    if res_multi:
        print(f"\n  MULTI-RATE (100 Hz + 1 kHz):  IAE={res_multi.get('iae',0)/10000:.4f}  "
              f"max_tick_load={res_multi.get('max_tick_load')} cyc")

    # --- single-rate baseline: both at 1 kHz ---
    single = StreamCompiler([
        RateTask('outer_pos', 0.001, mk([4.0, 1.0, 2.0]), contract,
                 reads=['pos', 'vel'], writes=['vref']),
        RateTask('inner_vel', 0.001, mk([3.0, 5.0]), contract,
                 reads=['vref', 'vel'], writes=['u']),
    ])
    print("\n" + single.report())
    src_single = single.emit_closed_loop_c(CASCADE_PLANT, [OUTER_BODY, INNER_BODY], 3000)
    res_single = build_run(src_single, "single")
    if res_single:
        print(f"\n  SINGLE-RATE (both 1 kHz):     IAE={res_single.get('iae',0)/10000:.4f}  "
              f"max_tick_load={res_single.get('max_tick_load')} cyc")

    # --- comparison ---
    if res_multi and res_single:
        print("\n" + "=" * 68)
        print("  COMPARISON")
        print("=" * 68)
        im, isg = res_multi['iae']/10000, res_single['iae']/10000
        am = multi.result['avg_load']
        asg = single.result['avg_load']
        print(f"  multi-rate : IAE={im:.4f}  avg_load={am:.1f} cyc/tick")
        print(f"  single-rate: IAE={isg:.4f}  avg_load={asg:.1f} cyc/tick")
        print(f"  → multi-rate: IAE {100*(im-isg)/max(1e-9,isg):+.1f}%, "
              f"avg CPU {100*(am-asg)/max(1e-9,asg):+.1f}%")
        print("  (multi-rate runs slow loop at its design rate → better control;")
        print("   it runs 10x less often → lower average CPU)")
    print("\nDone.")


if __name__ == "__main__":
    main()
