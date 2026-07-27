# DSO-ControlLab

[![CI](https://github.com/kostyk348/DSO-Controllab/actions/workflows/ci.yml/badge.svg)](https://github.com/kostyk348/DSO-Controllab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![C](https://img.shields.io/badge/C-11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![Ada SPARK](https://img.shields.io/badge/Ada-SPARK-orange.svg)](https://www.adacore.com/about-spark)
[![GNATprove](https://img.shields.io/badge/GNATprove-verified-success.svg)](gp/controller.gpr)

DSO-ControlLab is a small, reproducible research bench for testing the core DSO idea in control systems:

> Move as many decisions as possible before runtime, then execute a verified deterministic plan.

For the broader project direction, see [VISION.md](VISION.md) and [ROADMAP.md](ROADMAP.md).

The current prototype compares:

- `PID`: tuned online by deterministic random search.
- `LQR`: fixed state-feedback controller for linear second-order worlds.
- `MPC`: short-horizon online planner.
- `DSO`: offline-selected fixed controller plan with explicit CPU, memory and jitter contract.

It generates random second-order worlds, runs every controller on each world, measures control quality and runtime uncertainty, verifies DSO plans against deployment contracts, then prints aggregate statistics.

## Quick Start

```bash
python3 -m dso_controllab --worlds 1000 --seed 7
```

For a fast smoke run:

```bash
python3 -m dso_controllab --worlds 50 --seed 1
```

## Metrics

- `iae`: integral absolute error, lower is better.
- `overshoot`: maximum response above the target.
- `energy`: mean squared control effort.
- `wcet_us`: estimated worst-case execution time.
- `jitter_us`: estimated runtime timing spread.
- `contract_pass_rate`: fraction of worlds satisfying the DSO contract.
- `score`: combined quality/resource cost.

## DSO Contract

The default deployment contract is intentionally simple:

Resource layer:

- CPU cycles per step: `<= 180`
- RAM bytes: `<= 96`
- WCET: `<= 8 us`
- jitter: `<= 0.8 us`

Control layer:

- IAE: `<= 4.0`
- overshoot: `<= 0.9`
- max absolute output: `<= 8.0`
- final error: `<= 1.25`

Runtime layer:

- saturation fraction: `<= 45%`
- no NaN / Inf

See [CONTRACTS.md](CONTRACTS.md) for the design model.

This makes the difference visible: MPC may improve quality on some worlds, but it performs more runtime search. DSO compiles the selected controller into a smaller fixed plan with bounded memory and timing, and rejects it if verification fails.

## GP Controller Synthesis (`gp/`)

Genetic programming engine in C that evolves compact control laws with formal
verification via Ada SPARK / GNATprove.

```bash
cd gp/
mkdir build && cd build
cmake .. && make
./dso_gp --pop 100 --gen 50 --worlds 30 --steps 300 --seed 7 --benchmark 100
```

### How it works

- **Expression tree** (63 nodes max, depth 4-6): terminals `{Error, Integral, Deriv, Y, 1.0}`,
  operators `{+, −, ×, /, abs, neg, clamp}`
- **Evolution**: population 100, tournament selection (size 7), subtree crossover (70%),
  point mutation (30%), elitism (top 2)
- **Fitness**: simulation over 300 steps on 30 random worlds → `IAE + overshoot + energy`
  with anti-bloat penalty (configurable via `--bloat`)
- **OpenMP** parallel fitness evaluation (~4× speedup on 4 cores)

### All Controllers Ported to C

LQR (DARE iteration), MPC (horizon-8 brute force), DSO (GP + contract verification) —
all ported from Python to C with matching resource metrics and score function.
The 5-way benchmark is a single binary, no Python dependency.

### CLI

| Flag | Description | Default |
|---|---|---|
| `--pop N` | population size | 100 |
| `--gen N` | generations | 50 |
| `--seed N` | RNG seed | time |
| `--worlds N` | training worlds per eval | 30 |
| `--steps N` | simulation steps | 500 |
| `--bloat F` | anti-bloat per node | 0.02 |
| `--benchmark N` | **5-way** benchmark on N worlds | — |
| `--sweep START N` | multi-seed sweep → CSV | — |
| `--gen-log FILE` | save gen-by-gen fitness as CSV | — |
| `--csv-out FILE` | save benchmark as CSV | — |
| `--plot FILE` | gnuplot convergence script | — |
| `--stability N` | stability analysis on N worlds | — |
| `--export-ada NAME` | export best as Ada SPARK | — |
| `--export-c FILE` | export best as C | — |
| `--verify` | run GNATprove on exported Ada | — |
| `--json` | JSON output | — |

### 5-Way Benchmark: GP vs PID vs LQR vs MPC vs DSO

100 random worlds, pop=100, gen=50, steps=300, dt=0.02, seed=7:

```
Ctrl          IAE     Overshoot   Energy     Sat%      Score    WCETus   Jitus   Cyc  RAM   Ctr%
------ ----------  ---------- ---------- ------- ---------- ------- ------- ----- ---- ------
GP       1.572439   0.105827   8.170317  31.35%   2.126792   0.88  0.095   42   40  71.0%
PID      2.656521   0.045307   3.440760   0.14%   3.000509   0.88  0.095   42   40  91.0%
LQR      2.111281   0.000200   8.122687  27.04%   2.666758   1.21  0.095   58   64  82.0%
MPC      5.238047   0.000000   0.128133   0.00%   7.324172  12.92  0.590  620 176   0.0%
DSO      1.463083   0.038820   5.922580  12.16%   1.829573   0.67  0.040   32   36 100.0%
```

**GP beats every conventional controller** (DSO ≈ GP verified):
- GP vs PID: **+29.1%** (p < 0.000001)
- GP vs LQR: **+20.2%** (p < 0.000001)
- GP vs MPC: **+71.0%** (p < 0.000001)
- GP vs DSO: **−3.1%** (DSO wins by lower resource cost: 32 vs 42 cycles)

**DSO is now GP + contract verification**: the evolved controller is verified
against the deployment contract on each world. If it passes, it executes with
DSO-level resources (32 cyc, 0 branch). If it fails, it falls back to the best
PID. This gives DSO both the quality of GP and the reliability of classical
control.

### Stability Analysis (`--stability N`)

Monte Carlo robustness test: perturb each plant parameter (wn, ζ, gain, delay)
by ±30%, run long simulation (2× steps), detect instability:

```
========== STABILITY ANALYSIS: GP ==========
  Worlds tested:         50
  Stable:                50/50 (100.0%)
  Oscillatory:           7/50 (14.0%)
  Well-settled:          36/50 (72.0%)
  Worst-case IAE:        97.18
  Max oscillation ratio: 0.78
  Est. gain margin:      10.00×
============================================
```

- **Stable**: finite IAE, no divergence (|y| < 100)
- **Oscillatory**: IAE in last 25% > 30% of total (sustained oscillations)
- **Well-settled**: final |error| < 0.05
- **Gain margin**: plant gain multiplier before instability (swept 1×–10×)

### Evolved Controller Example

```
(+ (- DER (/ -1.9503 (abs Y))) (+ (- INT Y) (- INT Y)))
```

14 nodes. Uses `(Integral - Y)` as integral error proxy and `Deriv / abs(Y)`
as nonlinear damping — no explicit `Error` terminal needed.

### Ada SPARK Formal Verification

```bash
./dso_gp --pop 100 --gen 50 --seed 7 --export-ada controller
gnatprove -P controller.gpr --level=2
# → Success: all checks proved (12 checks)
```

The generated code includes:
- `Safe_Div(X, Y)` wrapper — proven division-by-zero-free
- Preconditions on inputs (`Error, Integral, Deriv, Y in -10.0..10.0`)
- Postcondition (`Compute'Result in -4.0..4.0`)
- Output clamping to [-4.0, 4.0]

### Multi-Seed Sweep

```
./dso_gp --sweep 1 10 --pop 80 --gen 30 --worlds 20 2>/dev/null
seed,gen,worlds,steps,fitness,tree_size,best_controller
1,30,20,200,2.388814,5,   "(sq (sq (+ INT ERROR)))"
2,30,20,200,1.919553,15,  "(+ (+ (+ (+ INT ERROR) ERROR) ERROR) ...)"
3,30,20,200,2.476859,3,   "(- ERROR -0.9902)"
4,30,20,200,1.444131,11,  "(+ DER (+ DER (* (+ DER (sq (sq 2.1055))) ERROR)))"
5,30,20,200,1.703769,9,   "(+ (/ (+ ERROR INT) 0.3199) (+ DER ERROR))"
```

## Project Shape

```text
gp/                        Genetic programming (C11)
  gp.h                     GP tree + evolution API
  plant.c                  second-order plant model
  gp_tree.c                expression trees: alloc, eval, mutate, crossover
  gp_evolve.c              evolution loop + fitness (OpenMP)
  gp_export.c              C/Ada export
  controllers.h            LQR / MPC / DSO controller API
  controllers.c            LQR (DARE), MPC (horizon-8), DSO (PID bank)
  main.c                   CLI + 5-way benchmark + sweep
  controller.gpr           GNATprove project file

src/dso_controllab/        Python research prototype
  cli.py                   command-line runner
  contracts.py             resource/control/runtime/deployment contracts
  controllers.py           PID/LQR/MPC/DSO controllers
  experiment.py            random worlds, simulation, summary stats
  metrics.py               scoring and statistical helpers
  verifier.py              pre-deployment verification pass
  world.py                 plant model

tests/
  test_smoke.py
```

## Metrics

- `iae`: integral absolute error, lower is better.
- `overshoot`: maximum response above the target.
- `energy`: mean squared control effort.
- `wcet_us`: estimated worst-case execution time (`cycles / 48.0`).
- `jitter_us`: estimated runtime timing spread (`0.04 + 0.055 * branch_points`).
- `contract_pass_rate`: fraction of worlds satisfying the DSO contract.
- `score`: combined quality/resource cost: `iae + 0.35×overshoot + 0.04×energy + 0.12×wcet + 0.9×jitter`.

## DSO Contract

| Layer | Constraint | Value |
|---|---|---|
| Resource | CPU cycles ≤ | 180 |
| | RAM bytes ≤ | 96 |
| | WCET ≤ | 8 µs |
| | jitter ≤ | 0.8 µs |
| Control | IAE ≤ | 4.0 |
| | overshoot ≤ | 0.9 |
| | max |u| ≤ | 8.0 |
| | final error ≤ | 1.25 |
| Runtime | saturation ≤ | 45% |
| | NaN/Inf | prohibited |

## Next Research Steps

- ~~1. Add genetic programming controller synthesis.~~ ✓
- ~~2. Port LQR/MPC/DSO to C — full 5-way benchmark.~~ ✓
- ~~3. Multi-seed GP analysis.~~ ✓
- 4. Add 1000-world CSV export and plots.
- 5. Add verification traces: bounds, saturation, failure modes.
- 6. Replace estimated resource costs with measured embedded targets.
- **7. Auto-GNATprove pipeline** — verify every candidate, not just the best.
- **8. ARM cross-compilation** — measure real WCET/jitter on ESP32/STM32.
