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

Genetic programming engine in C that evolves compact control laws — the first step
toward offline formal verification of learned controllers.

```bash
cd gp/
mkdir build && cd build
cmake .. && make
./dso_gp --pop 100 --gen 50 --worlds 200 --seed 7
```

### How it works

- **Expression tree** (63 nodes max, depth 4-6): terminals `{Error, Integral, Deriv, Y, 1.0}`,
  operators `{+, −, ×, /, abs, neg, clamp}`
- **Evolution**: population 100, tournament selection (size 7), subtree crossover (70%),
  point mutation (30%), elitism (top 2)
- **Fitness**: simulation over 300 steps on 30 random worlds → `IAE + overshoot + energy`
  with anti-bloat penalty (configurable via `--bloat`)
- **OpenMP** parallel fitness evaluation (~4× speedup on 4 cores)

### CLI

```
--pop N         population size (default 100)
--gen N         generations (default 50)
--seed N        RNG seed (default time)
--worlds N      training worlds (default 30)
--steps N       simulation steps (default 300)
--bloat F       size penalty per node (default 0.02)
--benchmark N   compare best GP vs best PID over N worlds
--export-ada N  export best controller as Ada SPARK (named N)
--export-c N    export best controller as C (named N)
```

### Ada SPARK Formal Verification

The `--export-ada` flag generates a verified Ada SPARK controller:

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

### Benchmark: GP vs PID

```text
Metric               GP         Best PID    Δ
───────────────────────────────────────────────
Score (mean)         1.90       2.81        +32.5%
IAE                  1.53       2.66        -42.5%
Overshoot            0.09       0.05        +80.0%
Energy               8.10       3.44       +135.5%
Contract pass rate   71%        91%         -22.0%
Sign test p-value    <0.000001   —           —
```

GP finds compact controllers (~11-15 nodes) that beat PID on IAE and composite score,
at the cost of higher energy and overshoot. The contract pass rate reflects RMS
output constraint violations — configurable in post-processing.

### Evolved Controller Example

```
((Deriv - Safe_Div(-1.950344, abs(Y))) + ((Integral - Y) + (Integral - Y)))
```

15 nodes, no `Error` terminal — the GP discovered that `(Integral - Y)` (integral
error) plus a nonlinear `Deriv / abs(Y)` term works better than explicit PID.

## Project Shape

```text
dso_gp/                    ← NEW: genetic programming (C11)
  gp.h                     GP tree + evolution API
  plant.c                  second-order plant model (C port)
  gp_tree.c                expression trees: alloc, eval, mutate, crossover
  gp_evolve.c              evolution loop + fitness (OpenMP)
  gp_export.c              C/Ada export + benchmark harness
  main.c                   CLI entry point
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

## Next Research Steps

- ~~1. Add genetic programming controller synthesis.~~ ✓
- 2. Add 1000-world CSV export and plots.
- 3. Add paired statistical tests against PID/LQR/MPC.
- 4. Add verification traces: bounds, saturation, failure modes.
- 5. Replace estimated resource costs with measured embedded targets.
- **6. Port LQR/MPC/DSO to C** — full 4-way benchmark in native code (50× faster).
- **7. Multi-seed GP analysis** — evolution convergence across seeds.
- **8. Auto-GNATprove pipeline** — verify every candidate, not just the best.
