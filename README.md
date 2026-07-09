# DSO-ControlLab

[![CI](https://github.com/kostyk348/DSO-Controllab/actions/workflows/ci.yml/badge.svg)](https://github.com/kostyk348/DSO-Controllab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

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

## Project Shape

```text
src/dso_controllab/
  cli.py          command-line runner
  contracts.py    resource/control/runtime/deployment contracts
  controllers.py  PID/LQR/MPC/DSO controllers
  experiment.py   random worlds, simulation, summary stats
  metrics.py      scoring and statistical helpers
  verifier.py     pre-deployment verification pass
  world.py        plant model
tests/
  test_smoke.py
```

## Next Research Steps

1. Add genetic programming controller synthesis.
2. Add 1000-world CSV export and plots.
3. Add paired statistical tests against PID/LQR/MPC.
4. Add verification traces: bounds, saturation, failure modes.
5. Replace estimated resource costs with measured embedded targets.
