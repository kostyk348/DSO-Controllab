# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

No tagged releases yet — the entries below are reconstructed from `git log`
(first commit `c7d47f7`, 2026-07-05) and grouped by kind of change.
The maintainer can promote the `Unreleased` section to `v0.1.0` when the first
tag is cut.

## [Unreleased]

### Added

- **Python research prototype** — `src/dso_controllab`: PID / LQR / MPC / DSO controllers, random
  second-order worlds, resource / control / runtime contracts, pre-deployment verification pass and
  aggregate statistics (`c7d47f7`).
- **GP controller synthesis engine in C11** — `gp/`: expression trees (63 nodes, depth 4–6), tournament
  selection, subtree crossover, point mutation, elitism, OpenMP fitness evaluation (`130a2e9`).
- **Ada SPARK export + GNATprove verification** of evolved controllers (`130a2e9`, `5aa3f90`).
- **5-way benchmark** (GP vs PID vs LQR vs MPC vs DSO) in a single native C binary (`a8c9d40`).
- **Stability analysis** (`--stability N`): Monte-Carlo parameter perturbation, oscillation and gain-margin
  estimates; multi-seed sweep, CSV export and gnuplot output (`5aa3f90`).
- **Multi-tier adaptive controller runtime** with plant fingerprinting (`0644c13`).
- **sin/cos terminals and eigenvalue-based stability checks** in the GP engine (`705c564`).
- **RACS+DSO synthesis** — NSGA-II controller synthesis, variance-centric determinism, QEMU-calibrated
  cycle model (`d49da0f`).
- **Hardware-in-the-loop validation** on QEMU Cortex-M4 (`616625f`).
- **DSO streaming compiler** (multi-rate) plus on-target Bayesian optimization (`f1180ef`).
- Documentation: vision, roadmap, contracts (`VISION.md`, `ROADMAP.md`, `CONTRACTS.md`), GP engine section
  in the README with benchmark, formal verification and CLI reference (`f580da2`).
- CI workflow and status badges (`1452fca`, `b16ed8e`, `e2b614d`).
- `LICENSE` (`27e8b4c`, `e2b614d`).

### Changed

- License switched to **Apache 2.0** (patent clause, attribution, royalty-friendly) (`6bd69d6`);
  README badge follow-up corrected accordingly.
- GP fitness moved to ITAE with adaptive rates and settling / energy penalties (`b3d8a2f`).
- RACS+DSO status documented as complete and validated (`44e9284`).

### Fixed

- Removed the failing CI workflow and replaced it with a working one (`fd8b118`, `e2b614d`).
- CI: `pytest` was never installed, so every run died with `No module named pytest` — the workflow now
  installs `numpy pytest` and runs the README smoke command as an extra step.
- README quick start used a bare `python3 -m dso_controllab …`, which fails on a clean clone; the command
  now carries the `PYTHONPATH=src` (or `pip install`) requirement and verified runtimes.
