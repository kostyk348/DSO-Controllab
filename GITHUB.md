# GitHub Publishing Checklist

Recommended repository name:

```text
dso-controllab
```

Possible description:

```text
Research prototype for Deterministic Systems Optimization: verified execution plans and resource contracts for control systems.
```

Suggested topics:

```text
deterministic-systems
control-systems
real-time
scheduling
resource-contracts
verification
python
research
embedded
runtime
```

## Before First Push

Run:

```bash
python3 -m pytest -q
PYTHONPATH=src python3 -m dso_controllab --worlds 100 --seed 4 --steps 120 --json results_contracts_100.json
```

Commit useful files:

```text
README.md
CONTRACTS.md
VISION.md
ROADMAP.md
ARTICLE_LINKEDIN.md
GITHUB.md
pyproject.toml
src/
tests/
results_contracts_100.json
```

Avoid committing:

```text
.pytest_cache/
__pycache__/
large generated result files
```

## First Commit Message

```text
Initial DSO-ControlLab prototype
```

## Suggested README Pitch

DSO-ControlLab is a research prototype for Deterministic Systems Optimization.

It tests a simple idea: move runtime decisions into pre-verified execution plans with explicit resource, control and runtime contracts.

The first benchmark applies DSO to control systems and compares PID, LQR, MPC and a DSO-style pre-planned controller across random worlds.
