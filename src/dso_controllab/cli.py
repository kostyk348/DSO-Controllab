from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSO-ControlLab benchmark")
    parser.add_argument("--worlds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report = run_suite(args.worlds, args.seed, args.steps, args.dt)
    print_report(report, args.worlds, args.seed)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_report(report: dict, worlds: int, seed: int) -> None:
    print("DSO-ControlLab")
    print(f"worlds={worlds} seed={seed}")
    print(f"contract={report['contract']}")
    print()
    print(f"{'controller':<8} {'IAE mean':>10} {'score':>10} {'WCET us':>10} {'jitter':>10} {'contract':>10}")
    for name, data in sorted(report["summary"].items()):
        print(
            f"{name:<8} "
            f"{data['iae']['mean']:>10.4f} "
            f"{data['score']['mean']:>10.4f} "
            f"{data['wcet_us']['mean']:>10.3f} "
            f"{data['jitter_us']['mean']:>10.3f} "
            f"{data['contract_pass_rate']:>10.2%}"
        )
    print()
    print("paired sign tests: DSO score < other score")
    for other, test in sorted(report["sign_tests_vs_dso"].items()):
        print(f"DSO vs {other:<4}: wins={test['wins']:.0f} losses={test['losses']:.0f} z={test['z']:.2f}")


if __name__ == "__main__":
    main()
