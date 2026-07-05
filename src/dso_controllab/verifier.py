from __future__ import annotations

import math

import numpy as np

from .contracts import DeploymentContract, ResourceContract, VerificationResult
from .metrics import resource_contract_pass, resource_metrics
from .world import World


def verify_plan(
    world: World,
    controller,
    contract: DeploymentContract,
    *,
    steps: int,
    dt: float,
    seed: int,
    target: float = 1.0,
) -> VerificationResult:
    rng = np.random.default_rng(seed)
    controller.reset()
    x = np.zeros(2, dtype=np.float64)
    y = 0.0
    iae = 0.0
    energy = 0.0
    overshoot = 0.0
    max_abs_y = 0.0
    saturated = 0
    finite = True

    for _ in range(steps):
        u = controller.compute(target, y, dt)
        if abs(u) >= 3.999:
            saturated += 1
        x, y = world.step(x, u, dt, rng)
        finite = finite and math.isfinite(y) and bool(np.all(np.isfinite(x)))
        error = target - y
        iae += abs(error) * dt
        energy += u * u
        overshoot = max(overshoot, y - target)
        max_abs_y = max(max_abs_y, abs(y))

    resources = resource_metrics(controller.cycles, controller.ram_bytes, controller.branch_points)
    metrics = {
        "verify_iae": float(iae),
        "verify_overshoot": float(max(0.0, overshoot)),
        "verify_energy": float(energy / steps),
        "verify_max_abs_y": float(max_abs_y),
        "verify_final_error": float(abs(target - y)),
        "verify_saturation_fraction": float(saturated / steps),
        **{f"verify_{k}": v for k, v in resources.items()},
    }

    violations: list[str] = []
    if not resource_contract_pass(resources, contract.resource):
        violations.append("resource")
    if metrics["verify_iae"] > contract.control.iae_max:
        violations.append("control.iae")
    if metrics["verify_overshoot"] > contract.control.overshoot_max:
        violations.append("control.overshoot")
    if metrics["verify_max_abs_y"] > contract.control.max_abs_y:
        violations.append("control.max_abs_y")
    if metrics["verify_final_error"] > contract.control.final_error_max:
        violations.append("control.final_error")
    if metrics["verify_saturation_fraction"] > contract.runtime.saturation_fraction_max:
        violations.append("runtime.saturation")
    if contract.runtime.finite_required and not finite:
        violations.append("runtime.nonfinite")

    return VerificationResult(passed=not violations, violations=tuple(violations), metrics=metrics)


def resource_only_contract() -> ResourceContract:
    return ResourceContract()
