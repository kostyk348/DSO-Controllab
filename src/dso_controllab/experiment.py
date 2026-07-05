from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .contracts import DeploymentContract
from .controllers import DSOPlan, MPC, PID, candidate_pids, lqr_for
from .metrics import resource_contract_pass, resource_metrics, score, sign_test, summarize
from .verifier import verify_plan
from .world import World


def simulate(
    world: World,
    controller,
    *,
    steps: int,
    dt: float,
    seed: int,
    target: float = 1.0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    controller.reset()
    x = np.zeros(2, dtype=np.float64)
    y = 0.0
    iae = 0.0
    energy = 0.0
    overshoot = 0.0
    for _ in range(steps):
        u = controller.compute(target, y, dt)
        x, y = world.step(x, u, dt, rng)
        error = target - y
        iae += abs(error) * dt
        energy += u * u
        overshoot = max(overshoot, y - target)
    resources = resource_metrics(controller.cycles, controller.ram_bytes, controller.branch_points)
    result = {
        "iae": float(iae),
        "overshoot": float(max(0.0, overshoot)),
        "energy": float(energy / steps),
        **resources,
    }
    result["score"] = score(result)
    return result


def tune_pid(world: World, steps: int, dt: float, seed: int) -> PID:
    best = None
    best_score = float("inf")
    for idx, pid in enumerate(candidate_pids()):
        res = simulate(world, pid, steps=steps, dt=dt, seed=seed + idx)
        if res["score"] < best_score:
            best_score = res["score"]
            best = pid
    assert best is not None
    return PID(best.kp, best.ki, best.kd)


def build_dso_plan(world: World, steps: int, dt: float, seed: int, contract: DeploymentContract) -> DSOPlan:
    """Offline planning stage: pick the best verified PID from a fixed bank."""

    best = None
    best_score = float("inf")
    best_verification = None
    for idx, pid in enumerate(candidate_pids()):
        resources = resource_metrics(32, 36, 0)
        if not resource_contract_pass(resources, contract.resource):
            continue
        pid.cycles = 32
        pid.ram_bytes = 36
        pid.branch_points = 0
        verification = verify_plan(world, pid, contract, steps=steps, dt=dt, seed=seed + idx)
        if not verification.passed:
            continue
        res = verification.metrics
        planned_score = (
            res["verify_iae"]
            + 0.35 * res["verify_overshoot"]
            + 0.04 * res["verify_energy"]
            + 0.02 * res["verify_saturation_fraction"]
        )
        if planned_score < best_score:
            best_score = planned_score
            best = pid
            best_verification = verification
    if best is None:
        best = PID(1.1, 0.08, 0.05)
        best.cycles = 32
        best.ram_bytes = 36
        best.branch_points = 0
        best_verification = verify_plan(world, best, contract, steps=steps, dt=dt, seed=seed + 999)
    resources = resource_metrics(32, 36, 0)
    return DSOPlan(
        controller=PID(best.kp, best.ki, best.kd),
        source=f"offline_pid_bank(kp={best.kp:.2f},ki={best.ki:.2f},kd={best.kd:.2f})",
        contract_passed=resource_contract_pass(resources, contract.resource) and best_verification.passed,
        verification=best_verification,
    )


def run_suite(worlds: int, seed: int, steps: int = 500, dt: float = 0.02) -> dict:
    rng = np.random.default_rng(seed)
    contract = DeploymentContract()
    rows: list[dict] = []

    for idx in range(worlds):
        world = World.random(rng)
        base_seed = seed * 100_000 + idx * 100
        controllers = [
            tune_pid(world, steps, dt, base_seed),
            lqr_for(world, dt),
            MPC(world),
            build_dso_plan(world, steps, dt, base_seed + 50, contract),
        ]
        for ctrl in controllers:
            result = simulate(world, ctrl, steps=steps, dt=dt, seed=base_seed + 99)
            result["controller"] = ctrl.name
            result["world"] = idx
            result["resource_contract_passed"] = float(resource_contract_pass(result, contract.resource))
            result["contract_passed"] = result["resource_contract_passed"]
            if isinstance(ctrl, DSOPlan):
                result["plan"] = ctrl.source
                result.update(ctrl.verification.compact())
                result["contract_passed"] = float(ctrl.verification.passed)
            rows.append(result)

    return summarize_rows(rows, contract)


def summarize_rows(rows: list[dict], contract: DeploymentContract) -> dict:
    controllers = sorted({row["controller"] for row in rows})
    by_controller = {name: [r for r in rows if r["controller"] == name] for name in controllers}
    summary = {}
    for name, items in by_controller.items():
        summary[name] = {
            "iae": summarize([r["iae"] for r in items]),
            "score": summarize([r["score"] for r in items]),
            "wcet_us": summarize([r["wcet_us"] for r in items]),
            "jitter_us": summarize([r["jitter_us"] for r in items]),
            "contract_pass_rate": float(np.mean([r["contract_passed"] for r in items])),
            "resource_pass_rate": float(np.mean([r["resource_contract_passed"] for r in items])),
        }

    score_by_name = {
        name: [r["score"] for r in sorted(items, key=lambda row: row["world"])]
        for name, items in by_controller.items()
    }
    tests = {
        other: sign_test(score_by_name["DSO"], score_by_name[other])
        for other in controllers
        if other != "DSO"
    }
    return {"contract": asdict(contract), "summary": summary, "sign_tests_vs_dso": tests, "rows": rows}
