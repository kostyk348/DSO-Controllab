from __future__ import annotations

import numpy as np

from .contracts import ResourceContract


def resource_metrics(cycles: int, ram_bytes: int, branch_points: int) -> dict[str, float]:
    wcet_us = cycles / 48.0
    jitter_us = 0.04 + 0.055 * branch_points
    determinism = 100.0 / (1.0 + 0.012 * cycles + 0.9 * jitter_us)
    return {
        "cycles": float(cycles),
        "ram_bytes": float(ram_bytes),
        "wcet_us": wcet_us,
        "jitter_us": jitter_us,
        "determinism": determinism,
    }


def resource_contract_pass(metrics: dict[str, float], contract: ResourceContract) -> bool:
    return (
        metrics["cycles"] <= contract.cycles_max
        and metrics["ram_bytes"] <= contract.ram_bytes_max
        and metrics["wcet_us"] <= contract.wcet_us_max
        and metrics["jitter_us"] <= contract.jitter_us_max
    )


def score(result: dict[str, float]) -> float:
    return (
        result["iae"]
        + 0.35 * result["overshoot"]
        + 0.04 * result["energy"]
        + 0.12 * result["wcet_us"]
        + 0.9 * result["jitter_us"]
    )


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "std": float(np.std(arr)),
    }


def sign_test(a: list[float], b: list[float]) -> dict[str, float]:
    """Normal-approx paired sign test over two score samples."""

    av = np.asarray(a)
    bv = np.asarray(b)
    wins = int(np.sum(av < bv))
    losses = int(np.sum(av > bv))
    n = wins + losses
    if n == 0:
        return {"wins": 0.0, "losses": 0.0, "z": 0.0}
    z = (wins - n / 2.0) / ((n * 0.25) ** 0.5)
    return {"wins": float(wins), "losses": float(losses), "z": float(z)}
