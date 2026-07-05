from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .contracts import VerificationResult
from .world import World


class Controller(Protocol):
    name: str
    cycles: int
    ram_bytes: int
    branch_points: int

    def reset(self) -> None:
        ...

    def compute(self, target: float, y: float, dt: float) -> float:
        ...


def clamp(value: float, lo: float = -4.0, hi: float = 4.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    name: str = "PID"
    cycles: int = 42
    ram_bytes: int = 40
    branch_points: int = 1

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target: float, y: float, dt: float) -> float:
        error = target - y
        self.integral = clamp(self.integral + error * dt, -8.0, 8.0)
        deriv = (error - self.prev_error) / dt
        self.prev_error = error
        return clamp(self.kp * error + self.ki * self.integral + self.kd * deriv)


@dataclass
class LQR:
    k: np.ndarray
    name: str = "LQR"
    cycles: int = 58
    ram_bytes: int = 64
    branch_points: int = 1

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.xhat = np.zeros(2, dtype=np.float64)
        self.last_y = 0.0

    def compute(self, target: float, y: float, dt: float) -> float:
        velocity = (y - self.last_y) / dt
        self.last_y = y
        self.xhat[:] = (y - target, velocity)
        return clamp(float(-(self.k @ self.xhat)))


@dataclass
class MPC:
    world: World
    horizon: int = 8
    name: str = "MPC"
    cycles: int = 620
    ram_bytes: int = 176
    branch_points: int = 10

    def reset(self) -> None:
        self.last_y = 0.0

    def compute(self, target: float, y: float, dt: float) -> float:
        velocity = (y - self.last_y) / dt
        self.last_y = y
        state = np.array([y, velocity], dtype=np.float64)
        best_u = 0.0
        best_cost = float("inf")
        for u in np.linspace(-3.0, 3.0, 13):
            x = state.copy()
            cost = 0.0
            for _ in range(self.horizon):
                x, yp = self.world.step(x, float(u), dt)
                cost += (target - yp) ** 2 + 0.015 * u * u
            if cost < best_cost:
                best_cost = cost
                best_u = float(u)
        return clamp(best_u)


@dataclass
class DSOPlan:
    controller: PID
    source: str
    contract_passed: bool
    verification: VerificationResult
    name: str = "DSO"
    cycles: int = 32
    ram_bytes: int = 36
    branch_points: int = 0

    def reset(self) -> None:
        self.controller.reset()

    def compute(self, target: float, y: float, dt: float) -> float:
        return self.controller.compute(target, y, dt)


def lqr_for(world: World, dt: float) -> LQR:
    a, b = world.matrices(dt)
    q = np.diag([12.0, 1.0])
    r = np.array([[0.08]])
    p = q.copy()
    bm = b.reshape(2, 1)
    for _ in range(160):
        s = r + bm.T @ p @ bm
        gain = np.linalg.solve(s, bm.T @ p @ a)
        p_next = a.T @ p @ a - a.T @ p @ bm @ gain + q
        if np.max(np.abs(p_next - p)) < 1e-9:
            p = p_next
            break
        p = p_next
    k = np.linalg.solve(r + bm.T @ p @ bm, bm.T @ p @ a).reshape(2)
    return LQR(k=k)


def candidate_pids() -> list[PID]:
    candidates: list[PID] = []
    for kp in (0.8, 1.3, 2.0, 2.9):
        for ki in (0.0, 0.12, 0.28):
            for kd in (0.0, 0.08, 0.20):
                candidates.append(PID(kp, ki, kd))
    return candidates
