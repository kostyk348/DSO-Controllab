from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class World:
    """Stable second-order control world.

    x0' = x1
    x1' = -wn^2 x0 - 2 zeta wn x1 + gain u + disturbance
    y = x0
    """

    wn: float
    zeta: float
    gain: float
    delay_steps: int
    disturbance: float
    noise: float

    @staticmethod
    def random(rng: np.random.Generator) -> "World":
        return World(
            wn=float(rng.uniform(0.6, 2.8)),
            zeta=float(rng.uniform(0.12, 1.4)),
            gain=float(rng.uniform(0.55, 1.8)),
            delay_steps=int(rng.integers(0, 8)),
            disturbance=float(rng.normal(0.0, 0.04)),
            noise=float(rng.uniform(0.0, 0.015)),
        )

    def matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        a = np.array(
            [
                [1.0, dt],
                [-(self.wn**2) * dt, 1.0 - 2.0 * self.zeta * self.wn * dt],
            ],
            dtype=np.float64,
        )
        b = np.array([0.0, self.gain * dt], dtype=np.float64)
        return a, b

    def step(
        self,
        x: np.ndarray,
        u: float,
        dt: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, float]:
        a, b = self.matrices(dt)
        xn = a @ x + b * u + np.array([0.0, self.disturbance * dt])
        y = float(xn[0])
        if rng is not None and self.noise > 0.0:
            y += float(rng.normal(0.0, self.noise))
        return xn, y
