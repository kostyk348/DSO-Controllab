from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResourceContract:
    cycles_max: int = 180
    ram_bytes_max: int = 96
    wcet_us_max: float = 8.0
    jitter_us_max: float = 0.8


@dataclass(frozen=True)
class ControlContract:
    iae_max: float = 4.0
    overshoot_max: float = 0.9
    max_abs_y: float = 8.0
    final_error_max: float = 1.25


@dataclass(frozen=True)
class RuntimeContract:
    saturation_fraction_max: float = 0.45
    finite_required: bool = True


@dataclass(frozen=True)
class DeploymentContract:
    resource: ResourceContract = field(default_factory=ResourceContract)
    control: ControlContract = field(default_factory=ControlContract)
    runtime: RuntimeContract = field(default_factory=RuntimeContract)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    violations: tuple[str, ...]
    metrics: dict[str, float]

    def compact(self) -> dict[str, float | str]:
        return {
            "verification_passed": float(self.passed),
            "violations": ",".join(self.violations),
            **self.metrics,
        }
