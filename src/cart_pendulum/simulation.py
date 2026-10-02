"""Fixed-step simulation runner and CSV-friendly results."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Callable, Iterable

from .config import ProjectConfig
from .interfaces import ActuatorCommand, State
from .model import CartPendulumModel
from .safety import SafetySupervisor


CommandPolicy = Callable[[float, State], ActuatorCommand]


@dataclass(frozen=True)
class SimulationSample:
    timestamp_s: float
    state: State
    command: ActuatorCommand
    safety_ok: bool
    safety_reason: str

    def as_csv_row(self) -> tuple[object, ...]:
        return (
            self.timestamp_s,
            self.state.cart_position_m,
            self.state.cart_velocity_m_s,
            self.state.pendulum_angle_rad,
            self.state.pendulum_angular_velocity_rad_s,
            self.command.normalized,
            int(self.command.enabled),
            int(self.safety_ok),
            self.safety_reason,
        )


@dataclass(frozen=True)
class SimulationResult:
    samples: tuple[SimulationSample, ...]
    completed: bool
    termination_reason: str

    @property
    def final_state(self) -> State:
        if not self.samples:
            raise ValueError("simulation result contains no samples")
        return self.samples[-1].state

    def write_csv(self, path: str | Path) -> None:
        """Write a portable log for later plotting/replay."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "timestamp_s",
                    "cart_position_m",
                    "cart_velocity_m_s",
                    "pendulum_angle_rad",
                    "pendulum_angular_velocity_rad_s",
                    "command_normalized",
                    "command_enabled",
                    "safety_ok",
                    "safety_reason",
                )
            )
            writer.writerows(sample.as_csv_row() for sample in self.samples)


class SimulationRunner:
    """Run a deterministic fixed-step simulation.

    The policy is called once per control step with the current time and state.
    A missing policy means zero force. Safety is checked before every model
    step; a violation is latched and the runner returns a disabled command.
    """

    def __init__(
        self,
        model: CartPendulumModel,
        config: ProjectConfig,
        safety: SafetySupervisor | None = None,
    ) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.safety = safety or SafetySupervisor(config.safety)

    def run(self, policy: CommandPolicy | None = None) -> SimulationResult:
        self.safety.reset()
        command_policy = policy or (lambda _time_s, _state: ActuatorCommand())
        state = self.config.simulation.initial_state
        dt_s = self.config.simulation.dt_s
        samples: list[SimulationSample] = []

        for index in range(self.config.simulation.step_count + 1):
            timestamp_s = index * dt_s
            requested = command_policy(timestamp_s, state)
            safety_result = self.safety.check(state, requested, dt_s)
            sample = SimulationSample(
                timestamp_s=timestamp_s,
                state=state,
                command=safety_result.command,
                safety_ok=safety_result.allowed,
                safety_reason=safety_result.reason,
            )
            samples.append(sample)
            if not safety_result.allowed:
                return SimulationResult(tuple(samples), False, safety_result.reason)
            if index == self.config.simulation.step_count:
                break
            state = self.model.step(state, safety_result.command, dt_s)

        return SimulationResult(tuple(samples), True, "completed")


def zero_force_policy(_time_s: float, _state: State) -> ActuatorCommand:
    """Explicit zero-force policy useful for smoke tests and examples."""

    return ActuatorCommand()


def constant_policy(normalized: float) -> CommandPolicy:
    """Return a constant command policy, clamped to the valid range."""

    command = ActuatorCommand.clamped(normalized)

    def policy(_time_s: float, _state: State) -> ActuatorCommand:
        return command

    return policy


def samples_to_rows(samples: Iterable[SimulationSample]) -> Iterable[tuple[object, ...]]:
    """Convert samples to rows without requiring a plotting dependency."""

    return (sample.as_csv_row() for sample in samples)
