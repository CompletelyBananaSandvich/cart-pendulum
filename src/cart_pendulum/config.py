"""Validated configuration objects and TOML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any
import tomllib

from .interfaces import State


@dataclass(frozen=True)
class PendulumParameters:
    """Physical parameters for the nonlinear model.

    ``pendulum_inertia_kg_m2`` is the moment of inertia about the pendulum
    centre of mass. The model adds ``m * l**2`` to obtain inertia about the
    pivot.
    """

    cart_mass_kg: float = 0.50
    pendulum_mass_kg: float = 0.20
    center_of_mass_length_m: float = 0.30
    pendulum_inertia_kg_m2: float = 0.018
    gravitational_acceleration_m_s2: float = 9.80665
    cart_viscous_friction_n_s_per_m: float = 0.05
    pendulum_viscous_friction_n_m_s_per_rad: float = 0.005
    max_force_n: float = 10.0

    def validate(self) -> None:
        positive = {
            "cart_mass_kg": self.cart_mass_kg,
            "pendulum_mass_kg": self.pendulum_mass_kg,
            "center_of_mass_length_m": self.center_of_mass_length_m,
            "pendulum_inertia_kg_m2": self.pendulum_inertia_kg_m2,
            "gravitational_acceleration_m_s2": self.gravitational_acceleration_m_s2,
            "max_force_n": self.max_force_n,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        non_negative = {
            "cart_viscous_friction_n_s_per_m": self.cart_viscous_friction_n_s_per_m,
            "pendulum_viscous_friction_n_m_s_per_rad": self.pendulum_viscous_friction_n_m_s_per_rad,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def total_mass_kg(self) -> float:
        return self.cart_mass_kg + self.pendulum_mass_kg

    @property
    def pendulum_inertia_about_pivot_kg_m2(self) -> float:
        return self.pendulum_inertia_kg_m2 + self.pendulum_mass_kg * self.center_of_mass_length_m**2


@dataclass(frozen=True)
class SimulationConfig:
    dt_s: float = 0.002
    duration_s: float = 4.0
    initial_state: State = field(default_factory=State)

    def validate(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("simulation dt_s must be finite and greater than zero")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("simulation duration_s must be finite and greater than zero")
        steps = self.duration_s / self.dt_s
        if not math.isclose(steps, round(steps), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("duration_s must be an integer multiple of dt_s")
        self.initial_state.validate()

    @property
    def step_count(self) -> int:
        self.validate()
        return int(round(self.duration_s / self.dt_s))


@dataclass(frozen=True)
class SafetyLimits:
    """Software safety limits used by simulation and, later, hardware."""

    max_cart_position_m: float = 1.8
    max_cart_velocity_m_s: float = 3.0
    max_pendulum_angle_rad: float | None = math.radians(85.0)
    max_pendulum_angular_velocity_rad_s: float = 15.0
    max_command_slew_per_s: float = 5.0

    def validate(self) -> None:
        values = {
            "max_cart_position_m": self.max_cart_position_m,
            "max_cart_velocity_m_s": self.max_cart_velocity_m_s,
            "max_pendulum_angular_velocity_rad_s": self.max_pendulum_angular_velocity_rad_s,
            "max_command_slew_per_s": self.max_command_slew_per_s,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if self.max_pendulum_angle_rad is not None:
            if not math.isfinite(self.max_pendulum_angle_rad) or self.max_pendulum_angle_rad <= 0.0:
                raise ValueError("max_pendulum_angle_rad must be finite and greater than zero")
            if self.max_pendulum_angle_rad >= math.pi:
                raise ValueError("max_pendulum_angle_rad must be less than pi")


@dataclass(frozen=True)
class ProjectConfig:
    parameters: PendulumParameters = field(default_factory=PendulumParameters)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    def validate(self) -> None:
        self.parameters.validate()
        self.simulation.validate()
        self.safety.validate()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"TOML section [{name}] must be a table")
    return value


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project configuration from a TOML file."""

    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)

    pendulum_data = _section(data, "pendulum")
    simulation_data = _section(data, "simulation")
    initial_state_data = _section(data, "initial_state")
    safety_data = _section(data, "safety")

    parameters = PendulumParameters(
        cart_mass_kg=float(pendulum_data.get("cart_mass_kg", 0.50)),
        pendulum_mass_kg=float(pendulum_data.get("pendulum_mass_kg", 0.20)),
        center_of_mass_length_m=float(pendulum_data.get("center_of_mass_length_m", 0.30)),
        pendulum_inertia_kg_m2=float(pendulum_data.get("pendulum_inertia_kg_m2", 0.018)),
        gravitational_acceleration_m_s2=float(
            pendulum_data.get("gravitational_acceleration_m_s2", 9.80665)
        ),
        cart_viscous_friction_n_s_per_m=float(
            pendulum_data.get("cart_viscous_friction_n_s_per_m", 0.05)
        ),
        pendulum_viscous_friction_n_m_s_per_rad=float(
            pendulum_data.get("pendulum_viscous_friction_n_m_s_per_rad", 0.005)
        ),
        max_force_n=float(pendulum_data.get("max_force_n", 10.0)),
    )
    initial_state = State(
        cart_position_m=float(initial_state_data.get("cart_position_m", 0.0)),
        cart_velocity_m_s=float(initial_state_data.get("cart_velocity_m_s", 0.0)),
        pendulum_angle_rad=float(initial_state_data.get("pendulum_angle_rad", 0.0)),
        pendulum_angular_velocity_rad_s=float(
            initial_state_data.get("pendulum_angular_velocity_rad_s", 0.0)
        ),
    )
    simulation = SimulationConfig(
        dt_s=float(simulation_data.get("dt_s", 0.002)),
        duration_s=float(simulation_data.get("duration_s", 4.0)),
        initial_state=initial_state,
    )
    angle_limit_raw = safety_data.get("max_pendulum_angle_rad", math.radians(85.0))
    if angle_limit_raw is None or angle_limit_raw is False or str(angle_limit_raw).lower() == "none":
        max_pendulum_angle_rad = None
    else:
        max_pendulum_angle_rad = float(angle_limit_raw)
    safety = SafetyLimits(
        max_cart_position_m=float(safety_data.get("max_cart_position_m", 1.8)),
        max_cart_velocity_m_s=float(safety_data.get("max_cart_velocity_m_s", 3.0)),
        max_pendulum_angle_rad=max_pendulum_angle_rad,
        max_pendulum_angular_velocity_rad_s=float(
            safety_data.get("max_pendulum_angular_velocity_rad_s", 15.0)
        ),
        max_command_slew_per_s=float(safety_data.get("max_command_slew_per_s", 5.0)),
    )
    config = ProjectConfig(parameters, simulation, safety)
    config.validate()
    return config
