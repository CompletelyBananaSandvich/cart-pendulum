"""Data contracts shared by the simulator, estimator and controller.

All quantities in this module use SI units unless explicitly stated otherwise.
The pendulum angle is unwrapped internally and is measured from the upright
position: theta = 0 means the pendulum points up, positive theta means that
the centre of mass is to the positive-x side of the pivot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


TAU = 2.0 * math.pi


def wrap_angle(angle_rad: float) -> float:
    """Return an angle in [-pi, pi).

    The simulation keeps the angle unwrapped so that angular velocity remains
    continuous. Sensor-facing code can use this function when it needs the
    conventional principal angle.
    """

    if not math.isfinite(angle_rad):
        raise ValueError("angle must be finite")
    return (angle_rad + math.pi) % TAU - math.pi


@dataclass(frozen=True)
class State:
    """Estimated or simulated state of the cart-pendulum system."""

    cart_position_m: float = 0.0
    cart_velocity_m_s: float = 0.0
    pendulum_angle_rad: float = 0.0
    pendulum_angular_velocity_rad_s: float = 0.0

    def validate(self) -> None:
        values = (
            self.cart_position_m,
            self.cart_velocity_m_s,
            self.pendulum_angle_rad,
            self.pendulum_angular_velocity_rad_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all state components must be finite")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.cart_position_m,
            self.cart_velocity_m_s,
            self.pendulum_angle_rad,
            self.pendulum_angular_velocity_rad_s,
        )

    @classmethod
    def from_tuple(cls, values: tuple[float, float, float, float]) -> "State":
        if len(values) != 4:
            raise ValueError("state must contain four values")
        state = cls(*values)
        state.validate()
        return state


@dataclass(frozen=True)
class Measurement:
    """Raw sensor measurement before state estimation.

    Velocities are optional because many real systems measure only position
    and angle. The estimator can derive them from timestamped samples.
    """

    timestamp_s: float
    cart_position_m: float
    pendulum_angle_rad: float
    cart_velocity_m_s: float | None = None
    pendulum_angular_velocity_rad_s: float | None = None
    valid: bool = True

    def validate(self) -> None:
        required = (
            self.timestamp_s,
            self.cart_position_m,
            self.pendulum_angle_rad,
        )
        if not all(math.isfinite(value) for value in required):
            raise ValueError("required measurement values must be finite")
        optional = (
            self.cart_velocity_m_s,
            self.pendulum_angular_velocity_rad_s,
        )
        if not all(value is None or math.isfinite(value) for value in optional):
            raise ValueError("optional measurement values must be finite or None")
        if self.timestamp_s < 0.0:
            raise ValueError("measurement timestamp cannot be negative")


@dataclass(frozen=True)
class ActuatorCommand:
    """Normalised actuator command.

    ``normalized`` is deliberately hardware-independent. The model maps it to
    force using ``PendulumParameters.max_force_n``; a hardware adapter can map
    the same value to PWM, current or another driver-specific quantity.
    """

    normalized: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.normalized):
            raise ValueError("actuator command must be finite")
        if not -1.0 <= self.normalized <= 1.0:
            raise ValueError("normalised actuator command must be in [-1, 1]")

    @classmethod
    def clamped(cls, value: float, enabled: bool = True) -> "ActuatorCommand":
        """Build a command after clamping a controller output to [-1, 1]."""

        if not math.isfinite(value):
            raise ValueError("actuator command must be finite")
        return cls(max(-1.0, min(1.0, value)), enabled=enabled)

    @classmethod
    def disabled(cls) -> "ActuatorCommand":
        return cls(0.0, enabled=False)
