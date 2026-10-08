"""Safety supervisor used before an actuator command is applied."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import SafetyLimits
from .interfaces import ActuatorCommand, State


@dataclass(frozen=True)
class SafetyResult:
    allowed: bool
    command: ActuatorCommand
    reason: str = "ok"


class SafetySupervisor:
    """Validate state and command, failing safe on the first violation.

    This is a software layer, not a replacement for an emergency stop or
    hardware current/position limits. A fault is latched until ``reset`` is
    called explicitly.
    """

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()
        self.limits.validate()
        self._latched_reason: str | None = None
        self._previous_command = ActuatorCommand.disabled()

    @property
    def latched_reason(self) -> str | None:
        return self._latched_reason

    def reset(self) -> None:
        self._latched_reason = None
        self._previous_command = ActuatorCommand.disabled()

    def check(
        self,
        state: State,
        requested: ActuatorCommand,
        dt_s: float,
        *,
        check_pendulum_angle: bool = True,
    ) -> SafetyResult:
        """Check a requested command and return either it or a disabled command."""

        if self._latched_reason is not None:
            return SafetyResult(False, ActuatorCommand.disabled(), self._latched_reason)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            return self._fault("invalid_safety_timestep")
        try:
            state.validate()
        except ValueError:
            return self._fault("non_finite_state")

        violations: list[tuple[bool, str]] = [
            (
                abs(state.cart_position_m) > self.limits.max_cart_position_m,
                "cart_position_limit",
            ),
            (
                abs(state.cart_velocity_m_s) > self.limits.max_cart_velocity_m_s,
                "cart_velocity_limit",
            ),
        ]
        if check_pendulum_angle and self.limits.max_pendulum_angle_rad is not None:
            violations.append(
                (
                    abs(state.pendulum_angle_rad) > self.limits.max_pendulum_angle_rad,
                    "pendulum_angle_limit",
                )
            )
        violations.append(
            (
                abs(state.pendulum_angular_velocity_rad_s)
                > self.limits.max_pendulum_angular_velocity_rad_s,
                "pendulum_angular_velocity_limit",
            )
        )

        for violated, reason in violations:
            if violated:
                return self._fault(reason)

        delta = abs(requested.normalized - self._previous_command.normalized)
        if delta > self.limits.max_command_slew_per_s * dt_s + 1e-12:
            return self._fault("command_slew_limit")

        if requested.enabled:
            accepted = requested
        else:
            accepted = ActuatorCommand.disabled()
        self._previous_command = accepted
        return SafetyResult(True, accepted)

    def _fault(self, reason: str) -> SafetyResult:
        self._latched_reason = reason
        return SafetyResult(False, ActuatorCommand.disabled(), reason)
