"""Coursework swing-up and LQR control algorithms for cart-pendulum.

Implements:
1. Linearized dynamics and continuous-time algebraic Riccati equation (CARE) solver
   for optimal state-feedback (LQR) upright stabilization.
2. Energy-based swing-up controller with cart centering.
3. Multi-mode controller coordinator with safe transitions, hysteresis,
   anti-windup, slew-rate limiting, and fault detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Sequence

from .config import PendulumParameters
from .interfaces import ActuatorCommand, State, wrap_angle


class ControllerMode(str, Enum):
    """Operational mode of the cart-pendulum controller."""

    DISARMED = "disarmed"
    SWING_UP = "swing_up"
    BALANCE = "balance"
    FAULT = "fault"


# Default LQR gains for nominal parameters in default.toml:
# M=0.5, m=0.2, l=0.3, J=0.036, Q=diag(10, 1, 100, 10), R=0.5
# Produces control force u = - (k_x * x + k_v * x_dot + k_th * th + k_w * th_dot)
DEFAULT_LQR_GAINS: tuple[float, float, float, float] = (
    -4.4721,
    -5.9144,
    -42.3764,
    -10.4569,
)


def linearize_dynamics(
    parameters: PendulumParameters,
) -> tuple[list[list[float]], list[list[float]]]:
    """Compute continuous-time linearized matrices A (4x4) and B (4x1) around upright.

    State vector: z = [x, x_dot, theta, theta_dot]^T
    Input: u = force (N)
    """

    parameters.validate()
    p = parameters
    M = p.cart_mass_kg
    m = p.pendulum_mass_kg
    l = p.center_of_mass_length_m
    J = p.pendulum_inertia_about_pivot_kg_m2
    g = p.gravitational_acceleration_m_s2
    b = p.cart_viscous_friction_n_s_per_m
    c = p.pendulum_viscous_friction_n_m_s_per_rad

    D = (M + m) * J - (m * l) ** 2
    if D <= 0.0 or not math.isfinite(D):
        raise ValueError("determinant of mass matrix must be positive")

    A = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -J * b / D, -((m * l) ** 2 * g) / D, (m * l * c) / D],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, (m * l * b) / D, ((M + m) * m * g * l) / D, -((M + m) * c) / D],
    ]
    B = [
        [0.0],
        [J / D],
        [0.0],
        [-m * l / D],
    ]
    return A, B


def solve_continuous_algebraic_riccati(
    A: list[list[float]],
    B: list[list[float]],
    Q: list[list[float]],
    r_scalar: float,
    tau_max: float = 3.0,
    dt: float = 0.002,
) -> list[list[float]]:
    """Solve A^T P + P A - P B R^-1 B^T P + Q = 0 via RK4 Riccati differential equation."""

    if r_scalar <= 0.0 or not math.isfinite(r_scalar):
        raise ValueError("r_scalar must be positive and finite")

    r_inv = 1.0 / r_scalar
    brb = [[B[i][0] * r_inv * B[j][0] for j in range(4)] for i in range(4)]
    at = [[A[j][i] for j in range(4)] for i in range(4)]

    def rhs(p_mat: list[list[float]]) -> list[list[float]]:
        at_p = [
            [sum(at[i][k] * p_mat[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)
        ]
        p_a = [
            [sum(p_mat[i][k] * A[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)
        ]
        p_brb = [
            [sum(p_mat[i][k] * brb[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)
        ]
        p_brb_p = [
            [sum(p_brb[i][k] * p_mat[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)
        ]
        return [
            [at_p[i][j] + p_a[i][j] - p_brb_p[i][j] + Q[i][j] for j in range(4)]
            for i in range(4)
        ]

    P = [[Q[i][j] for j in range(4)] for i in range(4)]
    steps = max(100, int(round(tau_max / dt)))
    for _ in range(steps):
        k1 = rhs(P)
        p_k1 = [[P[i][j] + 0.5 * dt * k1[i][j] for j in range(4)] for i in range(4)]
        k2 = rhs(p_k1)
        p_k2 = [[P[i][j] + 0.5 * dt * k2[i][j] for j in range(4)] for i in range(4)]
        k3 = rhs(p_k2)
        p_k3 = [[P[i][j] + dt * k3[i][j] for j in range(4)] for i in range(4)]
        k4 = rhs(p_k3)

        P = [
            [
                P[i][j]
                + (dt / 6.0)
                * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j])
                for j in range(4)
            ]
            for i in range(4)
        ]
        # Keep P symmetric
        for i in range(4):
            for j in range(i + 1, 4):
                sym = 0.5 * (P[i][j] + P[j][i])
                P[i][j] = sym
                P[j][i] = sym

    return P


def compute_lqr_gain(
    parameters: PendulumParameters,
    q_diag: Sequence[float] = (10.0, 1.0, 100.0, 10.0),
    r_scalar: float = 0.5,
) -> tuple[float, float, float, float]:
    """Compute LQR gain K = (k_x, k_x_dot, k_theta, k_theta_dot) for force = - K * z."""

    if len(q_diag) != 4 or any(v <= 0.0 or not math.isfinite(v) for v in q_diag):
        raise ValueError("q_diag must contain 4 positive finite weights")

    A, B = linearize_dynamics(parameters)
    Q = [[q_diag[i] if i == j else 0.0 for j in range(4)] for i in range(4)]
    P = solve_continuous_algebraic_riccati(A, B, Q, r_scalar)
    r_inv = 1.0 / r_scalar
    k_vec = tuple(r_inv * sum(B[m][0] * P[m][j] for m in range(4)) for j in range(4))
    return (float(k_vec[0]), float(k_vec[1]), float(k_vec[2]), float(k_vec[3]))


@dataclass(frozen=True)
class LQRConfig:
    """Configuration for LQR balancing controller."""

    gains: tuple[float, float, float, float] = DEFAULT_LQR_GAINS
    target_cart_position_m: float = 0.0
    integral_gain: float = 0.0
    integral_limit_n: float = 2.0
    max_force_n: float | None = None

    def validate(self) -> None:
        if len(self.gains) != 4 or any(not math.isfinite(g) for g in self.gains):
            raise ValueError("LQR gains must contain 4 finite numbers")
        if not math.isfinite(self.target_cart_position_m):
            raise ValueError("target_cart_position_m must be finite")
        if not math.isfinite(self.integral_gain) or self.integral_gain < 0.0:
            raise ValueError("integral_gain must be non-negative and finite")
        if not math.isfinite(self.integral_limit_n) or self.integral_limit_n < 0.0:
            raise ValueError("integral_limit_n must be non-negative and finite")
        if self.max_force_n is not None:
            if not math.isfinite(self.max_force_n) or self.max_force_n <= 0.0:
                raise ValueError("max_force_n must be positive and finite or None")


class LQRController:
    """Full-state feedback LQR controller for inverted pendulum upright balance."""

    def __init__(
        self,
        parameters: PendulumParameters,
        config: LQRConfig | None = None,
    ) -> None:
        self.parameters = parameters
        self.parameters.validate()
        self.config = config or LQRConfig()
        self.config.validate()
        self._integral_error = 0.0

    @property
    def max_force_n(self) -> float:
        return (
            self.config.max_force_n
            if self.config.max_force_n is not None
            else self.parameters.max_force_n
        )

    def reset(self) -> None:
        self._integral_error = 0.0

    def compute_force(self, state: State, dt_s: float = 0.0) -> float:
        """Compute desired balancing force in Newtons."""

        state.validate()
        pos_error = state.cart_position_m - self.config.target_cart_position_m
        if self.config.integral_gain > 0.0 and dt_s > 0.0:
            self._integral_error += pos_error * dt_s
            max_int = self.config.integral_limit_n / self.config.integral_gain
            self._integral_error = max(-max_int, min(max_int, self._integral_error))

        angle_error = wrap_angle(state.pendulum_angle_rad)
        k_x, k_v, k_th, k_w = self.config.gains
        force = -(
            k_x * pos_error
            + k_v * state.cart_velocity_m_s
            + k_th * angle_error
            + k_w * state.pendulum_angular_velocity_rad_s
            + self.config.integral_gain * self._integral_error
        )
        return max(-self.max_force_n, min(self.max_force_n, force))

    def compute_command(self, state: State, dt_s: float = 0.0) -> ActuatorCommand:
        """Return clamped ActuatorCommand."""

        force = self.compute_force(state, dt_s)
        return ActuatorCommand.clamped(force / self.parameters.max_force_n)


@dataclass(frozen=True)
class SwingUpConfig:
    """Configuration for energy-based swing-up controller."""

    k_energy: float = 25.0
    max_swing_force_n: float = 6.0
    cart_p_gain: float = 4.0
    cart_d_gain: float = 1.5
    target_cart_position_m: float = 0.0
    perturbation_force_n: float = 1.5

    def validate(self) -> None:
        values = {
            "k_energy": self.k_energy,
            "max_swing_force_n": self.max_swing_force_n,
            "cart_p_gain": self.cart_p_gain,
            "cart_d_gain": self.cart_d_gain,
            "perturbation_force_n": self.perturbation_force_n,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.target_cart_position_m):
            raise ValueError("target_cart_position_m must be finite")


class EnergySwingUpController:
    """Nonlinear energy-based swing-up controller with cart centering.

    Pumps energy into the pendulum when mechanical energy is below upright,
    decelerates when energy is excessive, and centers the cart on the track.
    """

    def __init__(
        self,
        parameters: PendulumParameters,
        config: SwingUpConfig | None = None,
    ) -> None:
        self.parameters = parameters
        self.parameters.validate()
        self.config = config or SwingUpConfig()
        self.config.validate()

    @property
    def upright_energy_joules(self) -> float:
        p = self.parameters
        return p.pendulum_mass_kg * p.gravitational_acceleration_m_s2 * p.center_of_mass_length_m

    def energy(self, state: State) -> float:
        """Return total mechanical energy of pendulum with pivot at origin."""

        p = self.parameters
        J = p.pendulum_inertia_about_pivot_kg_m2
        kinetic = 0.5 * J * state.pendulum_angular_velocity_rad_s**2
        potential = (
            p.pendulum_mass_kg
            * p.gravitational_acceleration_m_s2
            * p.center_of_mass_length_m
            * math.cos(state.pendulum_angle_rad)
        )
        return kinetic + potential

    def energy_error(self, state: State) -> float:
        return self.energy(state) - self.upright_energy_joules

    def compute_force(self, state: State) -> float:
        """Compute swing-up control force in Newtons."""

        state.validate()
        p = self.parameters
        cfg = self.config
        th = state.pendulum_angle_rad
        th_dot = state.pendulum_angular_velocity_rad_s
        cos_th = math.cos(th)

        e_err = self.energy_error(state)

        # Direction of energy pumping: sign(theta_dot * cos(theta))
        motion = th_dot * cos_th
        if abs(motion) > 1e-4:
            s = math.copysign(1.0, motion)
        else:
            # Escape static equilibrium (e.g. hanging down at bottom)
            s = 1.0 if cos_th < 0.0 else -1.0

        raw_force = cfg.k_energy * e_err * s
        # Apply swing-force saturation limit
        clamped_swing = max(-cfg.max_swing_force_n, min(cfg.max_swing_force_n, raw_force))

        # Cart centering PD term
        pos_error = state.cart_position_m - cfg.target_cart_position_m
        cart_force = -cfg.cart_p_gain * pos_error - cfg.cart_d_gain * state.cart_velocity_m_s

        force = clamped_swing + cart_force
        return max(-p.max_force_n, min(p.max_force_n, force))

    def compute_command(self, state: State) -> ActuatorCommand:
        force = self.compute_force(state)
        return ActuatorCommand.clamped(force / self.parameters.max_force_n)


@dataclass(frozen=True)
class ControllerConfig:
    """Master controller configuration for swing-up and balancing coordinator."""

    lqr: LQRConfig = field(default_factory=LQRConfig)
    swing_up: SwingUpConfig = field(default_factory=SwingUpConfig)
    catch_angle_rad: float = 0.35  # ~20 degrees
    exit_angle_rad: float = 0.55  # ~31 degrees
    catch_angular_velocity_rad_s: float = 3.0
    min_dwell_time_s: float = 0.10
    max_command_slew_per_s: float = 4.80
    initial_mode: ControllerMode = ControllerMode.SWING_UP

    def validate(self) -> None:
        self.lqr.validate()
        self.swing_up.validate()
        if not 0.0 < self.catch_angle_rad < self.exit_angle_rad < math.pi:
            raise ValueError(
                "catch_angle_rad must be positive and strictly less than exit_angle_rad (< pi)"
            )
        if (
            not math.isfinite(self.catch_angular_velocity_rad_s)
            or self.catch_angular_velocity_rad_s <= 0.0
        ):
            raise ValueError("catch_angular_velocity_rad_s must be positive and finite")
        if not math.isfinite(self.min_dwell_time_s) or self.min_dwell_time_s < 0.0:
            raise ValueError("min_dwell_time_s must be non-negative and finite")
        if (
            not math.isfinite(self.max_command_slew_per_s)
            or self.max_command_slew_per_s <= 0.0
        ):
            raise ValueError("max_command_slew_per_s must be positive and finite")


class CartPendulumController:
    """Combined coursework swing-up and LQR controller.

    Manages state machine modes (DISARMED, SWING_UP, BALANCE, FAULT) with
    hysteresis, minimum dwell time, slew-rate limiting and anti-chatter logic.
    Callable as a simulation command policy `(time_s, state) -> ActuatorCommand`.
    """

    def __init__(
        self,
        parameters: PendulumParameters,
        config: ControllerConfig | None = None,
    ) -> None:
        self.parameters = parameters
        self.parameters.validate()
        self.config = config or ControllerConfig()
        self.config.validate()

        self.lqr = LQRController(parameters, self.config.lqr)
        self.swing_up = EnergySwingUpController(parameters, self.config.swing_up)

        self._mode: ControllerMode = self.config.initial_mode
        self._dwell_time_s: float = 0.0
        self._previous_command_val: float = 0.0
        self._fault_reason: str | None = None

    @property
    def mode(self) -> ControllerMode:
        return self._mode

    @property
    def fault_reason(self) -> str | None:
        return self._fault_reason

    def arm(self) -> None:
        if self._mode == ControllerMode.DISARMED:
            self._mode = ControllerMode.SWING_UP
            self._dwell_time_s = 0.0

    def disarm(self) -> None:
        self._mode = ControllerMode.DISARMED
        self._previous_command_val = 0.0

    def reset(self) -> None:
        self._mode = self.config.initial_mode
        self._dwell_time_s = 0.0
        self._previous_command_val = 0.0
        self._fault_reason = None
        self.lqr.reset()

    def _fault(self, reason: str) -> ActuatorCommand:
        self._mode = ControllerMode.FAULT
        self._fault_reason = reason
        self._previous_command_val = 0.0
        return ActuatorCommand.disabled()

    def update_mode(self, state: State, dt_s: float) -> ControllerMode:
        """Evaluate mode transitions with hysteresis and minimum dwell time."""

        self._dwell_time_s += dt_s
        th_wrapped = abs(wrap_angle(state.pendulum_angle_rad))
        th_dot = abs(state.pendulum_angular_velocity_rad_s)

        if self._mode == ControllerMode.SWING_UP:
            if (
                th_wrapped <= self.config.catch_angle_rad
                and th_dot <= self.config.catch_angular_velocity_rad_s
            ):
                self._mode = ControllerMode.BALANCE
                self._dwell_time_s = 0.0

        elif self._mode == ControllerMode.BALANCE:
            if (
                th_wrapped > self.config.exit_angle_rad
                and self._dwell_time_s >= self.config.min_dwell_time_s
            ):
                self._mode = ControllerMode.SWING_UP
                self._dwell_time_s = 0.0
                self.lqr.reset()

        return self._mode

    def compute(
        self,
        time_s: float,
        state: State,
        dt_s: float = 0.002,
    ) -> ActuatorCommand:
        """Compute the actuator command for the current state and timestep."""

        if self._mode == ControllerMode.FAULT:
            return ActuatorCommand.disabled()
        if self._mode == ControllerMode.DISARMED:
            return ActuatorCommand.disabled()

        try:
            state.validate()
        except ValueError:
            return self._fault("non_finite_state")

        self.update_mode(state, dt_s)

        if self._mode == ControllerMode.BALANCE:
            target_force = self.lqr.compute_force(state, dt_s)
        elif self._mode == ControllerMode.SWING_UP:
            target_force = self.swing_up.compute_force(state)
        else:
            return ActuatorCommand.disabled()

        target_norm = max(-1.0, min(1.0, target_force / self.parameters.max_force_n))

        # Slew-rate limiting to stay strictly within safety supervisor limits
        max_delta = self.config.max_command_slew_per_s * dt_s
        delta = target_norm - self._previous_command_val
        if delta > max_delta:
            cmd_norm = self._previous_command_val + max_delta
        elif delta < -max_delta:
            cmd_norm = self._previous_command_val - max_delta
        else:
            cmd_norm = target_norm

        self._previous_command_val = cmd_norm
        return ActuatorCommand(cmd_norm)

    def __call__(self, time_s: float, state: State) -> ActuatorCommand:
        return self.compute(time_s, state)
