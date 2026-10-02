"""Nonlinear cart-pendulum dynamics."""

from __future__ import annotations

import math

from .config import PendulumParameters
from .interfaces import ActuatorCommand, State


class CartPendulumModel:
    """Planar cart-pendulum model with a force input.

    Coordinates and signs:

    * ``x`` grows to the right;
    * ``theta = 0`` is the upright position;
    * positive ``theta`` places the pendulum centre of mass to the right of
      the pivot;
    * positive command applies force in the positive-x direction.

    The pendulum is treated as a rigid body. Its inertia parameter is about
    the centre of mass; the parallel-axis term is added internally.
    """

    def __init__(self, parameters: PendulumParameters | None = None) -> None:
        self.parameters = parameters or PendulumParameters()
        self.parameters.validate()

    def accelerations(self, state: State, command: ActuatorCommand) -> tuple[float, float]:
        """Return cart and angular acceleration for the current state."""

        state.validate()
        force = self.parameters.max_force_n * command.normalized if command.enabled else 0.0
        p = self.parameters
        sin_theta = math.sin(state.pendulum_angle_rad)
        cos_theta = math.cos(state.pendulum_angle_rad)
        coupling = p.pendulum_mass_kg * p.center_of_mass_length_m * cos_theta
        inertia = p.pendulum_inertia_about_pivot_kg_m2

        # M(q) q_ddot = rhs. The equations follow from
        # y_com = l*cos(theta), x_com = x + l*sin(theta), with theta measured
        # from upright. Viscous friction is applied to both coordinates.
        rhs_x = (
            force
            - p.cart_viscous_friction_n_s_per_m * state.cart_velocity_m_s
            + p.pendulum_mass_kg
            * p.center_of_mass_length_m
            * sin_theta
            * state.pendulum_angular_velocity_rad_s**2
        )
        rhs_theta = (
            p.pendulum_mass_kg
            * p.gravitational_acceleration_m_s2
            * p.center_of_mass_length_m
            * sin_theta
            - p.pendulum_viscous_friction_n_m_s_per_rad
            * state.pendulum_angular_velocity_rad_s
        )

        mass_term = p.total_mass_kg
        determinant = mass_term * inertia - coupling * coupling
        if determinant <= 0.0 or not math.isfinite(determinant):
            raise ValueError("dynamics mass matrix is not positive definite")

        cart_acceleration = (inertia * rhs_x - coupling * rhs_theta) / determinant
        angular_acceleration = (-coupling * rhs_x + mass_term * rhs_theta) / determinant
        if not math.isfinite(cart_acceleration) or not math.isfinite(angular_acceleration):
            raise FloatingPointError("non-finite acceleration")
        return cart_acceleration, angular_acceleration

    def derivative(self, state: State, command: ActuatorCommand) -> State:
        cart_acceleration, angular_acceleration = self.accelerations(state, command)
        return State(
            cart_position_m=state.cart_velocity_m_s,
            cart_velocity_m_s=cart_acceleration,
            pendulum_angle_rad=state.pendulum_angular_velocity_rad_s,
            pendulum_angular_velocity_rad_s=angular_acceleration,
        )

    def step(self, state: State, command: ActuatorCommand, dt_s: float) -> State:
        """Advance by one fixed step using classical fourth-order RK."""

        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and greater than zero")
        state.validate()

        # State arithmetic is explicit to keep the public State immutable and
        # to avoid a dependency on a numerical package at this stage.
        def add_scaled(base: State, derivative: State, scale: float) -> State:
            return State(
                base.cart_position_m + scale * derivative.cart_position_m,
                base.cart_velocity_m_s + scale * derivative.cart_velocity_m_s,
                base.pendulum_angle_rad + scale * derivative.pendulum_angle_rad,
                base.pendulum_angular_velocity_rad_s
                + scale * derivative.pendulum_angular_velocity_rad_s,
            )

        half_dt = 0.5 * dt_s
        k1 = self.derivative(state, command)
        k2 = self.derivative(add_scaled(state, k1, half_dt), command)
        k3 = self.derivative(add_scaled(state, k2, half_dt), command)
        k4 = self.derivative(add_scaled(state, k3, dt_s), command)
        next_state = State(
            state.cart_position_m
            + dt_s
            / 6.0
            * (
                k1.cart_position_m
                + 2.0 * k2.cart_position_m
                + 2.0 * k3.cart_position_m
                + k4.cart_position_m
            ),
            state.cart_velocity_m_s
            + dt_s
            / 6.0
            * (
                k1.cart_velocity_m_s
                + 2.0 * k2.cart_velocity_m_s
                + 2.0 * k3.cart_velocity_m_s
                + k4.cart_velocity_m_s
            ),
            state.pendulum_angle_rad
            + dt_s
            / 6.0
            * (
                k1.pendulum_angle_rad
                + 2.0 * k2.pendulum_angle_rad
                + 2.0 * k3.pendulum_angle_rad
                + k4.pendulum_angle_rad
            ),
            state.pendulum_angular_velocity_rad_s
            + dt_s
            / 6.0
            * (
                k1.pendulum_angular_velocity_rad_s
                + 2.0 * k2.pendulum_angular_velocity_rad_s
                + 2.0 * k3.pendulum_angular_velocity_rad_s
                + k4.pendulum_angular_velocity_rad_s
            ),
        )
        next_state.validate()
        return next_state
