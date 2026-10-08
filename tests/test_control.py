import math
import unittest

from cart_pendulum import (
    ActuatorCommand,
    CartPendulumController,
    CartPendulumModel,
    ControllerConfig,
    ControllerMode,
    DEFAULT_LQR_GAINS,
    EnergySwingUpController,
    LQRConfig,
    LQRController,
    PendulumParameters,
    ProjectConfig,
    SafetyLimits,
    SimulationConfig,
    SimulationRunner,
    State,
    SwingUpConfig,
    compute_lqr_gain,
    linearize_dynamics,
    wrap_angle,
)


class ControlLinearizationAndLQRTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = PendulumParameters()

    def test_linearize_dynamics_returns_expected_shapes_and_values(self) -> None:
        A, B = linearize_dynamics(self.params)
        self.assertEqual(len(A), 4)
        self.assertEqual(len(A[0]), 4)
        self.assertEqual(len(B), 4)
        self.assertEqual(len(B[0]), 1)

        # Kinematic rows
        self.assertEqual(A[0], [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(A[2], [0.0, 0.0, 0.0, 1.0])

        # Positive force pushes cart positive and tilts pendulum negative
        self.assertGreater(B[1][0], 0.0)
        self.assertLess(B[3][0], 0.0)

        # Pendulum angle stiffness around upright is positive (unstable)
        self.assertGreater(A[3][2], 0.0)

    def test_compute_lqr_gain_returns_stabilizing_gains(self) -> None:
        gains = compute_lqr_gain(self.params)
        self.assertEqual(len(gains), 4)
        # All gains should be negative so that force = - sum(K * z) gives positive feedback on errors
        for g in gains:
            self.assertLess(g, 0.0)
        # Compare with default nominal constants
        self.assertAlmostEqual(gains[0], DEFAULT_LQR_GAINS[0], delta=0.1)
        self.assertAlmostEqual(gains[1], DEFAULT_LQR_GAINS[1], delta=0.1)
        self.assertAlmostEqual(gains[2], DEFAULT_LQR_GAINS[2], delta=0.2)
        self.assertAlmostEqual(gains[3], DEFAULT_LQR_GAINS[3], delta=0.2)

    def test_compute_lqr_gain_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_lqr_gain(self.params, q_diag=(1.0, 2.0, 3.0))  # too short
        with self.assertRaises(ValueError):
            compute_lqr_gain(self.params, r_scalar=-1.0)


class LQRControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = PendulumParameters()
        self.controller = LQRController(self.params)
        self.model = CartPendulumModel(self.params)

    def test_zero_equilibrium_produces_zero_force(self) -> None:
        force = self.controller.compute_force(State())
        self.assertAlmostEqual(force, 0.0, places=9)
        cmd = self.controller.compute_command(State())
        self.assertEqual(cmd, ActuatorCommand(0.0))

    def test_positive_angle_error_commands_positive_restoring_force(self) -> None:
        state = State(pendulum_angle_rad=0.05)
        force = self.controller.compute_force(state)
        self.assertGreater(force, 0.0)

    def test_large_error_is_clamped(self) -> None:
        state = State(pendulum_angle_rad=0.8)
        force = self.controller.compute_force(state)
        self.assertEqual(force, self.params.max_force_n)
        cmd = self.controller.compute_command(state)
        self.assertEqual(cmd.normalized, 1.0)

    def test_lqr_stabilizes_initial_angle_perturbation(self) -> None:
        state = State(pendulum_angle_rad=0.12, cart_position_m=0.1)
        dt = 0.002
        for _ in range(1500):  # 3.0 seconds
            cmd = self.controller.compute_command(state, dt)
            state = self.model.step(state, cmd, dt)

        self.assertAlmostEqual(state.cart_position_m, 0.0, delta=0.03)
        self.assertAlmostEqual(wrap_angle(state.pendulum_angle_rad), 0.0, delta=0.01)
        self.assertAlmostEqual(state.cart_velocity_m_s, 0.0, delta=0.05)
        self.assertAlmostEqual(state.pendulum_angular_velocity_rad_s, 0.0, delta=0.05)


class EnergySwingUpControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = PendulumParameters()
        self.controller = EnergySwingUpController(self.params)

    def test_upright_energy_is_zero_error(self) -> None:
        upright_state = State(pendulum_angle_rad=0.0)
        e_err = self.controller.energy_error(upright_state)
        self.assertAlmostEqual(e_err, 0.0, places=9)

    def test_hanging_down_energy_error_is_negative(self) -> None:
        hanging_state = State(pendulum_angle_rad=math.pi)
        e_err = self.controller.energy_error(hanging_state)
        expected = -2.0 * (
            self.params.pendulum_mass_kg
            * self.params.gravitational_acceleration_m_s2
            * self.params.center_of_mass_length_m
        )
        self.assertAlmostEqual(e_err, expected, places=7)

    def test_swing_up_force_pumps_energy_in_correct_direction(self) -> None:
        # Pendulum at bottom swinging counter-clockwise (th = pi, th_dot > 0)
        state = State(
            pendulum_angle_rad=math.pi,
            pendulum_angular_velocity_rad_s=1.0,
        )
        force = self.controller.compute_force(state)
        # Should push cart positive to accelerate swinging up
        self.assertGreater(force, 0.0)

    def test_cart_centering_opposes_large_cart_offset(self) -> None:
        state_centered = State(pendulum_angle_rad=math.pi, cart_position_m=0.0)
        state_offset = State(pendulum_angle_rad=math.pi, cart_position_m=0.8)
        force_centered = self.controller.compute_force(state_centered)
        force_offset = self.controller.compute_force(state_offset)
        # Offset to the right produces less force (pushes left)
        self.assertLess(force_offset, force_centered)


class CartPendulumControllerCoordinatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = PendulumParameters()
        self.model = CartPendulumModel(self.params)

    def test_initial_mode_and_transitions(self) -> None:
        ctrl = CartPendulumController(self.params)
        self.assertEqual(ctrl.mode, ControllerMode.SWING_UP)

        # Simulate state near upright within catch window
        upright_state = State(pendulum_angle_rad=0.05, pendulum_angular_velocity_rad_s=0.1)
        ctrl.compute(0.0, upright_state, 0.002)
        self.assertEqual(ctrl.mode, ControllerMode.BALANCE)

        # Small disturbance stays in BALANCE
        disturbed = State(pendulum_angle_rad=0.30, pendulum_angular_velocity_rad_s=0.5)
        ctrl.compute(0.01, disturbed, 0.002)
        self.assertEqual(ctrl.mode, ControllerMode.BALANCE)

        # Large angle exceeding exit threshold returns to SWING_UP after dwell time
        fallen = State(pendulum_angle_rad=1.0)
        # Before dwell time
        ctrl.compute(0.02, fallen, 0.002)
        self.assertEqual(ctrl.mode, ControllerMode.BALANCE)
        # Advance dwell time past min_dwell_time_s (0.10s)
        for i in range(60):
            ctrl.compute(0.02 + i * 0.002, fallen, 0.002)
        self.assertEqual(ctrl.mode, ControllerMode.SWING_UP)

    def test_disarmed_and_arm(self) -> None:
        cfg = ControllerConfig(initial_mode=ControllerMode.DISARMED)
        ctrl = CartPendulumController(self.params, cfg)
        self.assertEqual(ctrl.mode, ControllerMode.DISARMED)
        cmd = ctrl.compute(0.0, State())
        self.assertFalse(cmd.enabled)

        ctrl.arm()
        self.assertEqual(ctrl.mode, ControllerMode.SWING_UP)
        ctrl.disarm()
        self.assertEqual(ctrl.mode, ControllerMode.DISARMED)

    def test_non_finite_state_triggers_fault(self) -> None:
        ctrl = CartPendulumController(self.params)
        cmd = ctrl.compute(0.0, State(cart_position_m=float("nan")))
        self.assertEqual(ctrl.mode, ControllerMode.FAULT)
        self.assertFalse(cmd.enabled)
        self.assertEqual(ctrl.fault_reason, "non_finite_state")

    def test_full_swing_up_and_lqr_simulation(self) -> None:
        config = ProjectConfig(
            parameters=self.params,
            simulation=SimulationConfig(
                dt_s=0.002,
                duration_s=4.0,
                # Start hanging down almost at rest
                initial_state=State(pendulum_angle_rad=math.pi - 0.02),
            ),
            safety=SafetyLimits(
                max_cart_position_m=1.8,
                max_cart_velocity_m_s=3.0,
                max_pendulum_angle_rad=None,  # permissive for swing-up
                max_pendulum_angular_velocity_rad_s=15.0,
                max_command_slew_per_s=5.0,
            ),
        )

        controller = CartPendulumController(config.parameters)
        runner = SimulationRunner(self.model, config)
        result = runner.run(controller)

        self.assertTrue(result.completed)
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(controller.mode, ControllerMode.BALANCE)

        final = result.final_state
        final_angle_wrapped = abs(wrap_angle(final.pendulum_angle_rad))
        self.assertLess(final_angle_wrapped, 0.02)  # within 1.1 degrees
        self.assertLess(abs(final.cart_position_m), 0.20)  # within 20 cm
        self.assertLess(abs(final.cart_velocity_m_s), 0.15)
        self.assertLess(abs(final.pendulum_angular_velocity_rad_s), 0.10)


if __name__ == "__main__":
    unittest.main()
