import unittest

from cart_pendulum import ActuatorCommand, CartPendulumModel, PendulumParameters, State


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CartPendulumModel(
            PendulumParameters(
                cart_viscous_friction_n_s_per_m=0.0,
                pendulum_viscous_friction_n_m_s_per_rad=0.0,
            )
        )

    def test_upright_rest_state_is_equilibrium(self) -> None:
        state = State()
        cart_acceleration, angular_acceleration = self.model.accelerations(
            state, ActuatorCommand()
        )
        self.assertAlmostEqual(cart_acceleration, 0.0, places=12)
        self.assertAlmostEqual(angular_acceleration, 0.0, places=12)

    def test_upright_equilibrium_is_unstable_in_expected_direction(self) -> None:
        state = State(pendulum_angle_rad=0.01)
        _, angular_acceleration = self.model.accelerations(state, ActuatorCommand())
        self.assertGreater(angular_acceleration, 0.0)

    def test_positive_force_accelerates_cart_positive(self) -> None:
        state = State()
        cart_acceleration, angular_acceleration = self.model.accelerations(
            state, ActuatorCommand(0.2)
        )
        self.assertGreater(cart_acceleration, 0.0)
        self.assertLess(angular_acceleration, 0.0)

    def test_rk4_step_keeps_zero_equilibrium_at_zero(self) -> None:
        state = self.model.step(State(), ActuatorCommand(), 0.01)
        self.assertEqual(state, State())

    def test_step_rejects_invalid_timestep(self) -> None:
        with self.assertRaises(ValueError):
            self.model.step(State(), ActuatorCommand(), 0.0)


if __name__ == "__main__":
    unittest.main()
