import math
from pathlib import Path
import unittest

from cart_pendulum import ActuatorCommand, State, load_config, wrap_angle


ROOT = Path(__file__).resolve().parents[1]


class InterfacesTest(unittest.TestCase):
    def test_wrap_angle_uses_principal_interval(self) -> None:
        self.assertGreaterEqual(wrap_angle(-math.pi), -math.pi)
        self.assertLess(wrap_angle(math.pi), math.pi)
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi / 2.0), -math.pi / 2.0)
        self.assertAlmostEqual(wrap_angle(-5.0 * math.pi / 2.0), -math.pi / 2.0)

    def test_state_rejects_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            State(pendulum_angle_rad=float("nan")).validate()

    def test_command_is_hardware_independent_and_clamped_explicitly(self) -> None:
        self.assertEqual(ActuatorCommand.clamped(2.0).normalized, 1.0)
        self.assertEqual(ActuatorCommand.clamped(-2.0).normalized, -1.0)
        with self.assertRaises(ValueError):
            ActuatorCommand(1.1)


class ConfigTest(unittest.TestCase):
    def test_default_toml_loads_and_validates(self) -> None:
        config = load_config(ROOT / "configs" / "default.toml")
        self.assertEqual(config.simulation.step_count, 2000)
        self.assertAlmostEqual(config.parameters.total_mass_kg, 0.70)
        self.assertGreater(config.parameters.pendulum_inertia_about_pivot_kg_m2, 0.0)


if __name__ == "__main__":
    unittest.main()
