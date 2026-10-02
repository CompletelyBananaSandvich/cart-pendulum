import unittest

from cart_pendulum import (
    ActuatorCommand,
    CartPendulumModel,
    PendulumParameters,
    ProjectConfig,
    SafetyLimits,
    SafetySupervisor,
    SimulationConfig,
    SimulationRunner,
    State,
)
from cart_pendulum.simulation import zero_force_policy


class SafetyTest(unittest.TestCase):
    def test_safety_latches_position_fault_and_requires_reset(self) -> None:
        supervisor = SafetySupervisor(SafetyLimits(max_command_slew_per_s=100.0))
        result = supervisor.check(State(cart_position_m=2.0), ActuatorCommand(), 0.01)
        self.assertFalse(result.allowed)
        self.assertEqual(result.command, ActuatorCommand.disabled())
        self.assertEqual(result.reason, "cart_position_limit")

        healthy = supervisor.check(State(), ActuatorCommand(), 0.01)
        self.assertFalse(healthy.allowed)
        self.assertEqual(healthy.reason, "cart_position_limit")

        supervisor.reset()
        self.assertTrue(supervisor.check(State(), ActuatorCommand(), 0.01).allowed)

    def test_safety_rejects_fast_command_change(self) -> None:
        supervisor = SafetySupervisor(SafetyLimits(max_command_slew_per_s=1.0))
        result = supervisor.check(State(), ActuatorCommand(0.5), 0.01)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "command_slew_limit")


class SimulationTest(unittest.TestCase):
    def test_zero_force_simulation_stays_at_equilibrium(self) -> None:
        config = ProjectConfig(
            parameters=PendulumParameters(),
            simulation=SimulationConfig(
                dt_s=0.01,
                duration_s=0.1,
                initial_state=State(),
            ),
            safety=SafetyLimits(max_command_slew_per_s=100.0),
        )
        result = SimulationRunner(CartPendulumModel(config.parameters), config).run(
            zero_force_policy
        )
        self.assertTrue(result.completed)
        self.assertEqual(len(result.samples), 11)
        self.assertEqual(result.final_state, State())

    def test_simulation_stops_on_safety_violation(self) -> None:
        config = ProjectConfig(
            parameters=PendulumParameters(),
            simulation=SimulationConfig(
                dt_s=0.01,
                duration_s=1.0,
                initial_state=State(pendulum_angle_rad=0.5),
            ),
            safety=SafetyLimits(
                max_pendulum_angle_rad=0.6,
                max_command_slew_per_s=100.0,
            ),
        )
        result = SimulationRunner(CartPendulumModel(config.parameters), config).run()
        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "pendulum_angle_limit")
        self.assertFalse(result.samples[-1].safety_ok)


if __name__ == "__main__":
    unittest.main()
