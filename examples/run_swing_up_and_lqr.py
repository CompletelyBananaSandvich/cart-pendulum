"""Run closed-loop coursework swing-up and LQR balance simulation.

Usage:
    PYTHONPATH=src python examples/run_swing_up_and_lqr.py
"""

import math
from pathlib import Path

from cart_pendulum import (
    CartPendulumController,
    CartPendulumModel,
    ControllerConfig,
    ControllerMode,
    ProjectConfig,
    SafetyLimits,
    SimulationConfig,
    SimulationRunner,
    State,
    load_config,
    wrap_angle,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base_config = load_config(ROOT / "configs" / "default.toml")

    # Swing-up simulation starts with the pendulum hanging straight down
    # and allows unconstrained angle during the swing-up maneuver.
    config = ProjectConfig(
        parameters=base_config.parameters,
        simulation=SimulationConfig(
            dt_s=0.002,
            duration_s=5.0,
            initial_state=State(
                cart_position_m=0.0,
                cart_velocity_m_s=0.0,
                pendulum_angle_rad=math.pi - 0.02,
                pendulum_angular_velocity_rad_s=0.0,
            ),
        ),
        safety=SafetyLimits(
            max_cart_position_m=1.8,
            max_cart_velocity_m_s=3.0,
            max_pendulum_angle_rad=None,  # Unconstrained during swing-up
            max_pendulum_angular_velocity_rad_s=15.0,
            max_command_slew_per_s=5.0,
        ),
    )

    model = CartPendulumModel(config.parameters)
    controller = CartPendulumController(
        config.parameters,
        ControllerConfig(initial_mode=ControllerMode.SWING_UP),
    )

    runner = SimulationRunner(model, config)
    result = runner.run(controller)

    output = ROOT / "artifacts" / "swing_up_and_lqr.csv"
    result.write_csv(output)

    final_state = result.final_state
    final_th = wrap_angle(final_state.pendulum_angle_rad)

    print("=== Swing-up and LQR Simulation Result ===")
    print(f"Completed:            {result.completed}")
    print(f"Termination reason:   {result.termination_reason}")
    print(f"Controller final mode:{controller.mode.value}")
    print(f"Total samples:        {len(result.samples)}")
    print(f"Final cart position:  {final_state.cart_position_m:+.4f} m")
    print(f"Final cart velocity:  {final_state.cart_velocity_m_s:+.4f} m/s")
    print(f"Final pendulum angle: {math.degrees(final_th):+.2f} deg ({final_th:+.4f} rad)")
    print(f"Final angular vel:    {final_state.pendulum_angular_velocity_rad_s:+.4f} rad/s")
    print(f"Log written to:       {output}")


if __name__ == "__main__":
    main()
