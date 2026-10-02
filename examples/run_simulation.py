"""Run the first deterministic simulation and write a CSV log.

Usage:
    PYTHONPATH=src python examples/run_simulation.py
"""

from pathlib import Path

from cart_pendulum import CartPendulumModel, SimulationRunner, load_config
from cart_pendulum.simulation import zero_force_policy


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "configs" / "default.toml")
    model = CartPendulumModel(config.parameters)
    result = SimulationRunner(model, config).run(zero_force_policy)
    output = ROOT / "artifacts" / "open_loop_zero_force.csv"
    result.write_csv(output)
    print(f"completed={result.completed}")
    print(f"termination_reason={result.termination_reason}")
    print(f"samples={len(result.samples)}")
    print(f"final_state={result.final_state}")
    print(f"log={output}")


if __name__ == "__main__":
    main()
