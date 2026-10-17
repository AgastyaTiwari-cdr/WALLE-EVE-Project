import argparse
import json
from pathlib import Path

from src.agent import AutonomousWallEAgent
from src.environment import SimulatedWastelandEnvironment
from src.schema import ActionTrace


def render_ascii_visualization(env, agent):
    map_pos, _ = agent.filter.get_map_estimate()
    print("\nWALL-E SEARCH GRID (agent view; true EVE position is intentionally hidden)")
    for y in range(env.grid_height - 1, -1, -1):
        cells = []
        for x in range(env.grid_width):
            pos = (x, y)
            if pos == env.agent_pos.to_tuple():
                symbol = "W"
            elif pos == map_pos:
                symbol = "?"
            elif any(tuple(clue["position"]) == pos for clue in env.discovered_clues):
                symbol = "C"
            else:
                symbol = "."
            cells.append(f"{symbol:>2}")
        print(f"{y:2} | " + " ".join(cells))
    print("     " + "--" * env.grid_width)
    print("      " + " ".join(f"{x:2}" for x in range(env.grid_width)))
    print("Legend: W=WALL-E, ?=current MAP estimate, C=discovered clue, .=unknown cell\n")


def run_mission(config_path: str, output_path: str | None = None, quiet: bool = False):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    env = SimulatedWastelandEnvironment(config)
    costs = env.costs
    agent = AutonomousWallEAgent(
        tuple(config["environment"]["grid_shape"]),
        config["environment"]["signal_decay_factor"],
        config["environment"]["sensor_noise_sigma"],
        move_cost=costs["MOVE"], scan_cost=costs["SCAN"],
        recharge_amount=costs["RECHARGE"],
        max_battery=env.max_battery)

    traces = []
    obs = env.get_observation()
    if not quiet:
        print(f"Starting mission: {config['mission_id']}")

    while not env.is_terminal():
        action, reasoning = agent.select_action(obs)
        map_pos, confidence = agent.filter.get_map_estimate()
        traces.append(ActionTrace(
            step=env.current_step + 1, action=action.value,
            agent_pos=obs.agent_pos.to_tuple(), battery_remaining=obs.battery_level,
            signal_strength=obs.raw_signal_strength,
            belief_entropy=round(agent.filter.compute_entropy(), 4),
            inferred_location=map_pos, posterior_confidence=round(confidence, 4),
            reasoning=reasoning, timestamp=obs.timestamp,
            signal_direction=obs.signal_direction).to_dict())
        obs = env.step(action)
        if not quiet:
            print(f"Step {env.current_step:02d}: {action.value:12s} pos={obs.agent_pos.to_tuple()} battery={obs.battery_level:3d} signal={obs.raw_signal_strength:.3f}")

    final_pos, confidence = agent.filter.get_map_estimate()
    status = "SUCCESS" if env.agent_pos == env.true_eve_pos else "FAILED"
    report = {
        "mission_id": config["mission_id"],
        "status": status,
        "termination_reason": env.termination_reason or ("EVE_FOUND" if status == "SUCCESS" else "MISSION_ENDED"),
        "total_steps": env.current_step,
        "final_battery": env.battery,
        "inferred_location": final_pos,
        "posterior_confidence": round(confidence, 4),
        "final_entropy_bits": round(agent.filter.compute_entropy(), 4),
        "evidence_log": env.discovered_clues,
        "recommended_next_step": (
            "Verify EVE's signal and establish communication." if status == "SUCCESS"
            else "Review the trace, increase the step/battery budget, or improve signal/clue coverage.")
    }
    if not quiet:
        render_ascii_visualization(env, agent)
        print("FINAL MISSION REPORT")
        print(json.dumps(report, indent=2))

    output = {"config": config, "action_traces": traces, "final_report": report}
    destination = Path(output_path) if output_path else config_path.parent / "mission_traces_and_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if not quiet:
        print(f"Saved mission trace and report to: {destination}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WALL-E & EVE autonomous signal-search agent")
    parser.add_argument("--config", default="config.json", help="Path to mission JSON config")
    parser.add_argument("--output", default=None, help="Optional path for trace/report JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step console output")
    args = parser.parse_args()
    run_mission(args.config, args.output, args.quiet)
