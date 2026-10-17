"""Cooperative multi-agent search with shared Bayesian belief and evidence memory.

This is a lightweight team simulator: agents occupy independent grid positions,
share discoveries and posterior belief, and take one action per round.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.agent import AutonomousWallEAgent
from src.environment import SimulatedWastelandEnvironment
from src.schema import ActionType, Coordinate, Observation


class MultiAgentCoordinator:
    def __init__(self, config: dict[str, Any], agent_count: int = 2):
        if agent_count < 2:
            raise ValueError("agent_count must be at least 2")
        self.config = config
        self.world = SimulatedWastelandEnvironment(config)
        self.agent_count = agent_count
        env_cfg, init_cfg = config["environment"], config["agent_initial_state"]
        starts = config.get("team", {}).get("start_positions") or [init_cfg["start_position"]]
        while len(starts) < agent_count:
            # Deterministically spread agents across valid corners to reduce overlap.
            candidates = [(0, 0), (self.world.grid_width - 1, 0),
                          (0, self.world.grid_height - 1),
                          (self.world.grid_width - 1, self.world.grid_height - 1)]
            starts.append(list(candidates[len(starts) % len(candidates)]))
        self.positions = [Coordinate(*map(int, starts[i])) for i in range(agent_count)]
        for pos in self.positions:
            self.world._validate_position(pos.to_tuple(), "team start position")
        self.batteries = [int(init_cfg["battery_capacity"])] * agent_count
        self.max_battery = int(init_cfg["battery_capacity"])
        # Each robot receives the configured action budget; team budget scales with team size.
        self.step_budget = int(init_cfg["max_step_budget"]) * agent_count
        costs = self.world.costs
        self.agents = [AutonomousWallEAgent(
            tuple(env_cfg["grid_shape"]), env_cfg["signal_decay_factor"], env_cfg["sensor_noise_sigma"],
            move_cost=costs["MOVE"], scan_cost=costs["SCAN"],
            recharge_amount=costs["RECHARGE"], max_battery=self.max_battery)
            for _ in range(agent_count)]
        shared_filter = self.agents[0].filter
        shared_visits: Counter[tuple[int, int]] = Counter()
        shared_processed: set[tuple[int, int]] = set()
        shared_scanned: set[tuple[int, int]] = set()
        for agent in self.agents:
            agent.filter = shared_filter
            agent.visits = shared_visits
            agent.processed_clues = shared_processed
            agent.scanned_clues = shared_scanned
        self.shared_evidence: list[dict[str, Any]] = []
        self.evidence_positions: set[tuple[int, int]] = set()
        self.current_step = 0
        self.traces: list[dict[str, Any]] = []
        self.found_by: int | None = None

    def _observe(self, idx: int, scanned: bool = False) -> Observation:
        pos = self.positions[idx]
        clue_pos = pos.to_tuple()
        clue = self.world.clues.get(clue_pos)
        if clue and clue_pos not in self.evidence_positions:
            self.evidence_positions.add(clue_pos)
            self.shared_evidence.append(dict(clue))
        sigma = self.world.sensor_noise * (0.35 if scanned else 1.0)
        signal = self.world._compute_signal(pos, sigma)
        # Only the first visit supplies the clue as new evidence; the shared log is authoritative.
        if not hasattr(self, "_delivered_clues"):
            self._delivered_clues = set()
        clue_for_agent = clue if clue and clue_pos not in self._delivered_clues else None
        if clue_for_agent:
            self._delivered_clues.add(clue_pos)
        return Observation(pos, self.batteries[idx], signal, clue_for_agent, None, self.current_step, sigma)

    def run(self) -> dict[str, Any]:
        done = [False] * self.agent_count
        while self.current_step < self.step_budget and self.found_by is None and not all(done):
            for idx, agent in enumerate(self.agents):
                if self.current_step >= self.step_budget or self.found_by is not None:
                    break
                if done[idx]:
                    continue
                self.current_step += 1
                obs = self._observe(idx)
                action, reason = agent.select_action(obs)
                before = self.positions[idx]
                scanned = False
                if action in (ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH,
                              ActionType.MOVE_EAST, ActionType.MOVE_WEST):
                    dx, dy = {
                        ActionType.MOVE_NORTH: (0, 1), ActionType.MOVE_SOUTH: (0, -1),
                        ActionType.MOVE_EAST: (1, 0), ActionType.MOVE_WEST: (-1, 0)
                    }[action]
                    nx, ny = before.x + dx, before.y + dy
                    if 0 <= nx < self.world.grid_width and 0 <= ny < self.world.grid_height:
                        self.positions[idx] = Coordinate(nx, ny)
                        self.batteries[idx] = max(0, self.batteries[idx] - self.world.costs["MOVE"])
                elif action == ActionType.SCAN:
                    self.batteries[idx] = max(0, self.batteries[idx] - self.world.costs["SCAN"])
                    scanned = True
                elif action == ActionType.RECHARGE:
                    self.batteries[idx] = min(self.max_battery, self.batteries[idx] + self.world.costs["RECHARGE"])
                after = self.positions[idx]
                self.traces.append({"step": self.current_step, "agent_id": idx + 1,
                    "action": action.value, "from": before.to_tuple(), "position": after.to_tuple(),
                    "battery_remaining": self.batteries[idx], "signal_strength": obs.raw_signal_strength,
                    "inferred_location": agent.filter.get_map_estimate()[0], "reasoning": reason})
                if after == self.world.true_eve_pos:
                    self.found_by = idx + 1
                    break
                if self.batteries[idx] <= 0:
                    done[idx] = True
                elif scanned:
                    # Assimilate the higher-quality post-scan reading immediately.
                    agent._update_belief_once(self._observe(idx, scanned=True))
        map_pos, confidence = self.agents[0].filter.get_map_estimate()
        success = self.found_by is not None
        return {"mission_id": self.config.get("mission_id", "MULTI_AGENT_MISSION"),
                "mode": "cooperative_multi_agent", "agent_count": self.agent_count,
                "status": "SUCCESS" if success else "FAILED",
                "termination_reason": "EVE_FOUND" if success else ("STEP_BUDGET_EXHAUSTED" if self.current_step >= self.step_budget else "ALL_AGENTS_OUT_OF_BATTERY"),
                "found_by_agent": self.found_by, "total_team_actions": self.current_step,
                "positions": [p.to_tuple() for p in self.positions], "batteries": self.batteries,
                "inferred_location": map_pos, "posterior_confidence": round(confidence, 4),
                "evidence_log": self.shared_evidence, "action_traces": self.traces}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cooperative WALL-E team search")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--output", default="multi_agent_report.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = MultiAgentCoordinator(config, args.agents).run()
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "action_traces"}, indent=2))
    print(f"Saved full team trace to {args.output}")


if __name__ == "__main__":
    main()
