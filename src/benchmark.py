"""Planner benchmark: classical baseline, information-seeking heuristic, optional real LLM.

The LLM planner uses the OpenAI-compatible Chat Completions API only when explicitly
requested and OPENAI_API_KEY is set. No hidden ground-truth state is sent to the model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from src.agent import AutonomousWallEAgent
from src.belief_filter import GridBayesianBeliefFilter
from src.environment import SimulatedWastelandEnvironment
from src.schema import ActionType

MOVE_ACTIONS = {ActionType.MOVE_NORTH, ActionType.MOVE_SOUTH, ActionType.MOVE_EAST, ActionType.MOVE_WEST}


class ClassicalMAPPlanner:
    """Transparent baseline: update Bayesian belief, then greedily move toward MAP."""
    def __init__(self, config: dict[str, Any], env: SimulatedWastelandEnvironment):
        self.env = env
        self.filter = GridBayesianBeliefFilter(tuple(config["environment"]["grid_shape"]),
            config["environment"]["signal_decay_factor"], config["environment"]["sensor_noise_sigma"])
        self.visited: set[tuple[int, int]] = set()
        self.processed_clues: set[tuple[int, int]] = set()
        self.last_key = None
        self.action_count = 0

    def select_action(self, obs):
        pos = obs.agent_pos.to_tuple()
        key = (pos, obs.raw_signal_strength, repr(obs.detected_clue), obs.timestamp, obs.measurement_sigma)
        if key != self.last_key:
            clue = None
            if obs.detected_clue and pos not in self.processed_clues:
                clue = obs.detected_clue
                self.processed_clues.add(pos)
            self.filter.update(pos, obs.raw_signal_strength, clue, sigma=obs.measurement_sigma)
            self.last_key = key
        self.visited.add(pos)
        self.action_count += 1
        if obs.battery_level <= max(self.env.costs["MOVE"] * 2, self.env.costs["SCAN"]):
            return ActionType.RECHARGE, "Classical baseline: preserve enough battery to continue searching."
        # A simple fixed-cadence sensing rule makes this a fair baseline, not a movement-only policy.
        if self.action_count % 6 == 0 and obs.battery_level >= self.env.costs["SCAN"] + self.env.costs["MOVE"]:
            return ActionType.SCAN, "Classical baseline: scheduled scan every six actions."
        target, _ = self.filter.get_map_estimate()
        x, y = pos; tx, ty = target
        if x < tx: action = ActionType.MOVE_EAST
        elif x > tx: action = ActionType.MOVE_WEST
        elif y < ty: action = ActionType.MOVE_NORTH
        elif y > ty: action = ActionType.MOVE_SOUTH
        else: action = ActionType.SCAN
        return action, f"Classical MAP baseline: greedily move toward posterior mode {target}."


class LLMPlanner:
    """Optional real LLM planner; requires OPENAI_API_KEY and network access at runtime."""
    def __init__(self, config: dict[str, Any], env: SimulatedWastelandEnvironment, model: str):
        self.env, self.model = env, model
        self.base = AutonomousWallEAgent(tuple(config["environment"]["grid_shape"]),
            config["environment"]["signal_decay_factor"], config["environment"]["sensor_noise_sigma"],
            move_cost=env.costs["MOVE"], scan_cost=env.costs["SCAN"],
            recharge_amount=env.costs["RECHARGE"], max_battery=env.max_battery)
        self.fallback_count = 0
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY to use --include-llm; no key is stored in the project.")

    def select_action(self, obs):
        self.base._update_belief_once(obs)
        belief = self.base.filter
        top = [{"position": list(pos), "probability": round(prob, 4)}
               for pos, prob in belief.ranked_locations()[:8]]
        state = {"current_position": list(obs.agent_pos.to_tuple()),
                 "battery": obs.battery_level, "signal_strength": obs.raw_signal_strength,
                 "measurement_sigma": obs.measurement_sigma,
                 "new_clue": obs.detected_clue, "top_belief_hypotheses": top,
                 "entropy_bits": round(belief.compute_entropy(), 3),
                 "grid_shape": list(self.base.grid_shape),
                 "actions": [a.value for a in ActionType]}
        prompt = ("You control WALL-E in a partially observed grid. Choose exactly one action. "
                  "Do not assume hidden EVE position. Minimize wasted movement, gather clues, use SCAN "
                  "when useful, and RECHARGE before running out of battery. Return JSON only: "
                  '{"action":"MOVE_NORTH|MOVE_SOUTH|MOVE_EAST|MOVE_WEST|SCAN|RECHARGE", "reasoning":"..."}.\n'
                  "Current observation and belief summary:\n" + json.dumps(state))
        payload = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": "You are a cautious search-planning agent. Output valid JSON only."},
                                {"role": "user", "content": prompt}]}
        request = urllib.request.Request("https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode())
            answer = json.loads(body["choices"][0]["message"]["content"])
            action = ActionType(answer["action"])
            reason = str(answer.get("reasoning", "LLM-selected action"))[:500]
            pos = obs.agent_pos.to_tuple()
            deltas = {ActionType.MOVE_NORTH: (0, 1), ActionType.MOVE_SOUTH: (0, -1),
                      ActionType.MOVE_EAST: (1, 0), ActionType.MOVE_WEST: (-1, 0)}
            invalid_move = action in deltas and not (
                0 <= pos[0] + deltas[action][0] < self.base.grid_shape[0] and
                0 <= pos[1] + deltas[action][1] < self.base.grid_shape[1])
            invalid_scan = action == ActionType.SCAN and obs.battery_level < self.env.costs["SCAN"]
            invalid_recharge = action == ActionType.RECHARGE and obs.battery_level >= self.env.max_battery
            if invalid_move or invalid_scan or invalid_recharge:
                self.fallback_count += 1
                fallback_action, fallback_reason = self.base.select_action(obs)
                return fallback_action, f"LLM proposed an invalid/inefficient action; local fallback used. {fallback_reason}"
            return action, "LLM planner: " + reason
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            # Fail safely to a local planner if one API response is malformed/unavailable.
            self.fallback_count += 1
            action, reason = self.base.select_action(obs)
            return action, f"LLM request failed; safe local fallback used ({type(exc).__name__}). {reason}"


def run_one(config_template: dict[str, Any], planner_name: str, seed: int, llm_model: str) -> dict[str, Any]:
    config = json.loads(json.dumps(config_template))
    config["random_seed"] = seed
    # Avoid a stale precomputed reading masking the seeded noisy sensor in repeated trials.
    config.pop("initial_signal_reading", None)
    env = SimulatedWastelandEnvironment(config)
    if planner_name == "classical":
        planner = ClassicalMAPPlanner(config, env)
    elif planner_name == "information":
        costs = env.costs
        planner = AutonomousWallEAgent(tuple(config["environment"]["grid_shape"]),
            config["environment"]["signal_decay_factor"], config["environment"]["sensor_noise_sigma"],
            move_cost=costs["MOVE"], scan_cost=costs["SCAN"], recharge_amount=costs["RECHARGE"],
            max_battery=env.max_battery)
    elif planner_name == "llm":
        planner = LLMPlanner(config, env, llm_model)
    else:
        raise ValueError(planner_name)
    obs = env.get_observation()
    actions = 0; scans = 0
    while not env.is_terminal():
        action, _reason = planner.select_action(obs)
        scans += int(action == ActionType.SCAN)
        actions += 1
        obs = env.step(action)
    filter_obj = planner.filter if planner_name == "classical" else planner.base.filter if planner_name == "llm" else planner.filter
    return {"planner": planner_name, "seed": seed,
        "success": env.agent_pos == env.true_eve_pos, "steps": actions,
        "battery_remaining": env.battery, "clues_found": len(env.discovered_clues),
        "scans": scans, "llm_fallbacks": getattr(planner, "fallback_count", 0),
        "posterior_confidence": round(filter_obj.get_map_estimate()[1], 4),
        "termination_reason": env.termination_reason}


def benchmark(config_path: str, seeds: int = 20, include_llm: bool = False, llm_model: str = "gpt-4o-mini") -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    planners = ["classical", "information"] + (["llm"] if include_llm else [])
    rows = []
    for planner_name in planners:
        for seed in range(seeds):
            rows.append(run_one(config, planner_name, seed, llm_model))
    summaries = {}
    for name in planners:
        group = [r for r in rows if r["planner"] == name]
        summaries[name] = {"missions": len(group),
            "success_rate": round(sum(r["success"] for r in group) / len(group), 3),
            "mean_steps": round(sum(r["steps"] for r in group) / len(group), 2),
            "mean_battery_remaining": round(sum(r["battery_remaining"] for r in group) / len(group), 2),
            "mean_clues_found": round(sum(r["clues_found"] for r in group) / len(group), 2),
            "mean_scans": round(sum(r["scans"] for r in group) / len(group), 2)}
    return {"benchmark": "WALL-E planner comparison", "seeds": seeds,
            "llm_is_real_api_model": include_llm, "summary": summaries, "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classical, information-seeking, and optional LLM planners")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--include-llm", action="store_true", help="Call a real LLM API; requires OPENAI_API_KEY and incurs API cost")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--output", default="planner_benchmark.json")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    report = benchmark(args.config, args.seeds, args.include_llm, args.llm_model)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
