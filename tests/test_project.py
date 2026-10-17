import json
from pathlib import Path

import numpy as np
import pytest

from src.agent import AutonomousWallEAgent
from src.belief_filter import GridBayesianBeliefFilter
from src.environment import SimulatedWastelandEnvironment
from src.schema import ActionType, Coordinate, Observation
from main import run_mission

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return json.loads((ROOT / "config.json").read_text())


def test_belief_normalizes_and_entropy_is_valid():
    f = GridBayesianBeliefFilter((5, 4), 100, 2)
    f.update((0, 0), 10)
    assert np.isclose(f.belief.sum(), 1.0)
    assert np.all(f.belief >= 0)
    assert 0 <= f.compute_entropy() <= np.log2(20) + 1e-9


def test_invalid_sigma_rejected():
    with pytest.raises(ValueError):
        GridBayesianBeliefFilter((3, 3), 100, 0)


def test_environment_observation_does_not_expose_true_position():
    env = SimulatedWastelandEnvironment(load_config())
    obs = env.get_observation()
    assert not hasattr(obs, "true_eve_pos")
    assert isinstance(obs.agent_pos, Coordinate)
    assert obs.timestamp == 0
    assert obs.signal_direction == (1, 1)
    assert obs.raw_signal_strength == 1.2


def test_recharge_is_capped_at_capacity():
    env = SimulatedWastelandEnvironment(load_config())
    env.battery = 10
    env.step(ActionType.RECHARGE)
    assert env.battery == 35
    env.battery = 95
    env.step(ActionType.RECHARGE)
    assert env.battery == 100


def test_mission_stops_within_budget_and_writes_report(tmp_path):
    result = run_mission(str(ROOT / "config.json"), str(tmp_path / "report.json"), quiet=True)
    assert result["final_report"]["total_steps"] <= load_config()["agent_initial_state"]["max_step_budget"]
    assert result["final_report"]["status"] == "SUCCESS"
    assert result["final_report"]["termination_reason"] == "EVE_FOUND"
    assert tuple(result["final_report"]["inferred_location"]) == (7, 8)
    assert result["final_report"]["evidence_log"], "sample mission should discover at least one clue"
    assert any(trace["action"] == "SCAN" for trace in result["action_traces"])
    assert all("reasoning" in trace and "belief_entropy" in trace for trace in result["action_traces"])
    assert (tmp_path / "report.json").exists()
    assert len(result["action_traces"]) <= load_config()["agent_initial_state"]["max_step_budget"]


def test_agent_returns_supported_action():
    agent = AutonomousWallEAgent((4, 4), 100, 2)
    obs = Observation(Coordinate(0, 0), 100, 20.0)
    action, reason = agent.select_action(obs)
    assert isinstance(action, ActionType)
    assert reason


def test_decoys_and_sensor_anomalies_are_configurable():
    config = load_config()
    env = SimulatedWastelandEnvironment(config)
    assert len(env.decoys) >= 1
    assert 0 <= env.dropout_probability <= 1
    assert 0 <= env.outlier_probability <= 1


def test_cooperative_team_shares_evidence_and_can_find_eve():
    from src.multi_agent import MultiAgentCoordinator
    result = MultiAgentCoordinator(load_config(), agent_count=2).run()
    assert result["mode"] == "cooperative_multi_agent"
    assert result["agent_count"] == 2
    assert result["total_team_actions"] <= 2 * load_config()["agent_initial_state"]["max_step_budget"]
    assert result["status"] == "SUCCESS"
    assert result["found_by_agent"] in (1, 2)
    assert result["evidence_log"]


def test_planner_benchmark_compares_baselines(tmp_path):
    from src.benchmark import benchmark
    report = benchmark(str(ROOT / "config.json"), seeds=2)
    assert set(report["summary"]) == {"classical", "information"}
    for metrics in report["summary"].values():
        assert metrics["missions"] == 2
        assert 0 <= metrics["success_rate"] <= 1
        assert metrics["mean_steps"] <= load_config()["agent_initial_state"]["max_step_budget"]
