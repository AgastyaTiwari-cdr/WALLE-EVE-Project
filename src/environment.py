"""Hidden-state simulation environment; observations never reveal EVE's position."""
import math
import random
from typing import Any

from src.schema import ActionType, Coordinate, Observation


class SimulatedWastelandEnvironment:
    def __init__(self, config: dict[str, Any]):
        env_cfg = config["environment"]
        agent_cfg = config["agent_initial_state"]
        self.grid_width, self.grid_height = map(int, env_cfg["grid_shape"])
        if self.grid_width < 1 or self.grid_height < 1:
            raise ValueError("grid_shape dimensions must be positive")
        self.true_eve_pos = Coordinate(*map(int, env_cfg["true_eve_position"]))
        self._validate_position(self.true_eve_pos.to_tuple(), "true_eve_position")
        self.signal_decay = float(env_cfg["signal_decay_factor"])
        self.sensor_noise = float(env_cfg["sensor_noise_sigma"])
        if self.signal_decay <= 0 or self.sensor_noise <= 0:
            raise ValueError("signal decay and sensor noise must be positive")
        sensor_cfg = env_cfg.get("sensor_model", {})
        self.dropout_probability = float(sensor_cfg.get("dropout_probability", 0.0))
        self.outlier_probability = float(sensor_cfg.get("outlier_probability", 0.0))
        self.outlier_scale = float(sensor_cfg.get("outlier_scale", 3.0))
        self.sensor_bias = float(sensor_cfg.get("bias", 0.0))
        for name, value in (("dropout_probability", self.dropout_probability),
                            ("outlier_probability", self.outlier_probability)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"sensor_model.{name} must be between 0 and 1")
        if self.outlier_scale < 1.0 or not math.isfinite(self.sensor_bias):
            raise ValueError("outlier_scale must be >= 1 and bias must be finite")

        self.agent_pos = Coordinate(*map(int, agent_cfg["start_position"]))
        self._validate_position(self.agent_pos.to_tuple(), "start_position")
        self.battery = int(agent_cfg["battery_capacity"])
        self.max_battery = self.battery
        self.step_budget = int(agent_cfg["max_step_budget"])
        if self.battery <= 0 or self.step_budget <= 0:
            raise ValueError("battery_capacity and max_step_budget must be positive")
        self.costs = {key: int(value) for key, value in config["action_costs"].items()}
        for key in ("MOVE", "SCAN", "RECHARGE"):
            if key not in self.costs:
                raise ValueError(f"Missing action_costs.{key}")
        if self.costs["MOVE"] < 0 or self.costs["SCAN"] < 0 or self.costs["RECHARGE"] <= 0:
            raise ValueError("MOVE/SCAN costs must be >= 0 and RECHARGE must be a positive amount")

        self.decoys: list[dict[str, Any]] = []
        for decoy in config.get("decoys", env_cfg.get("decoys", [])):
            pos = tuple(map(int, decoy["position"]))
            self._validate_position(pos, "decoy position")
            strength = float(decoy.get("signal_decay_factor", self.signal_decay * 0.65))
            if not math.isfinite(strength) or strength <= 0:
                raise ValueError("decoy signal_decay_factor must be positive")
            self.decoys.append({"position": pos, "signal_decay_factor": strength,
                                "label": str(decoy.get("label", "SPOOFED_SIGNAL"))})
        self.clues = {}
        for clue in config.get("clues", []):
            pos = tuple(map(int, clue["position"]))
            self._validate_position(pos, "clue position")
            self.clues[pos] = dict(clue)
        self.discovered_clues: list[dict[str, Any]] = []
        self.current_step = 0
        self.rng = random.Random(config.get("random_seed", 7))
        self.termination_reason: str | None = None
        initial = config.get("initial_signal_reading")
        if initial is not None:
            initial_pos = tuple(map(int, initial["position"]))
            self._validate_position(initial_pos, "initial_signal_reading.position")
            if initial_pos != self.agent_pos.to_tuple():
                raise ValueError("initial_signal_reading.position must match agent start_position")
            strength = float(initial["strength"])
            timestamp = int(initial["timestamp"])
            direction = initial.get("direction")
            if not math.isfinite(strength) or strength < 0 or timestamp < 0:
                raise ValueError("initial signal strength and timestamp must be non-negative and valid")
            if direction is not None and (not isinstance(direction, (list, tuple)) or len(direction) != 2):
                raise ValueError("initial_signal_reading.direction must be null or a pair [dx, dy]")
            self.initial_signal_reading = {
                "position": initial_pos, "strength": strength, "timestamp": timestamp,
                "direction": tuple(direction) if direction is not None else None,
            }
        else:
            self.initial_signal_reading = None
        self._initial_reading_pending = self.initial_signal_reading is not None
        self._scan_pending = False

    def _validate_position(self, pos: tuple[int, int], label: str) -> None:
        if len(pos) != 2 or not (0 <= pos[0] < self.grid_width and 0 <= pos[1] < self.grid_height):
            raise ValueError(f"{label} {pos} is outside the grid")

    def _expected_signal(self, pos: Coordinate) -> float:
        """True source plus optional spoofed sources; source identities stay hidden."""
        distance = math.hypot(pos.x - self.true_eve_pos.x, pos.y - self.true_eve_pos.y)
        expected = self.signal_decay / ((distance + 1.0) ** 2)
        for decoy in self.decoys:
            dx = pos.x - decoy["position"][0]
            dy = pos.y - decoy["position"][1]
            decoy_distance = math.hypot(dx, dy)
            expected += decoy["signal_decay_factor"] / ((decoy_distance + 1.0) ** 2)
        return expected

    def _compute_signal(self, pos: Coordinate, sigma: float | None = None) -> float:
        sigma = self.sensor_noise if sigma is None else sigma
        # Dropout is represented as a zero/near-zero reading; outliers model interference.
        if self.rng.random() < self.dropout_probability:
            return 0.0
        expected = self._expected_signal(pos) + self.sensor_bias
        noise = self.rng.gauss(0, sigma)
        if self.rng.random() < self.outlier_probability:
            noise += self.rng.gauss(0, sigma * self.outlier_scale)
        return max(0.0, round(expected + noise, 3))

    def get_observation(self) -> Observation:
        clue = self.clues.get(self.agent_pos.to_tuple())
        if clue is not None and clue not in self.discovered_clues:
            self.discovered_clues.append(dict(clue))
        if self._initial_reading_pending:
            reading = self.initial_signal_reading
            self._initial_reading_pending = False
            return Observation(self.agent_pos, self.battery, reading["strength"], clue,
                               reading["direction"], reading["timestamp"])
        sigma = self.sensor_noise * 0.35 if self._scan_pending else self.sensor_noise
        self._scan_pending = False
        measured = self._compute_signal(self.agent_pos, sigma)
        return Observation(self.agent_pos, self.battery, measured, clue, None, self.current_step, sigma)

    def step(self, action: ActionType) -> Observation:
        if self.is_terminal():
            raise RuntimeError("Cannot step a terminal mission")
        self.current_step += 1
        x, y = self.agent_pos.to_tuple()
        candidate = {
            ActionType.MOVE_NORTH: (x, y + 1),
            ActionType.MOVE_SOUTH: (x, y - 1),
            ActionType.MOVE_EAST: (x + 1, y),
            ActionType.MOVE_WEST: (x - 1, y),
        }
        if action in candidate:
            nx, ny = candidate[action]
            # Illegal boundary moves are rejected without moving; they still consume a step.
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                self.agent_pos = Coordinate(nx, ny)
                self.battery -= self.costs["MOVE"]
        elif action == ActionType.SCAN:
            self.battery -= self.costs["SCAN"]
            self._scan_pending = True
        elif action == ActionType.RECHARGE:
            self.battery = min(self.max_battery, self.battery + self.costs["RECHARGE"])
        else:
            raise ValueError(f"Unsupported action: {action}")
        self.battery = max(0, self.battery)
        if self.agent_pos == self.true_eve_pos:
            self.termination_reason = "EVE_FOUND"
        elif self.battery <= 0:
            self.termination_reason = "BATTERY_DEPLETED"
        elif self.current_step >= self.step_budget:
            self.termination_reason = "STEP_BUDGET_EXHAUSTED"
        return self.get_observation()

    def is_terminal(self) -> bool:
        return (self.agent_pos == self.true_eve_pos or self.battery <= 0
                or self.current_step >= self.step_budget)
