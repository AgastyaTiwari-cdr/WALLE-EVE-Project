"""Information-seeking, battery-aware policy for the hidden-grid search task.

This is a practical heuristic planner, not an exact POMDP solver. It trades off
posterior probability, expected measurement informativeness, exploration, and travel.
"""
from collections import Counter
import math
from typing import Tuple

import numpy as np

from src.belief_filter import GridBayesianBeliefFilter
from src.schema import ActionType, Observation


class AutonomousWallEAgent:
    def __init__(self, grid_shape: Tuple[int, int], signal_decay: float, noise_sigma: float,
                 move_cost: int = 4, scan_cost: int = 8, recharge_amount: int = 25,
                 max_battery: int = 100):
        self.filter = GridBayesianBeliefFilter(grid_shape, signal_decay, noise_sigma)
        self.grid_shape = tuple(grid_shape)
        self.move_cost, self.scan_cost = move_cost, scan_cost
        self.recharge_amount, self.max_battery = recharge_amount, max_battery
        self.critical_battery_threshold = max(move_cost * 2 + 1, scan_cost + move_cost)
        self.visits: Counter[tuple[int, int]] = Counter()
        self.last_observation_key = None
        self.processed_clues: set[tuple[int, int]] = set()
        self.scanned_clues: set[tuple[int, int]] = set()
        self.last_action = None

    def _update_belief_once(self, obs: Observation) -> None:
        pos = obs.agent_pos.to_tuple()
        key = (pos, obs.raw_signal_strength, repr(obs.detected_clue), obs.battery_level,
               obs.signal_direction, obs.timestamp, obs.measurement_sigma)
        if key == self.last_observation_key:
            return
        clue = None
        if obs.detected_clue and pos not in self.processed_clues:
            clue = obs.detected_clue
            self.processed_clues.add(pos)
        elif obs.signal_direction is not None:
            clue = {"directional_bias": obs.signal_direction}
        self.filter.update(pos, obs.raw_signal_strength, clue,
                           sigma=obs.measurement_sigma)
        self.last_observation_key = key

    def _destination_scores(self, current: tuple[int, int]) -> list[tuple[float, tuple[int, int], str]]:
        """Rank cells by target probability + expected information + exploration value."""
        belief = self.filter.belief
        height, width = belief.shape
        # Information proxy is evaluated separately for each candidate sensing cell.
        # If hypotheses predict very different readings there, the next observation
        # should separate those hypotheses more effectively.
        scores = []
        for y in range(height):
            for x in range(width):
                cell = (x, y)
                distance = abs(x - current[0]) + abs(y - current[1])
                probability = float(belief[y, x])
                novelty = 1.0 / (1.0 + self.visits[cell])
                # Also prefer cells near high-probability hypotheses: their observations
                # can discriminate between nearby candidate locations.
                local_mass = 0.0
                for yy in range(max(0, y - 1), min(height, y + 2)):
                    for xx in range(max(0, x - 1), min(width, x + 2)):
                        local_mass += float(belief[yy, xx])
                predicted = np.empty_like(belief)
                for hy in range(height):
                    for hx in range(width):
                        d = math.hypot(hx - x, hy - y)
                        predicted[hy, hx] = self.filter.signal_decay / ((d + 1.0) ** 2)
                predicted_mean = float(np.sum(belief * predicted))
                predicted_variance = float(np.sum(belief * (predicted - predicted_mean) ** 2))
                information = min(1.0, predicted_variance / (self.filter.sigma ** 2 + predicted_variance + 1e-12))
                info_value = information * local_mass
                score = 5.0 * probability + 0.9 * novelty + 1.4 * info_value - 0.035 * distance
                scores.append((score, cell, f"P(EVE at cell)={probability:.1%}, novelty={novelty:.2f}, information_proxy={info_value:.3f}, travel={distance}"))
        scores.sort(key=lambda item: item[0], reverse=True)
        return scores

    @staticmethod
    def _next_step(current: tuple[int, int], target: tuple[int, int]) -> ActionType | None:
        x, y = current
        tx, ty = target
        if x < tx: return ActionType.MOVE_EAST
        if x > tx: return ActionType.MOVE_WEST
        if y < ty: return ActionType.MOVE_NORTH
        if y > ty: return ActionType.MOVE_SOUTH
        return None

    def select_action(self, obs: Observation) -> tuple[ActionType, str]:
        pos = obs.agent_pos.to_tuple()
        self._update_belief_once(obs)
        self.visits[pos] += 1

        if obs.battery_level <= self.critical_battery_threshold and obs.battery_level < self.max_battery:
            return ActionType.RECHARGE, f"Reserve policy: battery={obs.battery_level}; recharge before committing to another route."

        # A clue is useful evidence itself; scan it once for a lower-noise measurement.
        if obs.detected_clue and pos not in self.scanned_clues and obs.battery_level >= self.scan_cost + self.move_cost:
            self.scanned_clues.add(pos)
            return ActionType.SCAN, f"Evidence collection: scanning newly discovered {obs.detected_clue.get('type', 'clue')} at {pos} for a higher-quality reading."

        ranked = self._destination_scores(pos)
        # Start with a short local exploration phase. This prevents the agent from
        # immediately following a noisy early MAP estimate and never encountering clues.
        # Exploration ends as soon as evidence is found or enough distinct cells were sampled.
        distinct_cells = len(self.visits)
        max_probability = self.filter.get_map_estimate()[1]
        if not self.processed_clues and distinct_cells < min(22, self.grid_shape[0] * self.grid_shape[1]) and max_probability < 0.60:
            width, height = self.grid_shape
            unvisited = [(x, y) for y in range(height) for x in range(width)
                         if self.visits[(x, y)] == 0 and (x, y) != pos]
            if unvisited:
                cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
                target = min(unvisited, key=lambda cell: (
                    abs(cell[0] - pos[0]) + abs(cell[1] - pos[1]),
                    abs(cell[0] - cx) + abs(cell[1] - cy),
                    -float(self.filter.belief[cell[1], cell[0]])
                ))
                action = self._next_step(pos, target)
                if action is not None:
                    return action, f"Exploration phase: sample nearby unvisited cell toward the search area centre ({target}) before committing to a noisy early estimate."
        # Prefer any cell with high combined score, but avoid wasting a step on the current cell.
        target_item = next((item for item in ranked if item[1] != pos), None)
        if target_item is None:
            return ActionType.SCAN, "Single-cell environment: take a final measurement."
        _, target, rationale = target_item
        action = self._next_step(pos, target)
        if action is None:
            return ActionType.SCAN, "At selected information-gathering cell; scan to refine the posterior."
        return action, f"Information-seeking route to {target}: {rationale}; adapt target after the next observation."
