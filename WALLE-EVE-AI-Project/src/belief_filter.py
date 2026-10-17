"""Numerically stable grid-based Bayesian state estimator."""
import math
from typing import Any

import numpy as np


class GridBayesianBeliefFilter:
    def __init__(self, grid_shape: tuple[int, int], signal_decay: float, noise_sigma: float):
        width, height = grid_shape
        if width < 1 or height < 1:
            raise ValueError("grid_shape dimensions must be positive")
        if signal_decay <= 0:
            raise ValueError("signal_decay_factor must be positive")
        if noise_sigma <= 0:
            raise ValueError("sensor_noise_sigma must be positive")
        self.width, self.height = int(width), int(height)
        self.signal_decay = float(signal_decay)
        self.sigma = float(noise_sigma)
        self.belief = np.full((self.height, self.width), 1.0 / (self.width * self.height), dtype=float)

    def update(self, agent_pos: tuple[int, int], measured_signal: float,
               clue: dict[str, Any] | None = None, sigma: float | None = None) -> None:
        """Apply a measurement likelihood and any clue-direction evidence."""
        ax, ay = agent_pos
        if not (0 <= ax < self.width and 0 <= ay < self.height):
            raise ValueError("agent_pos is outside the configured grid")
        if not math.isfinite(measured_signal) or measured_signal < 0:
            raise ValueError("measured_signal must be finite and non-negative")

        # A scan may provide a lower-noise reading than passive sensing.
        measurement_sigma = self.sigma if sigma is None else float(sigma)
        if not math.isfinite(measurement_sigma) or measurement_sigma <= 0:
            raise ValueError("measurement sigma must be finite and positive")
        # Log likelihood avoids underflow when sensor noise is small.
        log_likelihood = np.empty_like(self.belief)
        for y in range(self.height):
            for x in range(self.width):
                distance = math.hypot(x - ax, y - ay)
                expected = self.signal_decay / ((distance + 1.0) ** 2)
                z = (measured_signal - expected) / measurement_sigma
                log_likelihood[y, x] = -0.5 * z * z

        if clue and "directional_bias" in clue:
            bias = clue["directional_bias"]
            if not isinstance(bias, (list, tuple)) or len(bias) != 2:
                raise ValueError("directional_bias must be a pair [dx, dy]")
            dx, dy = bias
            for y in range(self.height):
                for x in range(self.width):
                    # A direction clue is soft evidence, not a hard exclusion.
                    if dx > 0:
                        log_likelihood[y, x] += math.log(2.0) if x > ax else (math.log(0.1) if x < ax else 0.0)
                    elif dx < 0:
                        log_likelihood[y, x] += math.log(2.0) if x < ax else (math.log(0.1) if x > ax else 0.0)
                    if dy > 0:
                        log_likelihood[y, x] += math.log(2.0) if y > ay else (math.log(0.1) if y < ay else 0.0)
                    elif dy < 0:
                        log_likelihood[y, x] += math.log(2.0) if y < ay else (math.log(0.1) if y > ay else 0.0)

        log_prior = np.log(np.maximum(self.belief, np.finfo(float).tiny))
        log_posterior = log_prior + log_likelihood
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        total = float(posterior.sum())
        if not math.isfinite(total) or total <= 0:
            # Defensive fallback; should not occur with log-space normalization.
            self.belief.fill(1.0 / self.belief.size)
        else:
            self.belief = posterior / total

    def get_map_estimate(self) -> tuple[tuple[int, int], float]:
        iy, ix = np.unravel_index(int(np.argmax(self.belief)), self.belief.shape)
        return (int(ix), int(iy)), float(self.belief[iy, ix])

    def ranked_locations(self) -> list[tuple[tuple[int, int], float]]:
        flat_indices = np.argsort(self.belief.ravel())[::-1]
        return [((int(i % self.width), int(i // self.width)), float(self.belief.ravel()[i]))
                for i in flat_indices]

    def compute_entropy(self) -> float:
        p = self.belief[self.belief > 0]
        return float(-np.sum(p * np.log2(p)))
