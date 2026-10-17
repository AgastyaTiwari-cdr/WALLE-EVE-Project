from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    MOVE_NORTH = "MOVE_NORTH"
    MOVE_SOUTH = "MOVE_SOUTH"
    MOVE_EAST = "MOVE_EAST"
    MOVE_WEST = "MOVE_WEST"
    SCAN = "SCAN"
    RECHARGE = "RECHARGE"


@dataclass(frozen=True)
class Coordinate:
    x: int
    y: int

    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Observation:
    agent_pos: Coordinate
    battery_level: int
    raw_signal_strength: float
    detected_clue: dict[str, Any] | None = None
    signal_direction: tuple[int, int] | None = None
    timestamp: int = 0
    measurement_sigma: float | None = None


@dataclass
class ActionTrace:
    step: int
    action: str
    agent_pos: tuple[int, int]
    battery_remaining: int
    signal_strength: float
    belief_entropy: float
    inferred_location: tuple[int, int]
    posterior_confidence: float
    reasoning: str
    timestamp: int = 0
    signal_direction: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
