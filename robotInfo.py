from dataclasses import dataclass
from typing import Optional
import numpy as np
from robotCalc_pygeos import Coord
from enum import Enum, auto


class Position(Enum):
    LEFT: auto()
    RIGHT: auto()
    UP: auto()
    DOWN: auto()


@dataclass
class Robot:
    id: int
    position: Position
    path: Optional[np.ndarray] = None
    joints: Optional[np.ndarray] = None
    coord: Optional[Coord] = None
    interp: Optional[np.ndarray] = None

    def __post_init__(self):
        self.delimiter = -self.id
