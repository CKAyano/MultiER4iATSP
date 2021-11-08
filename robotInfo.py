from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from enum import Enum, auto
import yaml


@dataclass
class Coord:
    xx: float
    yy: float
    zz: float

    def coordToNp(self):
        return np.array([self.xx, self.yy, self.zz])

    def __add__(self, other):
        if isinstance(other, Coord):
            new_xx = self.xx + other.xx
            new_yy = self.yy + other.yy
            new_zz = self.zz + other.zz
            return Coord(new_xx, new_yy, new_zz)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Coord):
            new_xx = self.xx - other.xx
            new_yy = self.yy - other.yy
            new_zz = self.zz - other.zz
            return Coord(new_xx, new_yy, new_zz)
        return NotImplemented


@dataclass
class Coord_all:
    v1: Coord
    v2: Coord
    v3: Coord
    v4: Coord
    v5: Coord


class Position(Enum):
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


@dataclass
class Robot:
    id: int
    position: Position
    robot_path: np.ndarray = None
    point_index: np.ndarray = None

    def __post_init__(self):
        self.delimiter = -self.id

    def __len__(self):
        return len(self.robot_path)


class Config:
    def __init__(self, config_path) -> None:
        with open(config_path, "r") as config_file:
            config = yaml.load(config_file)
        self.robots_count = config["robots_count"]
        self.points_range = config["points_range"]
        self.link_width = config["link_width"]
        self.org_pos = np.radians(np.array(config["org_pos"]))
        self.joints_range = np.radians(np.array(config["joints_range"]))
        self.direct_array = np.array(config["direct_array"])
        self.baseX_offset = sum(self.points_range[0])
        self.baseY_offset = (self.points_range[1][1] - self.points_range[1][0]) / 2
