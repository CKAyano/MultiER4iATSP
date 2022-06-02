from dataclasses import dataclass
from typing import Optional, List, Tuple
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

    def distance(self, other):
        if isinstance(other, Coord):
            np_self = self.coordToNp()
            np_other = other.coordToNp()
            return np.sqrt(np.sum(np.square(np_self - np_other)))
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
            config = yaml.load(config_file, Loader=yaml.FullLoader)

        if "interp_mode" in config:
            self.interp_mode = config["interp_mode"]
        else:
            self.interp_mode = "max_deg"

        if "interp_step_period" in config:
            self.interp_step_period = config["interp_step_period"]
        elif "interp_step_freq" in config:
            self.interp_step_period = config["interp_step_freq"]

        if "mean_motion_velocity_deg" in config:
            self.mean_motion_velocity_rad = np.radians(config["mean_motion_velocity_deg"])

        self.custom_initChrom = config["custom_initChrom"]

        if "is_recombine_init_chromo" in config:
            self.is_recombine_init_chromo = config["is_recombine_init_chromo"]
        elif "adj_chromo" in config:
            self.is_recombine_init_chromo = config["adj_chromo"]

        if "replace_chromo" in config:
            self.is_hamming_crowding = config["replace_chromo"][0]
            self.hamming_crowding_dist = config["replace_chromo"][1]
            self.hamming_crowding_mode = config["replace_chromo"][2]
        elif "is_hamming_crowding" in config:
            self.is_hamming_crowding = config["is_hamming_crowding"][0]
            self.hamming_crowding_dist = config["is_hamming_crowding"][1]
            self.hamming_crowding_mode = config["is_hamming_crowding"][2]

        self.robots_count = config["robots_count"]
        self.points_range = config["points_range"]
        self.link_width = config["link_width"]
        self.org_pos = np.radians(np.array(config["org_pos"]))
        self.joints_range = np.radians(np.array(config["joints_range"]))
        self.direct_array = np.array(config["direct_array"])
        self.baseX_offset = sum(self.points_range[0])
        self.baseY_offset = (self.points_range[1][1] - self.points_range[1][0]) / 2 + self.points_range[0][0]
