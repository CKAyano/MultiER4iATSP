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
        self.adj_chromo = config["adj_chromo"]
        self.replace_chromo = config["replace_chromo"][0]
        self.replace_chromo_dist = config["replace_chromo"][1]
        self.custom_initChrom = config["custom_initChrom"]
        # self.mode = config["mode"]
        self.robots_count = config["robots_count"]
        self.points_range = config["points_range"]
        self.link_width = config["link_width"]
        self.org_pos = np.radians(np.array(config["org_pos"]))
        self.joints_range = np.radians(np.array(config["joints_range"]))
        self.direct_array = np.array(config["direct_array"])
        self.baseX_offset = sum(self.points_range[0])
        self.baseY_offset = (self.points_range[1][1] - self.points_range[1][0]) / 2 + self.points_range[0][0]

    #     if self.mode == 2:
    #         self.baseY_offset = sum(self.points_range[1])
    #         self.toe_in_angle = np.arctan2(self.baseY_offset, self.baseX_offset)
    #         self.h01_rbs = self.get_h01_rbs()

    # @staticmethod
    # def mat_rotz(theta):
    #     mat = np.array(
    #         [
    #             [np.cos(theta), -np.sin(theta), 0, 0],
    #             [np.sin(theta), np.cos(theta), 0, 0],
    #             [0, 0, 1, 0],
    #             [0, 0, 0, 1],
    #         ]
    #     )
    #     return mat

    # @staticmethod
    # def mat_transl(transl_list: list):
    #     mat = np.array(
    #         [[1, 0, 0, transl_list[0]], [0, 1, 0, transl_list[1]], [0, 0, 1, transl_list[2]], [0, 0, 0, 1]]
    #     )
    #     return mat

    # def get_h01_rbs(self) -> Tuple[List, List, List]:
    #     h01_rbs = []
    #     inv_h01_rbs = []
    #     rot_mat = []

    #     rot_mat_1 = self.mat_rotz(self.toe_in_angle)
    #     h01_1 = rot_mat_1
    #     inv_h01_1 = np.linalg.inv(h01_1)
    #     h01_rbs.append(h01_1)
    #     inv_h01_rbs.append(inv_h01_1)
    #     rot_mat.append(rot_mat_1)

    #     rot_mat_2 = self.mat_rotz(self.toe_in_angle + np.pi)
    #     h01_2 = self.mat_transl([self.baseX_offset, self.baseY_offset, 0]).dot(rot_mat_2)
    #     inv_h01_2 = np.linalg.inv(h01_2)
    #     h01_rbs.append(h01_2)
    #     inv_h01_rbs.append(inv_h01_2)
    #     rot_mat.append(rot_mat_2)

    #     if self.robots_count > 2:
    #         rot_mat_3 = self.mat_rotz(-self.toe_in_angle)
    #         h01_3 = self.mat_transl([0, self.baseY_offset, 0]).dot(rot_mat_3)
    #         inv_h01_3 = np.linalg.inv(h01_3)
    #         h01_rbs.append(h01_3)
    #         inv_h01_rbs.append(inv_h01_3)
    #         rot_mat.append(rot_mat_3)

    #         rot_mat_4 = self.mat_rotz(np.pi - self.toe_in_angle)
    #         h01_4 = self.mat_transl([self.baseX_offset, 0, 0]).dot(rot_mat_4)
    #         inv_h01_4 = np.linalg.inv(h01_4)
    #         h01_rbs.append(h01_4)
    #         inv_h01_rbs.append(inv_h01_4)
    #         rot_mat.append(rot_mat_4)

    #     return h01_rbs, inv_h01_rbs, rot_mat
