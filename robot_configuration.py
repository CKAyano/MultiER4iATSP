from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple
import numpy as np
from enum import Enum, auto
import yaml
from dataclasses import dataclass
from abc import ABC, abstractmethod


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
    v1: Optional[Coord] = None
    v2: Optional[Coord] = None
    v3: Optional[Coord] = None
    v4: Optional[Coord] = None
    v5: Optional[Coord] = None
    v6: Optional[Coord] = None


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

        self.robot_name = config["robot_name"]

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
        if "direct_array" in config:
            self.direct_array = np.array(config["direct_array"])
        if "zyx_euler" in config:
            self.zyx_euler = np.array(config["zyx_euler"])
        self.baseX_offset = sum(self.points_range[0])
        self.baseY_offset = (self.points_range[1][1] - self.points_range[1][0]) / 2 + self.points_range[0][0]


class RobotKinematics(ABC):
    @abstractmethod
    def forward_kines(self, q: np.ndarray) -> Coord_all:
        pass

    @abstractmethod
    def inverse_kines(self, vv: Coord, zyx_euler: np.ndarray) -> np.ndarray:
        pass


class FanucKinematics(RobotKinematics):
    d_1 = 0
    a_2 = 260
    a_3 = 20
    d_4 = 290

    def forward_kines(self, q: np.ndarray) -> Coord_all:
        try:
            if q.ndim > 1 and q.shape[0] > 1:
                raise RuntimeError("theta can only import one set")
        except RuntimeError as e:
            print(repr(e))
            raise
        if q.ndim > 1:
            q = np.squeeze(q)
        c1 = np.cos(q[0])
        s1 = np.sin(q[0])
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        c3 = np.cos(q[2])
        s3 = np.sin(q[2])
        c4 = np.cos(0)
        s4 = np.sin(0)

        axisM1 = np.array([[c1, 0, -s1, 0], [s1, 0, c1, 0], [0, -1, 0, self.d_1], [0, 0, 0, 1]])

        axisM2 = np.array(
            [[s2, c2, 0, self.a_2 * s2], [-c2, s2, 0, -self.a_2 * c2], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        axisM3 = np.array(
            [[c3, 0, -s3, self.a_3 * c3], [s3, 0, c3, self.a_3 * s3], [0, -1, 0, 0], [0, 0, 0, 1]]
        )

        axisM4 = np.array([[c4, 0, s4, 0], [s4, 0, -c4, 0], [0, 1, 0, self.d_4], [0, 0, 0, 1]])

        fk1 = axisM1
        fk2 = fk1.dot(axisM2)
        fk3 = fk2.dot(axisM3)
        fk4 = fk3.dot(axisM4)

        v1 = Coord(0, 0, 0)
        v2 = Coord(fk1[0, 3], fk1[1, 3], fk1[2, 3])
        v3 = Coord(fk2[0, 3], fk2[1, 3], fk2[2, 3])
        v4 = Coord(fk3[0, 3], fk3[1, 3], fk3[2, 3])
        v5 = Coord(fk4[0, 3], fk4[1, 3], fk4[2, 3])

        v_all = Coord_all(v1, v2, v3, v4, v5)

        return v_all

    def inverse_kines(self, vv: Coord, zyx_euler: np.ndarray) -> np.ndarray:
        if zyx_euler.shape == (3, 3):
            joint_three2six = self._joint_three2six_from_third_joint
        else:
            joint_three2six = self._joint_three2six_from_first_joint

        px = vv.xx
        py = vv.yy
        pz = vv.zz

        q1 = np.zeros(2)
        q2 = np.zeros(4)
        q3 = np.zeros(2)
        q23 = np.zeros(4)
        numq2 = -1

        q1[0] = np.arctan2(py, px) - np.arctan2(0, np.sqrt(np.square(px) + np.square(py)))
        q1[1] = np.arctan2(py, px) - np.arctan2(0, -np.sqrt(np.square(px) + np.square(py)))

        k = (
            np.square(self.d_1)
            - 2 * self.d_1 * pz
            + np.square(px)
            + np.square(py)
            + np.square(pz)
            - np.square(self.a_3)
            - np.square(self.d_4)
            - np.square(self.a_2)
        ) / (2 * self.a_2)

        q3[0] = np.arctan2(self.a_3, self.d_4) - np.arctan2(
            k, np.sqrt(np.square(self.a_3) + np.square(self.d_4) - np.square(k))
        )
        q3[1] = np.arctan2(self.a_3, self.d_4) - np.arctan2(
            k, -np.sqrt(np.square(self.a_3) + np.square(self.d_4) - np.square(k))
        )

        for jj in range(2):
            for ii in range(2):
                numq2 = numq2 + 1
                q23[numq2] = np.arctan2(
                    (
                        self.d_1 * self.d_4
                        - self.d_4 * pz
                        + self.a_3 * px * np.cos(q1[jj])
                        - self.a_2 * self.d_1 * np.sin(q3[ii])
                        + self.a_3 * py * np.sin(q1[jj])
                        + self.a_2 * pz * np.sin(q3[ii])
                        + self.a_2 * px * np.cos(q1[jj]) * np.cos(q3[ii])
                        + self.a_2 * py * np.cos(q3[ii]) * np.sin(q1[jj])
                    ),
                    -(
                        self.a_3 * self.d_1
                        - self.a_3 * pz
                        - self.a_2 * pz * np.cos(q3[ii])
                        - self.d_4 * px * np.cos(q1[jj])
                        - self.d_4 * py * np.sin(q1[jj])
                        + self.a_2 * self.d_1 * np.cos(q3[ii])
                        + self.a_2 * py * np.sin(q1[jj]) * np.sin(q3[ii])
                        + self.a_2 * px * np.cos(q1[jj]) * np.sin(q3[ii])
                    ),
                )
                q2[numq2] = q23[numq2] - q3[ii]

        q1 = angleAdj(q1)
        q2 = angleAdj(q2)
        q3 = angleAdj(q3)

        q1 = np.array([q1[0], q1[0], q1[1], q1[1]])
        q3 = np.array([q3[0], q3[1], q3[0], q3[1]])

        group_1 = np.hstack((q1[0], q2[0], q3[0]))
        group_2 = np.hstack((q1[1], q2[1], q3[1]))
        group_3 = np.hstack((q1[2], q2[2], q3[2]))
        group_4 = np.hstack((q1[3], q2[3], q3[3]))

        q = np.vstack((group_1, group_2, group_3, group_4))
        q_all = joint_three2six(q, zyx_euler)
        return q_all

    def _joint_three2six_from_third_joint(self, q_array: np.ndarray, zyx_euler) -> np.ndarray:
        q_all = np.zeros((0, 6))
        len_q = q_array.shape[0]
        for i in range(len_q):
            q1 = q_array[i, 0]
            q2 = q_array[i, 1]
            q3 = q_array[i, 2]
            R1 = np.array([[1, 0, 0], [0, np.cos(q1), -np.sin(q1)], [0, np.sin(q1), np.cos(q1)]])
            R2 = np.array([[np.cos(-q2), 0, np.sin(-q2)], [0, 1, 0], [-np.sin(-q2), 0, np.cos(-q2)]])
            R3 = np.array([[np.cos(-q3), 0, np.sin(-q3)], [0, 1, 0], [-np.sin(-q3), 0, np.cos(-q3)]])
            R1_3 = R1.dot(R2).dot(R3)
            Rd = np.linalg.inv(R1_3).dot(zyx_euler)
            q5 = [np.arccos(Rd[2, 2]), -np.arccos(Rd[2, 2])]
            q4 = [
                np.arctan2(Rd[1, 2] / np.sin(q5[1]), Rd[0, 2] / np.sin(q5[1])),
                np.arctan2(Rd[1, 2] / np.sin(q5[0]), Rd[0, 2] / np.sin(q5[0])),
            ]
            q6 = [
                np.arctan2(Rd[2, 1] / np.sin(q5[1]), Rd[2, 0] / np.sin(q5[1])),
                np.arctan2(Rd[2, 1] / np.sin(q5[0]), Rd[2, 0] / np.sin(q5[0])),
            ]
            q = np.array([[q1, q2, q3, q4[0], q5[0], q6[0]], [q1, q2, q3, q4[1], q5[1], q6[1]]])
            q_all = np.vstack((q_all, q))
        return q_all

    def _joint_three2six_from_first_joint(self, q_array: np.ndarray, zyx_euler) -> np.ndarray:
        end_mat = EulerAngle.zyx2trans(zyx_euler[0], zyx_euler[1], zyx_euler[2])
        q_all = np.zeros((0, 6))
        # len_q = q_array.shape[0]
        for q in q_array:
            c1 = np.cos(q[0])
            s1 = np.sin(q[0])
            c23 = np.cos(q[1] + q[2])
            s23 = np.sin(q[1] + q[2])
            r_ax = end_mat[0, 2]
            r_ay = end_mat[1, 2]
            r_az = end_mat[2, 2]
            m5 = r_ax * c1 * c23 + r_ay * s1 * c23 - r_az * s23
            q5 = [np.arccos(m5), -np.arccos(m5)]

            m4_u = r_ax * s1 - r_ay * c1
            m4_d = r_ax * c1 * s23 + r_ay * s1 * s23 + r_az * c23
            q4 = [
                np.arctan2(m4_u / np.sin(q5[1]), m4_d / np.sin(q5[1])),
                np.arctan2(m4_u / np.sin(q5[0]), m4_d / np.sin(q5[0])),
            ]

            r_ox = end_mat[0, 1]
            r_oy = end_mat[1, 1]
            r_oz = end_mat[2, 1]
            r_nx = end_mat[0, 0]
            r_ny = end_mat[1, 0]
            r_nz = end_mat[2, 0]
            m6_u = -r_ox * c1 * c23 - r_oy * s1 * c23 + r_oz * s23
            m6_d = r_nx * c1 * c23 + r_ny * s1 * c23 - r_nz * s23
            q6 = [
                np.arctan2(m6_u / np.sin(q5[1]), m6_d / np.sin(q5[1])),
                np.arctan2(m6_u / np.sin(q5[0]), m6_d / np.sin(q5[0])),
            ]
            q = np.array([[q[0], q[1], q[2], q4[0], q5[0], q6[0]], [q[0], q[1], q[2], q4[1], q5[1], q6[1]]])
            q_all = np.vstack((q_all, q))
        return q_all


class Coord_trans:
    @staticmethod
    def mat_rotx(alpha):
        mat = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha), 0],
                [0, np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 0, 1],
            ]
        )
        return mat

    @staticmethod
    def mat_roty(beta):
        mat = np.array(
            [
                [np.cos(beta), 0, np.sin(beta), 0],
                [0, 1, 0, 0],
                [-np.sin(beta), 0, np.cos(beta), 0],
                [0, 0, 0, 1],
            ]
        )
        return mat

    @staticmethod
    def mat_rotz(theta):
        mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return mat

    @staticmethod
    def mat_transl(transl_list: list):
        mat = np.array(
            [[1, 0, 0, transl_list[0]], [0, 1, 0, transl_list[1]], [0, 0, 1, transl_list[2]], [0, 0, 0, 1]]
        )
        return mat


class EulerAngle:
    @staticmethod
    def trans2zyx(trans) -> List:
        if np.sqrt(trans[0, 0] ** 2 + trans[1, 0] ** 2) == 0 and -trans[2, 0] == 1:
            b = np.pi / 2
            a = 0
            r = np.atan2(trans[0, 1], trans[1, 1])
        elif np.sqrt(trans[0, 0] ** 2 + trans[1, 0] ** 2) == 0 and -trans[2, 0] == -1:
            b = -np.pi / 2
            a = 0
            r = -np.atan2(trans[0, 1], trans[1, 1])
        else:
            b = np.atan2(-trans[2, 0], np.sqrt(trans[0, 0] ** 2 + trans[1, 0] ** 2))
            cb = np.cos(b)
            a = np.atan2(trans[1, 0] / cb, trans[0, 0] / cb)
            r = np.atan2(trans[2, 1] / cb, trans[2, 2] / cb)
        return [a, b, r]

    @staticmethod
    def zyx2trans(alpha, beta, gamma) -> np.ndarray:
        ct = Coord_trans
        return ct.mat_rotz(alpha) @ ct.mat_roty(beta) @ ct.mat_rotx(gamma)


def angleAdj(ax):
    for ii in range(len(ax)):
        while ax[ii] > np.pi:
            ax[ii] = ax[ii] - np.pi * 2

        while ax[ii] < -np.pi:
            ax[ii] = ax[ii] + np.pi * 2
    return ax
