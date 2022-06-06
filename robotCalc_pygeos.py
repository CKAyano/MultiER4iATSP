import numpy as np
import pygeos.creation as pgc
import pygeos.set_operations as pgi
import warnings
from dataclasses import dataclass
from robotInfo import Position, Robot, Coord, Coord_all, Config

d_1 = 0
a_2 = 260
a_3 = 20
d_4 = 290


class RobotCalc_pygeos:
    def __init__(self, config: Config) -> None:
        self.config = config

    def angleAdj(self, ax):
        for ii in range(len(ax)):
            while ax[ii] > np.pi:
                ax[ii] = ax[ii] - np.pi * 2

            while ax[ii] < -np.pi:
                ax[ii] = ax[ii] + np.pi * 2
        return ax

    def userIK(self, vv: Coord) -> np.ndarray:

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
            np.square(d_1)
            - 2 * d_1 * pz
            + np.square(px)
            + np.square(py)
            + np.square(pz)
            - np.square(a_3)
            - np.square(d_4)
            - np.square(a_2)
        ) / (2 * a_2)

        q3[0] = np.arctan2(a_3, d_4) - np.arctan2(k, np.sqrt(np.square(a_3) + np.square(d_4) - np.square(k)))
        q3[1] = np.arctan2(a_3, d_4) - np.arctan2(
            k, -np.sqrt(np.square(a_3) + np.square(d_4) - np.square(k))
        )

        for jj in range(2):
            for ii in range(2):
                numq2 = numq2 + 1
                q23[numq2] = np.arctan2(
                    (
                        d_1 * d_4
                        - d_4 * pz
                        + a_3 * px * np.cos(q1[jj])
                        - a_2 * d_1 * np.sin(q3[ii])
                        + a_3 * py * np.sin(q1[jj])
                        + a_2 * pz * np.sin(q3[ii])
                        + a_2 * px * np.cos(q1[jj]) * np.cos(q3[ii])
                        + a_2 * py * np.cos(q3[ii]) * np.sin(q1[jj])
                    ),
                    -(
                        a_3 * d_1
                        - a_3 * pz
                        - a_2 * pz * np.cos(q3[ii])
                        - d_4 * px * np.cos(q1[jj])
                        - d_4 * py * np.sin(q1[jj])
                        + a_2 * d_1 * np.cos(q3[ii])
                        + a_2 * py * np.sin(q1[jj]) * np.sin(q3[ii])
                        + a_2 * px * np.cos(q1[jj]) * np.sin(q3[ii])
                    ),
                )
                q2[numq2] = q23[numq2] - q3[ii]

        q1 = self.angleAdj(q1)
        q2 = self.angleAdj(q2)
        q3 = self.angleAdj(q3)

        q1 = np.array([q1[0], q1[0], q1[1], q1[1]])
        q3 = np.array([q3[0], q3[1], q3[0], q3[1]])

        group_1 = np.hstack((q1[0], q2[0], q3[0]))
        group_2 = np.hstack((q1[1], q2[1], q3[1]))
        group_3 = np.hstack((q1[2], q2[2], q3[2]))
        group_4 = np.hstack((q1[3], q2[3], q3[3]))

        q = np.vstack((group_1, group_2, group_3, group_4))
        q_all = self.joint_three2six(q)
        return q_all

    def joint_three2six(self, q_array: np.ndarray) -> np.ndarray:
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
            Rd = np.linalg.inv(R1_3).dot(self.config.direct_array)
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

    def greedy_search(self, q_f_best: np.ndarray, q_array: np.ndarray) -> np.ndarray:
        diff_q1 = np.absolute(q_array[:, 0] - q_f_best[0])
        diff_q2 = np.absolute(q_array[:, 1] - q_f_best[1])
        diff_q3 = np.absolute(q_array[:, 2] - q_f_best[2])
        diff_q4 = np.absolute(q_array[:, 3] - q_f_best[3])
        diff_q5 = np.absolute(q_array[:, 4] - q_f_best[4])
        diff_q6 = np.absolute(q_array[:, 5] - q_f_best[5])
        jointDiff = np.vstack((diff_q1, diff_q2, diff_q3, diff_q4, diff_q5, diff_q6))
        jointMax_index = np.argmax(jointDiff, axis=0)  # 每個逆向解最大軸是第幾軸
        numOfPoint = q_array.shape[0]
        for ii in range(numOfPoint):
            if ii == 0:
                element_jointMax = jointDiff[jointMax_index[ii], ii]
            else:
                element_jointMax = np.hstack((element_jointMax, jointDiff[jointMax_index[ii], ii]))
        bestGroup_index = np.argmin(element_jointMax, axis=0)
        q_best = q_array[bestGroup_index, :]

        return q_best

    def userFK(self, q: np.ndarray) -> Coord_all:
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

        axisM1 = np.array([[c1, 0, -s1, 0], [s1, 0, c1, 0], [0, -1, 0, d_1], [0, 0, 0, 1]])

        axisM2 = np.array([[s2, c2, 0, a_2 * s2], [-c2, s2, 0, -a_2 * c2], [0, 0, 1, 0], [0, 0, 0, 1]])

        axisM3 = np.array([[c3, 0, -s3, a_3 * c3], [s3, 0, c3, a_3 * s3], [0, -1, 0, 0], [0, 0, 0, 1]])

        axisM4 = np.array([[c4, 0, s4, 0], [s4, 0, -c4, 0], [0, 1, 0, d_4], [0, 0, 0, 1]])

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

    def robot2world(self, vv_a: Coord, position: Position) -> Coord:
        vv_b = Coord(0, 0, 0)

        # if self.config.mode == 1:
        if position == Position.LEFT:
            vv_b = vv_a
        elif position == Position.RIGHT:
            vv_b.xx = -vv_a.xx + self.config.baseX_offset
            vv_b.yy = -vv_a.yy
            vv_b.zz = vv_a.zz
        elif position == Position.UP:
            vv_b.xx = self.config.baseY_offset - vv_a.yy
            vv_b.yy = vv_a.xx - self.config.baseX_offset / 2
            vv_b.zz = vv_a.zz
        elif position == Position.DOWN:
            vv_b.xx = self.config.baseY_offset + vv_a.yy
            vv_b.yy = self.config.baseX_offset / 2 - vv_a.xx
            vv_b.zz = vv_a.zz
        return vv_b

        # if self.config.mode == 2:
        #     vv_a_list = [vv_a.xx, vv_a.yy, vv_a.zz]
        #     if position == Position.LEFT:
        #         i = 0
        #     elif position == Position.RIGHT:
        #         i = 1
        #     elif position == Position.UP:
        #         i = 2
        #     elif position == Position.DOWN:
        #         i = 3
        #     h_02 = self.config.mat_transl(vv_a_list).dot(self.config.h01_rbs[2][i])
        #     h_12 = self.config.h01_rbs[1][i].dot(h_02)
        #     vv_b.xx = h_12[0, 3]
        #     vv_b.yy = h_12[1, 3]
        #     vv_b.zz = h_12[2, 3]
        #     return vv_b
        # raise RuntimeError("wrong config mode")

    def robot2world_v_all(self, v_all: Coord_all, position: Position) -> Coord_all:

        v1 = self.robot2world(v_all.v1, position)
        v2 = self.robot2world(v_all.v2, position)
        v3 = self.robot2world(v_all.v3, position)
        v4 = self.robot2world(v_all.v4, position)
        v5 = self.robot2world(v_all.v5, position)

        v_all = Coord_all(v1, v2, v3, v4, v5)

        return v_all

    def coord2bestAngle(self, vv: Coord, q_1_best, robot: Robot) -> np.ndarray:
        vv = self.robot2world(vv, robot.position)
        q_2 = self.userIK(vv)
        len_q = q_2.shape[0]
        idx_notOutRange = np.zeros((0))
        for i in range(len_q):
            q_2_test = q_2[i, :]
            q_2_test = q_2_test[None, :]
            q_2_outRange = self.cv_joints_range(q_2_test)
            if not q_2_outRange:
                idx_notOutRange = np.hstack((idx_notOutRange, i))

        idx_notOutRange = idx_notOutRange.astype(int)
        q_2_output = q_2.copy()
        q_2_output = q_2_output[idx_notOutRange, :]
        if idx_notOutRange.shape[0] == 1:
            if q_2_output.ndim == 2:
                q_2_output = np.squeeze(q_2_output)
            return q_2_output
        elif idx_notOutRange.shape[0] == 0:
            q_2_output = np.repeat(np.nan, q_2.shape[1])
            return q_2_output
        else:
            q_2_best = self.greedy_search(q_1_best, q_2_output)
            # q_2_best_DG = np.degrees(q_2_best)
        return q_2_best

    def cv_joints_range(self, q_best: np.ndarray) -> bool:
        q_best_cp = q_best.copy()
        if q_best_cp.ndim == 1:
            q_best_cp = q_best_cp[None, :]
        joints_range = self.config.joints_range
        numOfAxis = joints_range.shape[0]
        isQNan = np.isnan(q_best_cp)
        if np.any(isQNan):
            return True
        else:
            q_best_cp[:, 2] = -(q_best_cp[:, 2] + q_best_cp[:, 1])
            for i in range(q_best_cp.shape[0]):
                q_best_cp[i, :] = self.angleAdj(q_best_cp[i, :])
            for i in range(numOfAxis):
                condition_ql = q_best_cp[:, i] < joints_range[i, 0]
                condition_qu = q_best_cp[:, i] > joints_range[i, 1]
                if np.any(condition_ql) or np.any(condition_qu):
                    which_L = q_best_cp[condition_ql, :]
                    which_L_DG = np.degrees(which_L)
                    which_U = q_best_cp[condition_qu, :]
                    which_U_DG = np.degrees(which_U)
                    return True
        return False

    def cv_collision(self, q_a, q_b, robot_a: Robot, robot_b: Robot) -> bool:

        # % ------------------------- qa ------------------------- % #
        point = self.get_link_points(q_a, robot_a)

        ring1 = pgc.linearrings([point[0], point[1], point[2], point[3]])
        ring2 = pgc.linearrings([point[0], point[1], point[5], point[4]])
        ring3 = pgc.linearrings([point[3], point[2], point[6], point[7]])
        ring4 = pgc.linearrings([point[1], point[2], point[6], point[5]])
        ring5 = pgc.linearrings([point[0], point[3], point[7], point[4]])
        ring6 = pgc.linearrings([point[4], point[5], point[6], point[7]])
        ring7 = pgc.linearrings([point[8], point[9], point[10], point[11]])
        ring8 = pgc.linearrings([point[8], point[9], point[13], point[12]])
        ring9 = pgc.linearrings([point[11], point[10], point[14], point[15]])
        ring10 = pgc.linearrings([point[9], point[10], point[14], point[13]])
        ring11 = pgc.linearrings([point[8], point[11], point[15], point[12]])
        ring12 = pgc.linearrings([point[12], point[13], point[14], point[15]])

        cpg_all = pgc.polygons(
            [ring1, ring2, ring3, ring4, ring5, ring6, ring7, ring8, ring9, ring10, ring11, ring12]
        )

        # % ------------------------- qb ------------------------- % #
        vb_all = self.userFK(q_b)
        vb_all = self.robot2world_v_all(vb_all, robot_b.position)

        gmSegB1 = pgc.linestrings(
            [[vb_all.v2.xx, vb_all.v2.yy, vb_all.v2.zz], [vb_all.v4.xx, vb_all.v4.yy, vb_all.v4.zz]]
        )
        gmSegB2 = pgc.linestrings(
            [[vb_all.v4.xx, vb_all.v4.yy, vb_all.v4.zz], [vb_all.v5.xx, vb_all.v5.yy, vb_all.v5.zz]]
        )

        # % -------------------- intersection -------------------- % #
        warnings.simplefilter("error")
        try:
            inter1 = pgi.intersection(gmSegB1, cpg_all)
            inter2 = pgi.intersection(gmSegB2, cpg_all)
        except RuntimeWarning:
            warnings.simplefilter("ignore")
            return True
        else:
            warnings.simplefilter("ignore")
            strInter1 = inter1.astype(str)
            strInter2 = inter2.astype(str)

            if np.all(strInter1 == "LINESTRING EMPTY") and np.all(strInter2 == "LINESTRING EMPTY"):
                return False
            else:
                return True

    def get_link_points(self, q, robot: Robot) -> tuple:
        def get_normal_vec(v_f: Coord, v_e: Coord):
            v1_point = v_f.coordToNp()
            v2_point = v_e.coordToNp()
            if q[0] < np.pi / 2 + 0.0001 and q[0] > np.pi / 2 - 0.0001:
                normed_normalVec_1 = np.array([1, 0, 0])
                normed_normalVec_2 = np.array([0, 1, 0])
            elif q[0] < -np.pi / 2 + 0.0001 and q[0] > -np.pi / 2 - 0.0001:
                normed_normalVec_1 = np.array([1, 0, 0])
                normed_normalVec_2 = np.array([0, -1, 0])
            else:
                vector_1 = np.array([np.tan(q[0]), -1, 0])
                vector_2 = v2_point - v1_point
                vector_3 = np.array([1, np.tan(q[0]), 0])
                vecCross_1 = np.cross(vector_1, vector_2)
                length_vec_1 = np.linalg.norm(vecCross_1)
                normed_normalVec_1 = vecCross_1 / length_vec_1
                vecCross_2 = np.cross(vector_3, vector_2)
                length_vec_2 = np.linalg.norm(vecCross_2)
                normed_normalVec_2 = vecCross_2 / length_vec_2
                if length_vec_1 == 0:
                    normed_normalVec_1 = np.array([1, 0, 0])
                if length_vec_2 == 0:
                    normed_normalVec_2 = np.array([0, 1, 0])
            return v1_point, v2_point, normed_normalVec_1, normed_normalVec_2

        v_all = self.userFK(q)
        v_all = self.robot2world_v_all(v_all, robot.position)
        v2_point, v4_point, normalVec_1, normalVec_2 = get_normal_vec(v_all.v2, v_all.v4)
        _, v5_point, normalVec_3, normalVec_4 = get_normal_vec(v_all.v4, v_all.v5)
        p1 = v2_point + (-normalVec_1 + normalVec_2) * self.config.link_width / 2
        p2 = v2_point + (normalVec_1 + normalVec_2) * self.config.link_width / 2
        p3 = v2_point + (normalVec_1 - normalVec_2) * self.config.link_width / 2
        p4 = v2_point + (-normalVec_1 - normalVec_2) * self.config.link_width / 2
        p5 = v4_point + (-normalVec_1 + normalVec_2) * self.config.link_width / 2
        p6 = v4_point + (normalVec_1 + normalVec_2) * self.config.link_width / 2
        p7 = v4_point + (normalVec_1 - normalVec_2) * self.config.link_width / 2
        p8 = v4_point + (-normalVec_1 - normalVec_2) * self.config.link_width / 2

        p9 = v4_point + (-normalVec_3 + normalVec_4) * self.config.link_width / 2
        p10 = v4_point + (normalVec_3 + normalVec_4) * self.config.link_width / 2
        p11 = v4_point + (normalVec_3 - normalVec_4) * self.config.link_width / 2
        p12 = v4_point + (-normalVec_3 - normalVec_4) * self.config.link_width / 2
        p13 = v5_point + (-normalVec_3 + normalVec_4) * self.config.link_width / 2
        p14 = v5_point + (normalVec_3 + normalVec_4) * self.config.link_width / 2
        p15 = v5_point + (normalVec_3 - normalVec_4) * self.config.link_width / 2
        p16 = v5_point + (-normalVec_3 - normalVec_4) * self.config.link_width / 2
        points = (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16)
        return points
