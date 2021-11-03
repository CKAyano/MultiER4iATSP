import numpy as np
import pygeos.creation as pgc
import pygeos.set_operations as pgi
import warnings
from dataclasses import dataclass
from robotInfo import Position, Robot, Coord, Coord_all

d_1 = 0
a_2 = 260
a_3 = 20
d_4 = 290


class RobotCalc_pygeos:
    def __init__(self, baseX_offset: float, baseY_offset: float, linkWidth: float):
        self.baseX_offset = baseX_offset
        self.baseY_offset = baseY_offset
        self.linkWidth = linkWidth

    def userIK(self, vv: Coord, direct_array: np.ndarray):
        def angleAdj(ax):
            for ii in range(len(ax)):
                while ax[ii] > np.pi:
                    ax[ii] = ax[ii] - np.pi * 2

                while ax[ii] < -np.pi:
                    ax[ii] = ax[ii] + np.pi * 2
            return ax

        px = vv.xx
        py = vv.yy
        pz = vv.zz

        q1 = np.zeros(2)
        q2 = np.zeros(4)
        q3 = np.zeros(2)
        q23 = np.zeros(4)
        numq2 = -1

        q1[0] = np.arctan2(py, px) - np.arctan2(
            0, np.sqrt(np.square(px) + np.square(py))
        )
        q1[1] = np.arctan2(py, px) - np.arctan2(
            0, -np.sqrt(np.square(px) + np.square(py))
        )

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

        q3[0] = np.arctan2(a_3, d_4) - np.arctan2(
            k, np.sqrt(np.square(a_3) + np.square(d_4) - np.square(k))
        )
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
        q_all = self.joint_three2six(q, direct_array)
        return q_all

    def joint_three2six(self, q_array: np.ndarray, direct_array: np.ndarray):
        q_all = np.zeros((0, 6))
        len_q = q_array.shape[0]
        for i in range(len_q):
            q1 = q_array[i, 0]
            q2 = q_array[i, 1]
            q3 = q_array[i, 2]
            R1 = np.array(
                [[1, 0, 0], [0, np.cos(q1), -np.sin(q1)], [0, np.sin(q1), np.cos(q1)]]
            )
            R2 = np.array(
                [
                    [np.cos(-q2), 0, np.sin(-q2)],
                    [0, 1, 0],
                    [-np.sin(-q2), 0, np.cos(-q2)],
                ]
            )
            R3 = np.array(
                [
                    [np.cos(-q3), 0, np.sin(-q3)],
                    [0, 1, 0],
                    [-np.sin(-q3), 0, np.cos(-q3)],
                ]
            )
            R1_3 = R1.dot(R2).dot(R3)
            Rd = np.linalg.inv(R1_3).dot(direct_array)
            q5 = [np.arccos(Rd[2, 2]), -np.arccos(Rd[2, 2])]
            q4 = [
                np.arctan2(Rd[1, 2] / np.sin(q5[1]), Rd[0, 2] / np.sin(q5[1])),
                np.arctan2(Rd[1, 2] / np.sin(q5[0]), Rd[0, 2] / np.sin(q5[0])),
            ]
            q6 = [
                np.arctan2(Rd[2, 1] / np.sin(q5[1]), Rd[2, 0] / np.sin(q5[1])),
                np.arctan2(Rd[2, 1] / np.sin(q5[0]), Rd[2, 0] / np.sin(q5[0])),
            ]
            q = np.array(
                [[q1, q2, q3, q4[0], q5[0], q6[0]], [q1, q2, q3, q4[1], q5[1], q6[1]]]
            )
            q_all = np.vstack((q_all, q))
        return q_all

    def greedySearch(self, q_f_best: np.ndarray, q_array: np.ndarray):
        diff_q1 = np.absolute(q_array[:, 0] - q_f_best[0])
        diff_q2 = np.absolute(q_array[:, 1] - q_f_best[1])
        diff_q3 = np.absolute(q_array[:, 2] - q_f_best[2])
        diff_q4 = np.absolute(q_array[:, 3] - q_f_best[3])
        diff_q5 = np.absolute(q_array[:, 4] - q_f_best[4])
        diff_q6 = np.absolute(q_array[:, 5] - q_f_best[5])
        jointDiff = np.vstack((diff_q1, diff_q2, diff_q3, diff_q4, diff_q5, diff_q6))
        jointMax_index = np.argmax(jointDiff, axis=0)
        numOfPoint = q_array.shape[0]
        for ii in range(numOfPoint):
            if ii == 0:
                element_jointMax = jointDiff[jointMax_index[ii], ii]
            else:
                element_jointMax = np.hstack(
                    (element_jointMax, jointDiff[jointMax_index[ii], ii])
                )
        bestGroup_index = np.argmin(element_jointMax, axis=0)
        q_best = q_array[bestGroup_index, :]

        return q_best

    def userFK(self, q: np.ndarray):
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

        axisM1 = np.array(
            [[c1, 0, -s1, 0], [s1, 0, c1, 0], [0, -1, 0, d_1], [0, 0, 0, 1]]
        )

        axisM2 = np.array(
            [[s2, c2, 0, a_2 * s2], [-c2, s2, 0, -a_2 * c2], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        axisM3 = np.array(
            [[c3, 0, -s3, a_3 * c3], [s3, 0, c3, a_3 * s3], [0, -1, 0, 0], [0, 0, 0, 1]]
        )

        axisM4 = np.array(
            [[c4, 0, s4, 0], [s4, 0, -c4, 0], [0, 1, 0, d_4], [0, 0, 0, 1]]
        )

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

    def robot2world(self, vv_a: Coord, position: Position):
        vv_b = Coord(0, 0, 0)
        if position == Position.LEFT:
            pass
        elif position == Position.RIGHT:
            vv_b.xx = -vv_a.xx + self.baseX_offset
            vv_b.yy = -vv_a.yy
            vv_b.zz = vv_a.zz
        elif position == Position.UP:
            vv_b.xx = self.baseY_offset - vv_a.yy
            vv_b.yy = vv_a.xx - self.baseX_offset / 2
            vv_b.zz = vv_a.zz
        elif position == Position.DOWN:
            vv_b.xx = self.baseY_offset + vv_a.yy
            vv_b.yy = self.baseX_offset / 2 - vv_a.xx
            vv_b.zz = vv_a.zz
        return vv_b

    def robot2world_v_all(self, v_all: Coord_all, position: Position):

        v1 = self.robot2world(v_all.v1, position)
        v2 = self.robot2world(v_all.v2, position)
        v3 = self.robot2world(v_all.v3, position)
        v4 = self.robot2world(v_all.v4, position)
        v5 = self.robot2world(v_all.v5, position)

        v_all = Coord_all(v1, v2, v3, v4, v5)

        return v_all

    def cvAxisRange(self, q_best: np.ndarray, axisRange: np.ndarray):
        if q_best.ndim == 1:
            q_best = q_best[None, :]
        axisRange = np.radians(axisRange)
        numOfAxis = axisRange.shape[0]
        isQNan = np.isnan(q_best)
        if np.any(isQNan):
            return True
        else:
            for i in range(numOfAxis):
                condition_ql = q_best[:, i] < axisRange[i, 0]
                condition_qu = q_best[:, i] > axisRange[i, 1]
                if np.any(condition_ql) or np.any(condition_qu):
                    return True
        return False

    def cvCollision(self, robot_a: Robot, robot_b: Robot):

        # % ------------------------- qa ------------------------- % #
        point = self.generateLinkWidenPoint(robot_a)

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
            [
                ring1,
                ring2,
                ring3,
                ring4,
                ring5,
                ring6,
                ring7,
                ring8,
                ring9,
                ring10,
                ring11,
                ring12,
            ]
        )

        # % ------------------------- qb ------------------------- % #
        vb_all = self.userFK(robot_b.joints_best)
        vb_all = self.robot2world_v_all(vb_all, robot_b.position)

        gmSegB1 = pgc.linestrings(
            [
                [vb_all.v2.xx, vb_all.v2.yy, vb_all.v2.zz],
                [vb_all.v4.xx, vb_all.v4.yy, vb_all.v4.zz],
            ]
        )
        gmSegB2 = pgc.linestrings(
            [
                [vb_all.v4.xx, vb_all.v4.yy, vb_all.v4.zz],
                [vb_all.v5.xx, vb_all.v5.yy, vb_all.v5.zz],
            ]
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

            if np.all(strInter1 == "LINESTRING EMPTY") and np.all(
                strInter2 == "LINESTRING EMPTY"
            ):
                return False
            else:
                return True

    def generateLinkWidenPoint(self, robot: Robot):
        def calcNormalVec(v_f: Coord, v_e: Coord):
            v1_point = v_f.coordToNp()
            v2_point = v_e.coordToNp()
            if (
                robot.joints_best[0] < np.pi / 2 + 0.0001
                and robot.joints_best[0] > np.pi / 2 - 0.0001
            ):
                normed_normalVec_1 = np.array([1, 0, 0])
                normed_normalVec_2 = np.array([0, 1, 0])
            elif (
                robot.joints_best[0] < -np.pi / 2 + 0.0001
                and robot.joints_best[0] > -np.pi / 2 - 0.0001
            ):
                normed_normalVec_1 = np.array([1, 0, 0])
                normed_normalVec_2 = np.array([0, -1, 0])
            else:
                vector_1 = np.array([np.tan(robot.joints_best[0]), -1, 0])
                vector_2 = v2_point - v1_point
                vector_3 = np.array([1, np.tan(robot.joints_best[0]), 0])
                vecCross_1 = np.cross(vector_1, vector_2)
                length_vec_1 = np.linalg.norm(vecCross_1)
                normed_normalVec_1 = vecCross_1 / length_vec_1
                vecCross_2 = np.cross(vector_3, vector_2)
                length_vec_2 = np.linalg.norm(vecCross_2)
                normed_normalVec_2 = vecCross_2 / length_vec_2
            return v1_point, v2_point, normed_normalVec_1, normed_normalVec_2

        v_all = self.userFK(robot.joints_best)
        v_all = self.robot2world_v_all(v_all, robot.position)
        v2_point, v4_point, normalVec_1, normalVec_2 = calcNormalVec(v_all.v2, v_all.v4)
        _, v5_point, normalVec_3, normalVec_4 = calcNormalVec(v_all.v4, v_all.v5)
        p1 = v2_point + (-normalVec_1 + normalVec_2) * self.linkWidth / 2
        p2 = v2_point + (normalVec_1 + normalVec_2) * self.linkWidth / 2
        p3 = v2_point + (normalVec_1 - normalVec_2) * self.linkWidth / 2
        p4 = v2_point + (-normalVec_1 - normalVec_2) * self.linkWidth / 2
        p5 = v4_point + (-normalVec_1 + normalVec_2) * self.linkWidth / 2
        p6 = v4_point + (normalVec_1 + normalVec_2) * self.linkWidth / 2
        p7 = v4_point + (normalVec_1 - normalVec_2) * self.linkWidth / 2
        p8 = v4_point + (-normalVec_1 - normalVec_2) * self.linkWidth / 2

        p9 = v4_point + (-normalVec_3 + normalVec_4) * self.linkWidth / 2
        p10 = v4_point + (normalVec_3 + normalVec_4) * self.linkWidth / 2
        p11 = v4_point + (normalVec_3 - normalVec_4) * self.linkWidth / 2
        p12 = v4_point + (-normalVec_3 - normalVec_4) * self.linkWidth / 2
        p13 = v5_point + (-normalVec_3 + normalVec_4) * self.linkWidth / 2
        p14 = v5_point + (normalVec_3 + normalVec_4) * self.linkWidth / 2
        p15 = v5_point + (normalVec_3 - normalVec_4) * self.linkWidth / 2
        p16 = v5_point + (-normalVec_3 - normalVec_4) * self.linkWidth / 2
        point = (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16)
        return point


if __name__ == "__main__":
    qa = np.array([0.09499919, 0.59284477, -0.26421251])
    qb = np.array([0.27674711, 0.64568055, -0.3508449])
    test = RobotCalc_pygeos(750, 200)
    print(test.cvCollision(qa, qb))
    vv_a = Coord(150, 0, 100)
    # vv_b = robot2world(vv_a)
    direct_array = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    axisRange = np.array(
        [[-170, 170], [-110, 120], [-204, 69], [-190, 190], [-120, 120], [-360, 360]]
    )
    q = test.userIK(vv_a, direct_array)
    print(q)
    for i in range(q.shape[0]):
        isOutOfRange = test.cvAxisRange(q[i, :], axisRange)
        print(isOutOfRange)
