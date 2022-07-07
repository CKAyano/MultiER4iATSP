import numpy as np
import pygeos.creation as pgc
import pygeos.set_operations as pgi
import warnings
from robot_configuration import Position, Robot, Coord, Coord_all, Config, angleAdj
from robot_configuration import FanucKinematics, PumaKinematics


class RobotCalc_pygeos:
    def __init__(self, config: Config) -> None:
        self.config = config
        if "zyx_euler" in self.config.__dict__ and "direct_array" not in self.config.__dict__:
            self.direction = self.config.zyx_euler
        elif "direct_array" in self.config.__dict__ and "zyx_euler" not in self.config.__dict__:
            # !! 此方向表示方法以第三軸座標方向當基準，不適合用於所有機械手臂
            # !! 只用於舊FANUC CONFIG.yml
            self.direction = self.config.direct_array
            print("-------- config: direct_array --------")
        else:
            raise RuntimeError('choose either "direct_array" or "zyx_euler" in config file')

        if config.robot_name == "fanuc":
            self.robot_kine = FanucKinematics()
        if config.robot_name == "puma":
            self.robot_kine = PumaKinematics()

        test_links = np.array(self.robot_kine.collision_links, dtype=str)
        test_links_width = self.robot_kine.links_width
        if test_links.ndim == 1:
            test_links = test_links[None, :]
            test_links_width = test_links_width[None, :]
        if test_links.shape[1] != 2:
            raise TypeError("col of argument should be two")
        if len(test_links_width) != test_links.shape[0]:
            raise TypeError("number of collision_links and links_width is not matched")

    def userFK(self, q: np.ndarray) -> Coord_all:
        return self.robot_kine.forward_kines(q)

    def userIK(self, vv: Coord) -> np.ndarray:
        return self.robot_kine.inverse_kines(vv, self.direction)

    @staticmethod
    def greedy_search(q_f_best: np.ndarray, q_array: np.ndarray) -> np.ndarray:
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

    def robot2world_v_all(self, v_all: Coord_all, position: Position) -> Coord_all:
        for k, v in v_all.__dict__.items():
            if v is None:
                break
            new_v = self.robot2world(v, position)
            v_all.set_new_value_by_str_of_key(k, new_v)
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
        # joints_range = self.config.joints_range
        joints_range = np.radians(np.array(self.robot_kine.joints_range_deg))
        joints_count = joints_range.shape[0]
        if q_best_cp.shape[1] != joints_count:
            raise TypeError("length of q_best is incorrect")
        isQNan = np.isnan(q_best_cp)
        if np.any(isQNan):
            return True
        else:
            q_best_cp_2 = q_best_cp.copy()
            q_best_cp_2 = self.robot_kine.range_joints_relative_to_absolute(q_best_cp_2)
            if q_best_cp_2 is not None:
                q_best_cp = q_best_cp_2
            # q_best_cp[:, 2] = -(q_best_cp[:, 2] + q_best_cp[:, 1])
            for i in range(q_best_cp.shape[0]):
                q_best_cp[i, :] = angleAdj(q_best_cp[i, :])
            for i in range(joints_count):
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
        polygons = self.get_robot_polygons(q_a, robot_a)

        # % ------------------------- qb ------------------------- % #
        segments = self.get_robot_segments(q_b, robot_b)

        # % -------------------- intersection -------------------- % #
        warnings.simplefilter("error")
        try:
            inter_all = []
            for seg in segments:
                inter = pgi.intersection(seg, polygons)
                inter_all.append(inter)
        except RuntimeWarning:
            warnings.simplefilter("ignore")
            return True
        else:
            warnings.simplefilter("ignore")
            inter_all_str = [int.astype(str) for int in inter_all]
            for inter in inter_all_str:
                if np.any(inter != "LINESTRING EMPTY"):
                    return True
            return False

    def get_robot_polygons(self, q, robot: Robot) -> tuple:
        link_count = len(self.robot_kine.collision_links)
        v_all = self.userFK(q)
        v_all = self.robot2world_v_all(v_all, robot.position)
        rings_all = tuple()
        for i in range(link_count):
            v_f = v_all.__dict__.get(self.robot_kine.collision_links[i][0])
            v_e = v_all.__dict__.get(self.robot_kine.collision_links[i][1])
            if v_f is None or v_e is None:
                raise TypeError(
                    "setting wrong links for collision detection"
                    + "(change collision_links in robot_configuraion.py)"
                )
            points = self._get_link_points_by_joints_position(q, v_f, v_e, self.robot_kine.links_width[i])
            rings = self._get_link_rings(points)
            rings_all += rings
        cpg_all = pgc.polygons(list(rings_all))
        return cpg_all

    @staticmethod
    def _get_link_rings(points):
        ring1 = pgc.linearrings([points[0], points[1], points[2], points[3]])
        ring2 = pgc.linearrings([points[0], points[1], points[5], points[4]])
        ring3 = pgc.linearrings([points[3], points[2], points[6], points[7]])
        ring4 = pgc.linearrings([points[1], points[2], points[6], points[5]])
        ring5 = pgc.linearrings([points[0], points[3], points[7], points[4]])
        ring6 = pgc.linearrings([points[4], points[5], points[6], points[7]])
        return ring1, ring2, ring3, ring4, ring5, ring6

    @staticmethod
    def _get_normal_vec(q, v_f: Coord, v_e: Coord):
        v1_point = v_f.coordToNp()
        v2_point = v_e.coordToNp()
        # if q[0] < np.pi / 2 + 0.0001 and q[0] > np.pi / 2 - 0.0001:
        #     normed_normalVec_1 = np.array([1, 0, 0])
        #     normed_normalVec_2 = np.array([0, 1, 0])
        # elif q[0] < -np.pi / 2 + 0.0001 and q[0] > -np.pi / 2 - 0.0001:
        #     normed_normalVec_1 = np.array([1, 0, 0])
        #     normed_normalVec_2 = np.array([0, -1, 0])
        # else:
        # vector_1 = np.array([np.tan(q[0]), -1, 0])
        vector_2 = v2_point - v1_point
        vector_1 = np.array([vector_2[0], vector_2[1], v1_point[2]])
        # vector_3 = np.array([1, np.tan(q[0]), 0])
        # vector_3 = np.array([v1_point[0], vector_2[1], vector_2[2]])
        vecCross_1 = np.cross(vector_1, vector_2)
        length_vec_1 = np.linalg.norm(vecCross_1)
        normed_normalVec_1 = vecCross_1 / length_vec_1
        vecCross_2 = np.cross(vecCross_1, vector_2)
        length_vec_2 = np.linalg.norm(vecCross_2)
        normed_normalVec_2 = vecCross_2 / length_vec_2
        if length_vec_1 == 0:
            normed_normalVec_1 = np.array([1, 0, 0])
        if length_vec_2 == 0:
            normed_normalVec_2 = np.array([0, 1, 0])
        return v1_point, v2_point, normed_normalVec_1, normed_normalVec_2

    def _get_link_points_by_joints_position(self, q, v_f: Coord, v_e: Coord, link_width: float):
        vf_point, ve_point, normalVec_1, normalVec_2 = self._get_normal_vec(q, v_f, v_e)
        p1 = vf_point + (-normalVec_1 + normalVec_2) * link_width
        p2 = vf_point + (normalVec_1 + normalVec_2) * link_width
        p3 = vf_point + (normalVec_1 - normalVec_2) * link_width
        p4 = vf_point + (-normalVec_1 - normalVec_2) * link_width
        p5 = ve_point + (-normalVec_1 + normalVec_2) * link_width
        p6 = ve_point + (normalVec_1 + normalVec_2) * link_width
        p7 = ve_point + (normalVec_1 - normalVec_2) * link_width
        p8 = ve_point + (-normalVec_1 - normalVec_2) * link_width
        return p1, p2, p3, p4, p5, p6, p7, p8

    def get_robot_segments(self, q, robot):
        link_count = len(self.robot_kine.collision_links)
        v_all = self.userFK(q)
        v_all = self.robot2world_v_all(v_all, robot.position)
        segment_all = tuple()
        for i in range(link_count):
            v_f = v_all.__dict__.get(self.robot_kine.collision_links[i][0])
            v_e = v_all.__dict__.get(self.robot_kine.collision_links[i][1])
            segment = pgc.linestrings([[v_f.xx, v_f.yy, v_f.zz], [v_e.xx, v_e.yy, v_e.zz]])
            segment_all += (segment,)
        return segment_all
