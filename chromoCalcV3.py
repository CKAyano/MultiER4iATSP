import numpy as np
from robotCalc_pygeos import RobotCalc_pygeos, Coord
from robotInfo import Config, Robot, Position
from typing import List, Tuple, Optional
import functools

# from caching import np_cache


class ChromoCalcV3:
    def __init__(
        self, config: Config, points: np.ndarray, step: int, num_slicing: int, feasibleSol_list: List
    ) -> None:
        try:
            if step >= num_slicing:
                raise RuntimeError(f"'step'(={step}) should under {num_slicing}.")
        except RuntimeError as e:
            print(e)
            raise

        self.config = config
        self.rc = RobotCalc_pygeos(self.config)
        self.px = points[:, 0]
        self.py = points[:, 1]
        self.pz = points[:, 2]
        self.step = step
        self.num_slicing = num_slicing
        # if len(feasibleSol_list) == 0:
        self.feasibleSol_count = 0
        # else:
        #     self.feasibleSol_count = feasibleSol_list[-1]
        self.feasibleSol_list = feasibleSol_list
        self.robots: list[Robot] = []
        position = [Position.LEFT, Position.RIGHT, Position.UP, Position.DOWN]
        for i in range(self.config.robots_count):
            self.robots.append(Robot(i, position[i]))

    @staticmethod
    def _dominates(obj_self: np.ndarray, obj_other: np.ndarray) -> bool:
        better_at_least_one_obj = obj_self[0] < obj_other[0] or obj_self[1] < obj_other[1]
        better_or_equal_for_all_obj = obj_self[0] <= obj_other[0] and obj_self[1] <= obj_other[1]
        return better_or_equal_for_all_obj and better_at_least_one_obj

    def a_dominates_b(self, obj_a, obj_b):
        return self._dominates(obj_a, obj_b)

    # def __arreq_in_list(self, arr, list_arrays):
    #     return next((True for elem in list_arrays if np.array_equal(elem, arr)), False)

    def _fast_nondominated_sorting(self, pop):
        chromos = pop.Chrom.copy()
        for rb in range(self.config.robots_count - 1):
            chromos = np.where(chromos != self.robots[rb + 1].delimiter, chromos, self.robots[0].delimiter)
        obj_chromos = pop.ObjV.copy()

        fonts = []
        fonts.append([])
        dominated_sol = []
        n = np.zeros(chromos.shape[0])
        for i_self, self_chrom in enumerate(chromos):
            dominated_sol.append(set())
            for i_other, other_chrom in enumerate(chromos):
                if self._dominates(obj_chromos[i_self, :], obj_chromos[i_other, :]):
                    dominated_sol[i_self].add((i_other, tuple(other_chrom)))
                elif self._dominates(obj_chromos[i_other, :], obj_chromos[i_self, :]):
                    n[i_self] += 1
            if n[i_self] == 0:
                fonts[0].append((i_self, self_chrom))
        i = 0
        while len(fonts[i]) != 0:
            temp = []
            for i_self, self_chrom in fonts[i]:
                for i_other, other_chrom in dominated_sol[i_self]:
                    other_chrom = np.array(other_chrom)
                    n[i_other] -= 1
                    if n[i_other] == 0:
                        temp.append((i_other, other_chrom))
            i += 1
            fonts.append(temp)
        return fonts

    @staticmethod
    def _distance(self_chromo, other_chromo):
        dist = np.count_nonzero(self_chromo != other_chromo)
        return dist

    def _set_chromo(self, pop, idx_chromo_need_replace):
        insert_count = len(idx_chromo_need_replace)
        chromos = pop.Chrom[idx_chromo_need_replace]
        pop.Chrom = np.delete(pop.Chrom, idx_chromo_need_replace, 0)
        pop.Phen = np.delete(pop.Phen, idx_chromo_need_replace, 0)
        if self.config.replace_mode == "random":
            for _ in range(insert_count):
                need_append = np.arange(self.robots[-2].delimiter, self.px.shape[0] + 1)
                np.random.shuffle(need_append)
                pop.Chrom = np.vstack((pop.Chrom, need_append))
                pop.Phen = np.vstack((pop.Phen, need_append))
        elif self.config.replace_mode == "reverse":
            delim_list = [self.robots[rb].delimiter for rb in range(self.config.robots_count)]
            for i in range(len(idx_chromo_need_replace)):
                # chromo = pop.Chrom[idx_chromo]
                chromo = chromos[i]
                num = 0
                pre_i = 0
                need_append = np.zeros((0,))
                for i, gene in enumerate(chromo):
                    if gene in delim_list:
                        temp = chromo[pre_i:i]
                        pre_i = i + 1
                        temp = np.flip(temp)
                        need_append = np.hstack((need_append, temp, num))
                        num -= 1
                need_append = np.hstack((need_append, np.flip(chromo[pre_i:])))
                need_append = need_append.astype(int)
                pop.Chrom = np.vstack((pop.Chrom, need_append))
                pop.Phen = np.vstack((pop.Phen, need_append))

                # pop.Chrom = temp_chromo
                # pop.Phen = temp_chromo

    def replace_chromosome(self, pop, threshold):
        fonts = self._fast_nondominated_sorting(pop)
        n = []
        for f in fonts[0:-1]:
            i, chromo_1 = f[0]
            for i, chromo in f[1:]:
                dist = self._distance(chromo_1, chromo)
                if dist <= threshold:
                    n.append(i)
        if len(n) > 0:
            print()
        self._set_chromo(pop, n)
        # pop.Chrom = np.delete(pop.Chrom, n, 0)
        # pop.Phen = np.delete(pop.Phen, n, 0)
        # for _ in range(len(n)):
        #     need_append = np.arange(self.robots[-2].delimiter, self.px.shape[0] + 1)
        #     np.random.shuffle(need_append)
        #     pop.Chrom = np.vstack((pop.Chrom, need_append))

    def _need_preAdj(self, p_id: int, robot: Robot) -> bool:
        vv_robot = Coord(self.px[p_id], self.py[p_id], self.pz[p_id])
        vv_robot = self.rc.robot2world(vv_robot, robot.position)
        q_robot = self.rc.userIK(vv_robot)

        isOutRange = []
        for i in range(q_robot.shape[0]):
            isOutRange.append(self.rc.cv_joints_range(q_robot[i, :]))
        if np.all(isOutRange):
            return True
        return False

    def _get_robot_id(self, position: Position) -> int:
        which_rb = [_rb for _rb in range(self.config.robots_count) if self.robots[_rb].position == position]
        return which_rb[0]

    def _throw_path(self, org_pos: Position, dest_Pos: Position) -> None:
        org_rb = self._get_robot_id(org_pos)
        dest_rb = self._get_robot_id(dest_Pos)

        del_id_org = []
        append_path_dest = []
        for i, path in enumerate(self.robots[org_rb].robot_path):
            point_id = path - 1
            if self._need_preAdj(point_id, self.robots[org_rb]):
                del_id_org.append(i)
                append_path_dest.append(path)

        self.robots[org_rb].robot_path = np.delete(self.robots[org_rb].robot_path, del_id_org)
        if len(append_path_dest):
            self.robots[dest_rb].robot_path = np.hstack((self.robots[dest_rb].robot_path, append_path_dest))

    def adj_chromo(self, chromosome: np.ndarray, chromo_id: int, pop) -> None:

        self.set_robotsPath(chromosome)

        if self.config.robots_count == 2:
            self._throw_path(Position.LEFT, Position.RIGHT)
            self._throw_path(Position.RIGHT, Position.LEFT)

        if self.config.robots_count > 2:
            self._throw_path(Position.LEFT, Position.RIGHT)
            self._throw_path(Position.LEFT, Position.UP)
            self._throw_path(Position.LEFT, Position.DOWN)
            self._throw_path(Position.RIGHT, Position.LEFT)
            self._throw_path(Position.RIGHT, Position.UP)
            self._throw_path(Position.RIGHT, Position.DOWN)
            self._throw_path(Position.UP, Position.DOWN)
            self._throw_path(Position.UP, Position.RIGHT)
            self._throw_path(Position.UP, Position.LEFT)
            self._throw_path(Position.DOWN, Position.UP)
            self._throw_path(Position.DOWN, Position.LEFT)
            self._throw_path(Position.DOWN, Position.RIGHT)

        chromo = self.robots[0].robot_path
        for i in range(self.config.robots_count - 1):
            chromo = np.hstack((chromo, 0 - i, self.robots[i + 1].robot_path))
        pop.Phen[chromo_id, :] = chromo
        pop.Chrom[chromo_id, :] = chromo

    def set_robotsPath(self, chromosome: np.ndarray) -> None:
        if chromosome.ndim >= 2:
            chromosome = np.squeeze(chromosome)
        # chromosome = population[chromoIndex, :]  # [1, 2, 3, 4, 0, 5, 6]
        robotPath_left = chromosome.copy()

        for i in range(self.config.robots_count - 1):
            mask = np.isin(
                robotPath_left, [self.robots[rb].delimiter for rb in range(self.config.robots_count - 1)],
            )
            delimiter_index = np.where(mask)  # index of 0 = 4
            _index = delimiter_index[0][0]
            self.robots[i].robot_path = robotPath_left[:(_index)]  # [1, 2, 3, 4]
            self.robots[i].point_index = self.robots[i].robot_path - 1
            robotPath_left = robotPath_left[(_index + 1) :]  # [5, 6]
        self.robots[-1].robot_path = robotPath_left
        self.robots[-1].point_index = self.robots[-1].robot_path - 1

        max_len = max([len(self.robots[i]) for i in range(self.config.robots_count)])

        for i in range(self.config.robots_count):
            appendArray = np.ones(max_len - len(self.robots[i]), dtype="int32") * -1
            self.robots[i].point_index = np.hstack(
                (self.robots[i].point_index, appendArray, -1)
            )  # [4, 5, -1, -1]

    def _get_interp_oneSeq(
        self, checkPoints_count: int, q_1_best: np.ndarray, q_2_best: np.ndarray
    ) -> np.ndarray:

        if q_1_best.ndim == 1:
            int_q = np.expand_dims(q_1_best, axis=0)
        if checkPoints_count == 0:
            int_q = np.vstack((int_q, q_2_best))
        else:
            offset_q = (q_2_best - q_1_best) / checkPoints_count
            for i in range(checkPoints_count):
                int_q = np.vstack((int_q, int_q[i, :] + offset_q))
        int_q = np.delete(int_q, 0, axis=0)
        return int_q

    def _get_angle_offset(
        self, p_id_rbs: List[int], q_1_best_rbs: List[np.ndarray]
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:

        q_2_best_rbs = []
        angOffset_rbs = []
        for rb in range(self.config.robots_count):
            if p_id_rbs[rb] == -1:
                q_2_best_rbs.append(self.config.org_pos)
            else:
                vv_a = Coord(self.px[p_id_rbs[rb]], self.py[p_id_rbs[rb]], self.pz[p_id_rbs[rb]])
                q_2_best_rbs.append(self.rc.coord2bestAngle(vv_a, q_1_best_rbs[rb], self.robots[rb]))
                is_q_nan = np.isnan(q_2_best_rbs[rb])
                if np.any(is_q_nan):
                    return None
            angOffset_rbs.append(np.degrees(np.abs(q_2_best_rbs[rb] - q_1_best_rbs[rb])))

        return angOffset_rbs, q_2_best_rbs

    def _get_interp_robots(
        self, q_1_best_rbs: List[np.ndarray], p_id_rbs: List[int]
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:

        _angleOffset = self._get_angle_offset(p_id_rbs, q_1_best_rbs)
        if _angleOffset is None:
            return None
        angOffset_rbs, q_2_best_rbs = _angleOffset

        interp_q_rbs = []
        for rb in range(self.config.robots_count):
            checkPoints_count = int(np.max(angOffset_rbs[rb]))
            interp_q_rbs.append(
                self._get_interp_oneSeq(checkPoints_count, q_1_best_rbs[rb], q_2_best_rbs[rb])
            )

        return interp_q_rbs, q_2_best_rbs, angOffset_rbs

    def interpolation(self, chromosome: np.ndarray) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        self.set_robotsPath(chromosome)

        is_firstLoop = True
        totalAngle_rbs = 0
        joints_count = self.config.org_pos.size

        len_pointIndex = len(self.robots[0].point_index)
        totalInt_q_rbs = [np.zeros((0, joints_count))] * 4
        totalAngle_rbs = 0
        for i in range(len_pointIndex):
            if is_firstLoop:
                q_1_best = [self.config.org_pos] * self.config.robots_count
                is_firstLoop = False
            else:
                q_1_best = q_2_best

            p_id_rbs = [self.robots[rb].point_index[i] for rb in range(self.config.robots_count)]
            _interp_rbs = self._get_interp_robots(q_1_best, p_id_rbs)
            if _interp_rbs is None:
                return None
            int_q, q_2_best, angOffset = _interp_rbs
            for rb in range(self.config.robots_count):
                totalInt_q_rbs[rb] = np.vstack((totalInt_q_rbs[rb], int_q[rb]))
                # max_angleOffset[rb] =
                # totalAngle[rb] = np.vstack((totalAngle[rb], angOffset[rb]))
            totalAngle_rbs = totalAngle_rbs + np.max(np.array(angOffset), axis=1)

        int_count = [np.shape(totalInt_q_rbs[rb])[0] for rb in range(self.config.robots_count)]
        max_int_count = max(int_count)
        for rb in range(self.config.robots_count):
            need_append_count = max_int_count - int_count[rb]
            org = self.config.org_pos.copy()[None, :]
            org_need_append = np.repeat(org, need_append_count, axis=0)
            totalInt_q_rbs[rb] = np.vstack((totalInt_q_rbs[rb], org_need_append))

        return totalInt_q_rbs, totalAngle_rbs

    def _get_interp_slicing(
        self, totalInt_q_rbs: List[np.ndarray], level: int, preInd
    ) -> Optional[Tuple[List[np.ndarray], bool, np.ndarray]]:

        slicing_count = 0
        int_count = totalInt_q_rbs[0].shape[0]
        if int_count == 0:
            return None
        for i in range(level):
            slicing_count = slicing_count + np.power(2, i)
        spacing = int_count // (slicing_count + 1)
        slicingInd = np.arange(0, int_count, spacing)
        lastEleFlag = int_count % (slicing_count + 1)
        if lastEleFlag == 0 and level == 1:
            slicingInd = np.hstack((slicingInd, int_count - 1))
        checkSlicing = slicingInd.copy()
        for i in preInd:
            slicingInd = np.delete(slicingInd, np.where(slicingInd == i))
        for rb in range(self.config.robots_count):
            totalInt_q_rbs[rb] = totalInt_q_rbs[rb][slicingInd, :]
        if np.all(checkSlicing == np.arange(int_count)):
            return (
                totalInt_q_rbs,
                True,
                slicingInd,
            )
        return totalInt_q_rbs, False, slicingInd

    def interpolation_step(self, chromosome: np.ndarray) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        _interpolation = self.interpolation(chromosome)
        if _interpolation is None:
            return None
        totalInt_q_rbs, totalAngle_rbs = _interpolation
        int_count = totalInt_q_rbs[0].shape[0]
        len_path = np.min([len(self.robots[rb]) for rb in range(self.config.robots_count)])
        if self.step + 1 == self.num_slicing:
            return _interpolation
        split_num = int_count // len_path // (self.step + 1)
        if split_num == 0:
            split_num = 1
        for rb in range(self.config.robots_count):
            totalInt_q_rbs[rb] = totalInt_q_rbs[rb][::split_num, :]
        return totalInt_q_rbs, totalAngle_rbs

    def is_near_orgPos(self, q: np.ndarray) -> bool:
        diff = np.abs(q - self.config.org_pos)
        cond = diff <= 0.000001
        if np.all(cond):
            return True
        return False

    def is_outOfRange(self, totalInt_q) -> bool:
        isOutOfRange_q = [self.rc.cv_joints_range(totalInt_q[rb]) for rb in range(self.config.robots_count)]

        if any(isOutOfRange_q):
            return True
        return False

    def is_collision(self, totalInt_q) -> bool:
        int_count = np.shape(totalInt_q[0])[0]
        for rb in range(0, self.config.robots_count):
            for rb_next in range(rb + 1, self.config.robots_count):
                for i in range(int_count):
                    if not self.is_near_orgPos(totalInt_q[rb][i, :]) or not self.is_near_orgPos(
                        totalInt_q[rb_next][i, :]
                    ):
                        if self.rc.cv_collision(
                            totalInt_q[rb][i, :],
                            totalInt_q[rb_next][i, :],
                            self.robots[rb],
                            self.robots[rb_next],
                        ):
                            return True
        return False

    def score_slicing(self, chromosome, logging) -> Tuple[float, float]:
        # chromosome = np.array(hashable_chromosome)
        _interpolation = self.interpolation(chromosome)
        if _interpolation is not None:
            totalInt_q, totalAngle, std_rbs_angleOffset = (
                _interpolation[0],
                np.sum(_interpolation[1]),
                np.std(_interpolation[1]),
            )

            if self.is_outOfRange(totalInt_q):
                collisionScore = 90000000
                msg = "Out of Range!"
                print(msg)
                logging.save_status(msg)
            else:
                levelOfSlicing = 0
                collisionScore = 0
                baseScore = 1000000
                preInd = np.zeros(0)
                while True:
                    levelOfSlicing = levelOfSlicing + 1
                    checkPoint = self._get_interp_slicing(totalInt_q, levelOfSlicing, preInd)
                    if checkPoint is None:
                        collisionScore = 0
                        msg = "Save, but all points on one side!"
                        print(msg)
                        logging.save_status(msg)
                        self.feasibleSol_count += 1
                        break
                    preInd = np.hstack((preInd, checkPoint[3]))
                    if checkPoint[2] is not True:
                        if self.is_collision(checkPoint[0], checkPoint[1]):
                            collisionScore = 1000000
                            collisionScore = collisionScore + (baseScore / (levelOfSlicing * 2))
                            msg = "Collision!"
                            print(msg)
                            logging.save_status(msg)
                            break
                    else:
                        if self.is_collision(checkPoint[0], checkPoint[1]):
                            collisionScore = 1000000
                            collisionScore = collisionScore + (baseScore / (levelOfSlicing * 2))
                            msg = "Collision!"
                            print(msg)
                            logging.save_status(msg)
                            break
                        else:
                            collisionScore = 0
                            msg = "Save!"
                            print(msg)
                            logging.save_status(msg)
                            self.feasibleSol_count += 1
                            break
        else:
            totalAngle = 0
            collisionScore = 90000000
            msg = "Out of Range!"
            print(msg)
            logging.save_status(msg)

        score_dist = totalAngle + collisionScore
        score_dist = score_dist // 5 * 5
        std_rbs_angleOffset = std_rbs_angleOffset // 5 * 5
        return score_dist, std_rbs_angleOffset

    def score_step(self, chromosome, logging) -> Tuple[float, float]:
        _interpolation = self.interpolation_step(chromosome)
        if _interpolation is not None:
            totalInt_q, totalAngle, std_rbs_angleOffset = (
                _interpolation[0],
                np.sum(_interpolation[1]),
                np.std(_interpolation[1]),
            )

            if self.is_outOfRange(totalInt_q):
                collisionScore = 90000000
                msg = "Out of Range! _ interp"
                print(msg)
                logging.save_status(msg)
            else:
                if self.is_collision(totalInt_q):
                    collisionScore = 1000000
                    msg = "Collision!"
                    print(msg)
                    logging.save_status(msg)
                else:
                    collisionScore = 0
                    msg = "Save!"
                    print(msg)
                    logging.save_status(msg)
                    self.feasibleSol_count += 1
        else:
            totalAngle = 0
            collisionScore = 90000000
            std_rbs_angleOffset = 10000
            msg = "Out of Range! _ None"
            print(msg)
            logging.save_status(msg)
        # self.feasibleSol_list.append(self.feasibleSol_count)
        score_dist = totalAngle + collisionScore
        score_dist = score_dist // 5 * 5
        std_rbs_angleOffset = std_rbs_angleOffset // 5 * 5
        return score_dist, std_rbs_angleOffset

    def score_single_robot(self):
        pass
