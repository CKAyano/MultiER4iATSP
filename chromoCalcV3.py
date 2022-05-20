from argparse import ArgumentError
from matplotlib import pyplot as plt
import numpy as np
from robotCalc_pygeos import RobotCalc_pygeos, Coord
from robotInfo import Config, Robot, Position
from typing import List, Tuple, Optional


class LeftoverPopulation:
    leftover_population = None


class CrowdingMode:
    def mode_random(elimed_pop, insert_count):
        pop_rand = elimed_pop.copy()
        pop_rand.initChrom(insert_count)
        # pop = elimed_pop + pop_rand
        return pop_rand

    def mode_replace(elimed_pop, insert_count):
        pop_leftover = LeftoverPopulation.leftover_population
        chooseflags = CrowdingMode._rand_bool(insert_count, pop_leftover.sizes)
        pop_choose = pop_leftover[chooseflags]
        pop = elimed_pop + pop_choose
        return pop

    def _rand_bool(true_count, length):
        chooseflags = np.full((length,), False)
        idx = np.arange(length)
        np.random.shuffle(idx)
        chooseflags[idx[:true_count]] = True
        return chooseflags


class ChromoCalcV3:
    def __init__(
        self, config: Config, points: np.ndarray, step: int, gen_step_count: int, feasibleSol_list: List
    ) -> None:
        try:
            if step >= gen_step_count:
                raise RuntimeError(f"'step'(={step}) should under {gen_step_count}.")
        except RuntimeError as e:
            print(e)
            raise

        self.config = config
        self.rc = RobotCalc_pygeos(self.config)
        self.px = points[:, 0]
        self.py = points[:, 1]
        self.pz = points[:, 2]
        self.step = step
        self.gen_step_count = gen_step_count
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
    def _hamming_distance(self_chromo, other_chromo):
        dist = np.count_nonzero(self_chromo != other_chromo)
        return dist

    def _set_chromo_crowding(self, pop, need_elim):
        insert_count = np.count_nonzero(need_elim)
        if insert_count == 0:
            return
        pop_copy = pop.copy()
        pop = pop_copy[~need_elim]
        # chromos = pop.Chrom[need_elim]
        # pop = np.delete(pop, idx_chromo_need_replace, 0)
        # pop.Chrom = np.delete(pop.Chrom, need_elim, 0)
        # pop.Phen = np.delete(pop.Phen, need_elim, 0)
        # pop.CV = np.delete(pop.Phen, need_elim, 0)
        # pop.ObjV = np.delete(pop.Phen, need_elim, 0)
        if self.config.hamming_crowding_mode == "random":
            # for _ in range(insert_count):
            #     if self.config.robots_count == 1:
            #         need_append = np.arange(1, self.px.shape[0] + 1)
            #     else:
            #         need_append = np.arange(self.robots[-2].delimiter, self.px.shape[0] + 1)
            #     np.random.shuffle(need_append)
            #     pop.Chrom = np.vstack((pop.Chrom, need_append))
            #     pop.Phen = np.vstack((pop.Phen, need_append))
            poped = CrowdingMode.mode_random(pop, insert_count)
            score_dist = np.ones((poped.sizes, 1))
            score_unif = np.ones((poped.sizes, 1))
            for chromo_id in range(poped.sizes):
                score_all = self.score_step(pop.Chrom[chromo_id, :])
                score_dist[chromo_id] = score_all[0]
                # aim2 手臂點分佈最平均
                score_unif[chromo_id] = score_all[1]
            poped.CV = np.hstack((score_dist - 1000000, score_unif - 10000))
            poped.ObjV = np.hstack((score_dist, score_unif))
            pop = pop + poped
        elif self.config.hamming_crowding_mode == "reverse":
            # delim_list = [self.robots[rb].delimiter for rb in range(self.config.robots_count)]
            # for i in range(insert_count):
            #     chromo = chromos[i]
            #     num = 0
            #     pre_i = 0
            #     need_append = np.zeros((0,))
            #     for i, gene in enumerate(chromo):
            #         if gene in delim_list:
            #             temp = chromo[pre_i:i]
            #             pre_i = i + 1
            #             temp = np.flip(temp)
            #             need_append = np.hstack((need_append, temp, num))
            #             num -= 1
            #     need_append = np.hstack((need_append, np.flip(chromo[pre_i:])))
            #     need_append = need_append.astype(int)
            #     pop.Chrom = np.vstack((pop.Chrom, need_append))
            #     pop.Phen = np.vstack((pop.Phen, need_append))
            try:
                print("suspended")
                raise RuntimeError("This mode is suspended.")
            except RuntimeError as e:
                print(repr(e))
                raise

        elif self.config.hamming_crowding_mode == "replace":
            poped = CrowdingMode.mode_replace(pop, insert_count)
            pop = poped

    def hamming_crowding(self, pop, threshold):
        fonts = self._fast_nondominated_sorting(pop)
        n = []
        need_elim = np.full((pop.sizes,), False)
        for f in fonts[0:-1]:
            i, chromo_1 = f[0]
            for i, chromo in f[1:]:
                dist = self._hamming_distance(chromo_1, chromo)
                if dist <= threshold:
                    need_elim[i] = True
                    n.append(i)
        if len(n) > 0:
            print()
        need_elim = np.array(need_elim)
        self._set_chromo_crowding(pop, need_elim)

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

    def recombine_init_chromo(self, chromosome: np.ndarray, chromo_id: int, pop) -> None:

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

    # def _get_angle_changes_best_q2(
    def _get_best_q_2(
        self, p_id_rbs: List[int], q_1_best_rbs: List[np.ndarray]
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:

        q_2_best_rbs = []
        # angle_changes_rbs = []
        for rb in range(self.config.robots_count):
            if p_id_rbs[rb] == -1:
                q_2_best_rbs.append(self.config.org_pos)
            else:
                vv_a = Coord(self.px[p_id_rbs[rb]], self.py[p_id_rbs[rb]], self.pz[p_id_rbs[rb]])
                q_2_best_rbs.append(self.rc.coord2bestAngle(vv_a, q_1_best_rbs[rb], self.robots[rb]))
                is_q_nan = np.isnan(q_2_best_rbs[rb])
                if np.any(is_q_nan):
                    return None
        return q_2_best_rbs

    def _get_interp_max_deg_between_points(
        self, max_deg: int, q_1_best: np.ndarray, q_2_best: np.ndarray
    ) -> np.ndarray:

        if q_1_best.ndim == 1:
            int_q = np.expand_dims(q_1_best, axis=0)
        if max_deg == 0:
            int_q = np.vstack((int_q, q_2_best))
        else:
            offset_q = (q_2_best - q_1_best) / max_deg
            for i in range(max_deg):
                int_q = np.vstack((int_q, int_q[i, :] + offset_q))
        int_q = np.delete(int_q, 0, axis=0)
        return int_q

    def _get_interp_poly_traj_between_points(self, q_1_best: np.ndarray, q_2_best: np.ndarray):
        time_step, int_q = Trajectory.get_trajectory(
            q_1_best, q_2_best, self.config.mean_motion_velocity_rad, self.config.interp_step_period
        )
        # int_q_out = int_q[1:, :]
        time_spend = time_step[-1, 0] - time_step[0, 0]
        return time_spend, int_q[1:, :]

    def _get_interp_for_all_robots(
        self, q_1_best_rbs: List[np.ndarray], p_id_rbs: List[int]
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:

        q_2_best_rbs = self._get_best_q_2(p_id_rbs, q_1_best_rbs)
        if q_2_best_rbs is None:
            return None

        interp_q_rbs = []
        obj_values_rbs = []
        for rb in range(self.config.robots_count):
            if self.config.interp_mode == "max_deg":
                angle_changes = np.degrees(np.abs(q_2_best_rbs[rb] - q_1_best_rbs[rb]))
                max_deg = int(np.max(angle_changes))
                interp_q_rbs.append(
                    self._get_interp_max_deg_between_points(max_deg, q_1_best_rbs[rb], q_2_best_rbs[rb])
                )
                obj_values_rbs.append(np.max(angle_changes))
            if self.config.interp_mode == "poly_traj":
                time_spend, interp_q = self._get_interp_poly_traj_between_points(
                    q_1_best_rbs[rb], q_2_best_rbs[rb]
                )
                interp_q_rbs.append(interp_q)
                obj_values_rbs.append(time_spend)

        return interp_q_rbs, q_2_best_rbs, obj_values_rbs

    def interpolation(self, chromosome: np.ndarray) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        self.set_robotsPath(chromosome)

        is_firstLoop = True
        total_obj_values_rbs = 0
        joints_count = self.config.org_pos.size

        len_pointIndex = len(self.robots[0].point_index)
        totalInt_q_rbs = [np.zeros((0, joints_count))] * self.config.robots_count
        total_obj_values_rbs = 0
        for i in range(len_pointIndex):
            if is_firstLoop:
                q_1_best = [self.config.org_pos] * self.config.robots_count
                is_firstLoop = False
            else:
                q_1_best = q_2_best

            p_id_rbs = [self.robots[rb].point_index[i] for rb in range(self.config.robots_count)]
            _interp_rbs = self._get_interp_for_all_robots(q_1_best, p_id_rbs)
            if _interp_rbs is None:
                return None
            int_q, q_2_best, obj_values_rbs = _interp_rbs
            for rb in range(self.config.robots_count):
                totalInt_q_rbs[rb] = np.vstack((totalInt_q_rbs[rb], int_q[rb]))
                # max_angleOffset[rb] =
                # totalAngle[rb] = np.vstack((totalAngle[rb], angOffset[rb]))
            total_obj_values_rbs = total_obj_values_rbs + np.array(obj_values_rbs)

        int_count = [np.shape(totalInt_q_rbs[rb])[0] for rb in range(self.config.robots_count)]
        max_int_count = max(int_count)
        for rb in range(self.config.robots_count):
            need_append_count = max_int_count - int_count[rb]
            org = self.config.org_pos.copy()[None, :]
            org_need_append = np.repeat(org, need_append_count, axis=0)
            totalInt_q_rbs[rb] = np.vstack((totalInt_q_rbs[rb], org_need_append))

        return totalInt_q_rbs, total_obj_values_rbs

    def __SUSPENDED__get_interp_slicing(
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
        if self.step + 1 == self.gen_step_count:
            return _interpolation

        totalInt_q_rbs, totalAngle_rbs = _interpolation
        int_count = totalInt_q_rbs[0].shape[0]
        len_path = np.min([len(self.robots[rb]) for rb in range(self.config.robots_count)])
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

    def __SUSPENDED__score_slicing(self, chromosome, logging) -> Tuple[float, float]:
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
                    checkPoint = self.__SUSPENDED__get_interp_slicing(totalInt_q, levelOfSlicing, preInd)
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

    def score_step(self, chromosome, logging=None) -> Tuple[float, float]:
        if self.config.robots_count == 1:
            obj_1, totalDistance = self.score_single_robot(chromosome, logging)
            return obj_1, totalDistance

        _interpolation = self.interpolation_step(chromosome)
        if _interpolation is not None:
            totalInt_q, obj_1, obj_2 = (
                _interpolation[0],
                np.max(_interpolation[1]),
                np.std(_interpolation[1]),
            )

            if self.is_outOfRange(totalInt_q):
                collisionScore = 90000000
                msg = "Out of Range! _ interp"
                print(msg)
                if logging:
                    logging.save_status(msg)
            else:
                if self.is_collision(totalInt_q):
                    collisionScore = 1000000
                    msg = "Collision!"
                    print(msg)
                    if logging:
                        logging.save_status(msg)
                else:
                    collisionScore = 0
                    msg = "Save!"
                    print(msg)
                    if logging:
                        logging.save_status(msg)
                    self.feasibleSol_count += 1
        else:
            obj_1 = 0
            collisionScore = 90000000
            obj_2 = 10000
            msg = "Out of Range! _ None"
            print(msg)
            if logging:
                logging.save_status(msg)
        # self.feasibleSol_list.append(self.feasibleSol_count)
        obj_1 = obj_1 + collisionScore
        # obj_1 = obj_1 // 5 * 5
        # obj_2 = obj_2 // 5 * 5
        return obj_1, obj_2

    def get_point_distance(self, totalInt_q_rbs):
        dist_rbs = []
        for rb in range(self.config.robots_count):
            totalInt_q = totalInt_q_rbs[rb]
            dist = 0
            for i in range(totalInt_q.shape[0] - 1):
                i_next = i + 1
                vv_all_i = self.rc.userFK(totalInt_q[i, :])
                vv_i = vv_all_i.v5
                vv_all_i_next = self.rc.userFK(totalInt_q[i_next, :])
                vv_i_next = vv_all_i_next.v5
                _dist = vv_i.distance(vv_i_next)
                dist += _dist

            dist_rbs.append(dist)
        return dist_rbs

    def score_single_robot(self, chromosome, logging):
        def _get_score(_totalAngle, _collisionScore, _totalDistance):
            _score_dist = _totalAngle + _collisionScore
            _score_dist = _score_dist // 5 * 5
            _totalDistance = _totalDistance // 5 * 5
            return _score_dist, _totalDistance

        _interpolation = self.interpolation_step(chromosome)
        if _interpolation is None:
            totalAngle = 0
            collisionScore = 90000000
            totalDistance = 1000000
            msg = "Out of Range! _ None"
            print(msg)
            logging.save_status(msg)
            score_dist, totalDistance = _get_score(totalAngle, collisionScore, totalDistance)
            return score_dist, totalDistance

        totalInt_q, totalAngle = (
            _interpolation[0],
            np.sum(_interpolation[1]),
        )
        totalDistance = self.get_point_distance(totalInt_q)
        totalDistance = totalDistance[0]

        if self.is_outOfRange(totalInt_q):
            collisionScore = 90000000
            msg = "Out of Range! _ interp"
            print(msg)
            logging.save_status(msg)
            score_dist, totalDistance = _get_score(totalAngle, collisionScore, totalDistance)
            return score_dist, totalDistance

        collisionScore = 0
        msg = "Save!"
        print(msg)
        logging.save_status(msg)
        self.feasibleSol_count += 1
        score_dist, totalDistance = _get_score(totalAngle, collisionScore, totalDistance)

        return score_dist, totalDistance


class Trajectory:
    def _traj_s(a, b, c, d, e, f, t):
        traj_s_poly = np.poly1d([a, b, c, d, e, f])
        traj_s = traj_s_poly(t)
        t_s = np.hstack((t[:, None], traj_s[:, None]))
        return t_s

    def _traj_v(a, b, c, d, e, t):
        traj_v_poly = np.poly1d([5 * a, 4 * b, 3 * c, 2 * d, e])
        traj_v = traj_v_poly(t)
        t_v = np.hstack((t[:, None], traj_v[:, None]))
        return t_v

    def _traj_a(a, b, c, d, t):
        traj_a_poly = np.poly1d([20 * a, 12 * b, 6 * c, 2 * d])
        traj_a = traj_a_poly(t)
        t_a = np.hstack((t[:, None], traj_a[:, None]))
        return t_a

    def quintic_polynomial_matrix(
        step_count: int,
        t_start: float,
        t_end: float,
        s_start: float,
        s_end: float,
        v_start=0.0,
        v_end=0.0,
        a_start=0.0,
        a_end=0.0,
        nargout=1,
    ):
        poly_martix = np.array(
            [
                [t_start ** 5, t_start ** 4, t_start ** 3, t_start ** 2, t_start, 1],
                [t_end ** 5, t_end ** 4, t_end ** 3, t_end ** 2, t_end, 1],
                [5 * (t_start ** 4), 4 * (t_start ** 3), 3 * (t_start ** 2), 2 * t_start, 1, 0],
                [5 * (t_end ** 4), 4 * (t_end ** 3), 3 * (t_end ** 2), 2 * t_end, 1, 0],
                [20 * (t_start ** 3), 12 * (t_start ** 2), 6 * t_start, 2, 0, 0],
                [20 * (t_end ** 3), 12 * (t_end ** 2), 6 * t_end, 2, 0, 0],
            ]
        )
        traj_obj = np.array([s_start, s_end, v_start, v_end, a_start, a_end])
        traj_obj = traj_obj[:, None]
        poly_martix_inv = np.linalg.inv(poly_martix)
        coeffs = poly_martix_inv.dot(traj_obj)
        a, b, c, d, e, f = coeffs[0, 0], coeffs[1, 0], coeffs[2, 0], coeffs[3, 0], coeffs[4, 0], coeffs[5, 0]
        print(a, b, c, d, e, f)
        time_step = np.linspace(t_start, t_end, step_count)
        if nargout == 1:
            s = Trajectory._traj_s(a, b, c, d, e, f, time_step)
            return s
        if nargout == 2:
            s = Trajectory._traj_s(a, b, c, d, e, f, time_step)
            v = Trajectory._traj_v(a, b, c, d, e, time_step)
            return s, v
        if nargout == 3:
            s = Trajectory._traj_s(a, b, c, d, e, f, time_step)
            v = Trajectory._traj_v(a, b, c, d, e, time_step)
            a = Trajectory._traj_a(a, b, c, d, time_step)
            return s, v, a
        try:
            if nargout not in [1, 2, 3]:
                raise ArgumentError("nargout should be 1 or 2 or 3")
        except ArgumentError as e:
            print(repr(e))
            raise

    def plot_trajectory(*trajs):
        plot_count = len(trajs)
        fig, axs = plt.subplots(plot_count, 1)
        fig.subplots_adjust(left=0.2, wspace=0.6)

        for i, p in enumerate(trajs):
            axs[i].plot(p[:, 0], p[:, 1])
            if i == 0:
                axs[i].set_xlabel("time")
                axs[i].set_ylabel("position (rad)")
            if i == 1:
                axs[i].set_xlabel("time")
                axs[i].set_ylabel("velocity (rad/s)")
            if i == 2:
                axs[i].set_xlabel("time")
                axs[i].set_ylabel("acceleration (rad/$s^2$)")
        fig.tight_layout()
        fig.align_ylabels(axs)
        plt.savefig("./fig.png", dpi=400)
        plt.show()

    def get_trajectory(
        angle_start: np.ndarray, angle_stop: np.ndarray, mean_ang_v, step_period=1 / 20, is_test=False
    ) -> Tuple[np.ndarray]:
        if angle_start.ndim == 2:
            angle_start = np.squeeze(angle_start)
        if angle_stop.ndim == 2:
            angle_stop = np.squeeze(angle_stop)

        t_start = 0
        t_end = np.max(np.abs(angle_stop - angle_start)) / mean_ang_v
        joints_count = len(angle_start)

        step_count = int((t_end - t_start) / step_period) + 1
        if step_count <= 2:
            return np.array([[t_start], [t_end]]), np.vstack((angle_start, angle_stop))

        s_joints = np.empty((step_count, 0))
        v_all = []
        for j in range(joints_count):
            s, v, a = Trajectory.quintic_polynomial(
                step_count, t_start, t_end, angle_start[j], angle_stop[j], nargout=3
            )
            if is_test:
                # ------------------------------------------ #
                v_all.append(np.mean(v))
                print(f"max v for joint {j+1} = {np.degrees(np.max(v)):.4f}")
                print(f"mean v for joint {j+1} = {np.degrees(np.mean(v)):.4f}")
                print("")
                Trajectory.plot_trajectory(s, v, a)
                # ------------------------------------------ #

            s_joints = np.hstack((s_joints, s[:, 1:2]))
        # s_joints = np.hstack((s[:, 0:1], s_joints))
        if is_test:
            print(f"mean v for all joints = {np.degrees(np.mean(v_all)):.4f}")
        return s[:, 0:1], s_joints

    def quintic_polynomial(
        step_count: int,
        t_start: float,
        t_end: float,
        s_start: float,
        s_end: float,
        v_start=0.0,
        v_end=0.0,
        a_start=0.0,
        a_end=0.0,
        nargout=1,
    ):

        a = (
            (6 / (t_start - t_end) ** 5) * (s_start - s_end)
            - (3 / (t_start + t_end) ** 4) * (v_start + v_end)
            + (1 / 2 * (t_start - t_end) ** 3) * (a_start - a_end)
        )

        b = (
            ((-15 * (t_start + t_end)) / (t_start - t_end) ** 5) * (s_start - s_end)
            + (1 / (t_start - t_end) ** 4)
            * ((7 * t_start + 8 * t_end) * v_start + (8 * t_start + 7 * t_end) * v_end)
            - (1 / (2 * (t_start - t_end) ** 3))
            * ((2 * t_start + 3 * t_end) * a_start - (3 * t_start + 2 * t_end) * a_end)
        )

        c = (
            ((10 * (t_start ** 2 + 4 * t_start * t_end + t_end ** 2)) / ((t_start - t_end) ** 5))
            * (s_start - s_end)
            - (2 / (t_start - t_end) ** 4)
            * (
                (2 * t_start ** 2 + 10 * t_start * t_end + 3 * t_end ** 2) * v_start
                + (3 * t_start ** 2 + 10 * t_start * t_end + 2 * t_end ** 2) * v_end
            )
            + (1 / 2 * (t_start - t_end) ** 3)
            * (
                (t_start ** 2 + 6 * t_start * t_end + 3 * t_end ** 2) * a_start
                - (3 * t_start ** 2 + 6 * t_start * t_end + t_end ** 2) * a_end
            )
        )

        d = (
            ((-30 * t_start * t_end * (t_start + t_end)) / ((t_start - t_end) ** 5)) * (s_start - s_end)
            + ((6 * t_start * t_end) / (t_start - t_end) ** 4)
            * ((2 * t_start + 3 * t_end) * v_start + (3 * t_start + 2 * t_end) * v_end)
            - (1 / 2 * (t_start - t_end) ** 3)
            * (
                t_end * (3 * t_start ** 2 + 6 * t_start * t_end + t_end ** 2) * a_start
                - t_start * (t_start ** 2 + 6 * t_start * t_end + 3 * t_end ** 2) * a_end
            )
        )

        e = (
            (30 * t_start ** 2 * t_end ** 2) / ((t_start - t_end) ** 5) * (s_start - s_end)
            + (1 / (t_start - t_end) ** 4)
            * (
                t_end ** 2 * (-6 * t_start + t_end) * (2 * t_start + t_end) * v_start
                - t_start ** 2 * (-t_start + 6 * t_end) * (t_start + 2 * t_end) * v_end
            )
            + ((t_start * t_end) / (2 * (t_start - t_end) ** 3))
            * ((t_end * (3 * t_start + 2 * t_end) * a_start) - (t_start * (2 * t_start + 3 * t_end) * a_end))
        )

        f = -(
            (1 / (t_start - t_end) ** 5)
            * (
                t_end ** 3 * (10 * t_start ** 2 - 5 * t_start * t_end + t_end ** 2) * s_start
                - t_start ** 3 * (t_start ** 2 - 5 * t_start * t_end + 10 * t_end ** 2) * s_end
            )
            + (t_start * t_end / (t_start - t_end) ** 4)
            * (t_end ** 2 * (4 * t_start - t_end) * v_start + t_start ** 2 * (-t_start + 4 * t_end) * v_end)
            - (t_start ** 2 * t_end ** 2)
            / (2 * (t_start - t_end) ** 3)
            * (t_end * a_start - t_start * a_end)
        )
        # print(a, b, c, d, e, f)
        # if np.isnan(a):
        #     print()

        time_step = np.linspace(t_start, t_end, step_count)
        if nargout == 1:
            s = Trajectory._traj_s(a, b, c, d, e, f, time_step)
            return s
        if nargout == 2:
            s = Trajectory._traj_s(a, b, c, d, e, f, time_step)
            v = Trajectory._traj_v(a, b, c, d, e, time_step)
            return s, v
        if nargout == 3:
            s = Trajectory._traj_s(a, b, c, d, e, f, time_step)
            v = Trajectory._traj_v(a, b, c, d, e, time_step)
            a = Trajectory._traj_a(a, b, c, d, time_step)
            return s, v, a
        try:
            if nargout not in [1, 2, 3]:
                raise ArgumentError("nargout should be 1 or 2 or 3")
        except ArgumentError as e:
            print(repr(e))
            raise
