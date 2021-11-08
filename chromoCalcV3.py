import numpy as np
from robotCalc_pygeos import RobotCalc_pygeos, Coord
from robotInfo import Config, Robot, Position

# from caching import np_cache


class ChromoCalcV3:
    def __init__(self, config: Config, points: np.ndarray, step: int, num_slicing: int):
        self.config = config
        self.rc = RobotCalc_pygeos(self.config)
        self.px = points[:, 0]
        self.py = points[:, 1]
        self.pz = points[:, 2]
        self.chromo_index = None
        self.step = step
        self.num_slicing = num_slicing
        self.robots: list[Robot] = []
        position = [Position.LEFT, Position.RIGHT, Position.UP, Position.DOWN]
        for i in range(self.config.robots_count):
            self.robots.append(Robot(i, position[i]))

    def adj_chromo(self, chromosome, pop):
        def need_preAdj(pointIndex: int, robot: Robot):
            vv_robot = Coord(
                self.px[pointIndex], self.py[pointIndex], self.pz[pointIndex]
            )
            vv_robot = self.rc.robot2world(vv_robot, robot.position)
            q_robot = self.rc.userIK(vv_robot)

            isOutRange = []
            for i in range(q_robot.shape[0]):
                isOutRange.append(self.rc.cv_joints_range(q_robot[i, :]))
            if np.all(isOutRange):
                return True
            return False

        def which_rb(position: Position):
            which_rb = [
                _rb
                for _rb in range(self.config.robots_count)
                if self.robots[_rb].position == position
            ]
            return which_rb[0]

        def throw_path(
            robotPath_after, robots_needMove, org_pos: Position, dist_Pos: Position
        ):
            org_rb = which_rb(org_pos)
            dist_rb = which_rb(dist_Pos)
            if self.robots[org_rb].position == org_pos:
                robotPath_after[org_rb] = np.hstack(
                    (
                        robotPath_after[org_rb],
                        self.robots[dist_rb].robot_path[robots_needMove[dist_rb]],
                    )
                )
                robotPath_after[dist_rb] = np.hstack(
                    (
                        robotPath_after[dist_rb],
                        self.robots[org_rb].robot_path[robots_needMove[org_rb]],
                    )
                )
                self.robots[org_rb].robot_path = robotPath_after[org_rb]
                self.robots[dist_rb].robot_path = robotPath_after[dist_rb]

        self.set_robotsPath(chromosome)

        robotPath_after = []
        robots_needMove = [np.zeros(0, dtype=int)] * self.config.robots_count
        for rb in range(self.config.robots_count):
            for i in range(len(self.robots[rb])):
                point_index = int(self.robots[rb].point_index[i])
                if need_preAdj(point_index, self.robots[rb]):
                    robots_needMove[rb] = np.hstack((robots_needMove[rb], i))
            _robotPath_after = self.robots[rb].robot_path.copy()
            _robotPath_after = np.delete(_robotPath_after, robots_needMove[rb])
            robotPath_after.append(_robotPath_after)

        throw_path(robotPath_after, robots_needMove, Position.LEFT, Position.RIGHT)
        throw_path(robotPath_after, robots_needMove, Position.UP, Position.DOWN)

        chromo = self.robots[0].robot_path
        for i in range(self.config.robots_count - 1):
            chromo = np.hstack((chromo, 0 - i, self.robots[i + 1].robot_path))
        pop.Phen[self.chromo_index, :] = chromo
        pop.Chrom[self.chromo_index, :] = chromo

    def set_robotsPath(self, chromosome):
        if chromosome.ndim >= 2:
            chromosome = np.squeeze(chromosome)
        # chromosome = population[chromoIndex, :]  # [1, 2, 3, 4, 0, 5, 6]
        robotPath_left = chromosome.copy()

        for i in range(self.config.robots_count - 1):
            mask = np.isin(
                robotPath_left,
                [
                    self.robots[rb].delimiter
                    for rb in range(self.config.robots_count - 1)
                ],
            )
            sep_index = np.where(mask)  # index of 0 = 4
            _index = sep_index[0][0]
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

    def interp_oneSeq(self, checkPoints_count, q_1_best, q_2_best):
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

    def angleOffset(self, loc, q_1_best):
        q_2_best = []
        angleOffset = []
        for rb in range(self.config.robots_count):
            if loc[rb] == -1:
                q_2_best.append(self.config.org_pos)
            else:
                vv_a = Coord(self.px[loc[rb]], self.py[loc[rb]], self.pz[loc[rb]])
                q_2_best.append(
                    self.rc.coord2bestAngle(vv_a, q_1_best[rb], self.robots[rb])
                )
                isQaNan = np.isnan(q_2_best[rb])
                if np.any(isQaNan):
                    return None
            angleOffset.append(np.degrees(np.abs(q_2_best[rb] - q_1_best[rb])))

        return angleOffset, q_2_best

    def get_robots_interp(self, q_1_best, loc):
        _angleOffset = self.angleOffset(loc, q_1_best)
        if _angleOffset is None:
            return None
        angOffset, q_2_best = _angleOffset[0], _angleOffset[1]

        int_q = []
        for rb in range(self.config.robots_count):
            seqCheckNum = int(np.max(angOffset[rb]))
            int_q.append(self.interp_oneSeq(seqCheckNum, q_1_best[rb], q_2_best[rb]))

        return int_q, q_2_best, angOffset

    def interpolation(self, chromosome):
        self.set_robotsPath(chromosome)

        is_firstLoop = True
        totalAngle = 0
        joints_count = self.config.org_pos.size

        len_pointIndex = len(self.robots[0].point_index)
        totalInt_q = [np.zeros((0, joints_count))] * 4
        totalAngle = 0
        for i in range(len_pointIndex):
            if is_firstLoop:
                q_1_best = [self.config.org_pos] * self.config.robots_count
                is_firstLoop = False
            else:
                q_1_best = q_2_best

            loc = [
                self.robots[rb].point_index[i] for rb in range(self.config.robots_count)
            ]
            _genTwoRobotInt = self.get_robots_interp(q_1_best, loc)
            if _genTwoRobotInt is None:
                return None
            int_q, q_2_best, angOffset = _genTwoRobotInt
            for rb in range(self.config.robots_count):
                totalInt_q[rb] = np.vstack((totalInt_q[rb], int_q[rb]))
                # max_angleOffset[rb] =
                # totalAngle[rb] = np.vstack((totalAngle[rb], angOffset[rb]))
            totalAngle = totalAngle + np.max(np.array(angOffset), axis=1)

        int_count = [
            np.shape(totalInt_q[rb])[0] for rb in range(self.config.robots_count)
        ]
        max_int_count = max(int_count)
        for rb in range(self.config.robots_count):
            need_append_count = max_int_count - int_count[rb]
            org = self.config.org_pos.copy()[None, :]
            org_need_append = np.repeat(org, need_append_count, axis=0)
            totalInt_q[rb] = np.vstack((totalInt_q[rb], org_need_append))

        return totalInt_q, totalAngle

    def slicingCheckPoint(self, totalInt_q, levelOfSlicing, preInd):
        slicing_count = 0
        int_count = totalInt_q[0].shape[0]
        if int_count == 0:
            return None
        for i in range(levelOfSlicing):
            slicing_count = slicing_count + np.power(2, i)
        spacing = int_count // (slicing_count + 1)
        slicingInd = np.arange(0, int_count, spacing)
        lastEleFlag = int_count % (slicing_count + 1)
        if lastEleFlag == 0 and levelOfSlicing == 1:
            slicingInd = np.hstack((slicingInd, int_count - 1))
        checkSlicing = slicingInd.copy()
        for i in preInd:
            slicingInd = np.delete(slicingInd, np.where(slicingInd == i))
        for rb in range(self.config.robots_count):
            totalInt_q[rb] = totalInt_q[rb][slicingInd, :]
        if np.all(checkSlicing == np.arange(int_count)):
            return (
                totalInt_q,
                True,
                slicingInd,
            )
        return totalInt_q, False, slicingInd

    def interpolation_step(self, chromosome):
        _interpolation = self.interpolation(chromosome)
        if _interpolation is None:
            return None
        totalInt_q, totalAngle = _interpolation
        int_count = totalInt_q[0].shape[0]
        len_path = np.min(
            [len(self.robots[rb]) for rb in range(self.config.robots_count)]
        )
        if self.step + 1 == self.num_slicing:
            return _interpolation
        split_num = int_count // len_path // (self.step + 1)
        if split_num == 0:
            split_num = 1
        for rb in range(self.config.robots_count):
            totalInt_q[rb] = totalInt_q[rb][::split_num, :]
        return totalInt_q, totalAngle

    def is_near_orgPos(self, q: np.ndarray) -> bool:
        diff = np.abs(q - self.config.org_pos)
        cond = diff <= 0.000001
        if np.all(cond):
            return True
        return False

    def is_outOfRange(self, totalInt_q) -> bool:
        isOutOfRange_q = [
            self.rc.cv_joints_range(totalInt_q[rb])
            for rb in range(self.config.robots_count)
        ]

        if any(isOutOfRange_q):
            return True
        return False

    def is_collision(self, totalInt_q) -> bool:
        int_count = np.shape(totalInt_q[0])[0]
        for rb in range(0, self.config.robots_count):
            for rb_next in range(rb + 1, self.config.robots_count):
                for i in range(int_count):
                    if not self.is_near_orgPos(
                        totalInt_q[rb][i, :]
                    ) or not self.is_near_orgPos(totalInt_q[rb_next][i, :]):
                        if self.rc.cv_collision(
                            totalInt_q[rb][i, :],
                            totalInt_q[rb_next][i, :],
                            self.robots[rb],
                            self.robots[rb_next],
                        ):
                            return True
        return False

    # @np_cache
    def score_slicing(self, chromosome, logging):
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
                    checkPoint = self.slicingCheckPoint(
                        totalInt_q, levelOfSlicing, preInd
                    )
                    if checkPoint is None:
                        collisionScore = 0
                        msg = "Save, but all points on one side!"
                        print(msg)
                        logging.save_status(msg)
                        break
                    preInd = np.hstack((preInd, checkPoint[3]))
                    if checkPoint[2] is not True:
                        if self.is_collision(checkPoint[0], checkPoint[1]):
                            collisionScore = 1000000
                            collisionScore = collisionScore + (
                                baseScore / (levelOfSlicing * 2)
                            )
                            msg = "Collision!"
                            print(msg)
                            logging.save_status(msg)
                            break
                    else:
                        if self.is_collision(checkPoint[0], checkPoint[1]):
                            collisionScore = 1000000
                            collisionScore = collisionScore + (
                                baseScore / (levelOfSlicing * 2)
                            )
                            msg = "Collision!"
                            print(msg)
                            logging.save_status(msg)
                            break
                        else:
                            collisionScore = 0
                            msg = "Save!"
                            print(msg)
                            logging.save_status(msg)
                            break
        else:
            totalAngle = 0
            collisionScore = 90000000
            msg = "Out of Range!"
            print(msg)
            logging.save_status(msg)

        score_dist = totalAngle + collisionScore
        return [score_dist, std_rbs_angleOffset]

    def score_step(self, chromosome, logging):
        # chromosome = np.array(hashable_chromosome)
        _interpolation = self.interpolation_step(chromosome)
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
        else:
            totalAngle = 0
            collisionScore = 90000000
            std_rbs_angleOffset = 10000
            msg = "Out of Range!"
            print(msg)
            logging.save_status(msg)

        score_dist = totalAngle + collisionScore
        return [score_dist, std_rbs_angleOffset]
