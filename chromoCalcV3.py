import numpy as np
from numpy.lib.function_base import append
from robotCalc_pygeos import RobotCalc_pygeos, Coord
from robotInfo import Robot, Position

# from caching import np_cache


class ChromoCalcV3:
    def __init__(self, config, points, step, num_slicing):
        self.points_range = config["points_range"]
        self.baseX_offset = sum(self.points_range[0])
        self.baseY_offset = (self.points_range[1][1] - self.points_range[1][0]) / 2
        self.linkWidth = config["linkWidth"]
        self.orgPos = np.array(np.radians(config["originalPosture"]))
        self.axisRange = np.array(config["axisRange"])
        self.direct_array = np.array(config["direct_array"])
        self.rc = RobotCalc_pygeos(self.baseX_offset, self.baseY_offset, self.linkWidth)
        self.points_count = points.shape[0]
        self.px = points[:, 0]
        self.py = points[:, 1]
        self.pz = points[:, 2]
        self.outRange = False
        self.allOneSide = False
        self.chromoIndex = None
        self.middle_x = self.baseX_offset / 2
        self.step = step
        self.num_slicing = num_slicing
        self.robot_count = config["robot_count"]
        self.robots: list[Robot] = []
        position = [Position.LEFT, Position.RIGHT, Position.UP, Position.DOWN]
        for i in range(self.robot_count):
            self.robots.append(Robot(i, position[i]))

    def adjChromo(self, chromosome, pop):
        def needPreAdj(pointIndex: int, robot: Robot):
            vv_robot = Coord(
                self.px[pointIndex], self.py[pointIndex], self.pz[pointIndex]
            )
            vv_robot = self.rc.robot2world(vv_robot, robot.position)
            q_robot = self.rc.userIK(vv_robot, self.direct_array)

            isOutRange = []
            for i in range(q_robot.shape[0]):
                isOutRange.append(self.rc.cvAxisRange(q_robot[i, :], self.axisRange))
            if np.all(isOutRange):
                return True
            return False

        self.set_robotsPath(chromosome)

        for rb in range(self.robot_count):
            robots_needMove = np.zeros(0, dtype=int)
            for i in range(len(self.robots[rb])):
                point_index = int(self.robots[rb].point_index[i])
                if needPreAdj(point_index, self.robots[rb]):
                    robots_needMove = np.hstack((robots_needMove, i))
            robotPath_after = self.robots[rb].robot_path.copy()
            robotPath_after = np.delete(robotPath_after, robots_needMove)
            robotPath_after = np.hstack(
                (robotPath_after, self.robots[rb].robot_path[robots_needMove])
            )
            self.robots[rb].robot_path = robotPath_after

        chromo = self.robots[0].robot_path
        for i in range(self.robot_count - 1):
            chromo = np.hstack((chromo, 0 - i, self.robots[i + 1].robot_path))
        pop.Phen[self.chromoIndex, :] = chromo
        pop.Chrom[self.chromoIndex, :] = chromo

    def set_robotsPath(self, chromosome):
        if chromosome.ndim >= 2:
            chromosome = np.squeeze(chromosome)
        # chromosome = population[chromoIndex, :]  # [1, 2, 3, 4, 0, 5, 6]
        robotPath_left = chromosome.copy()

        for i in range(self.robot_count - 1):
            mask = np.isin(
                robotPath_left,
                [self.robots[rb].delimiter for rb in range(self.robot_count - 1)],
            )
            sep_index = np.where(mask)  # index of 0 = 4
            _index = sep_index[0][0]
            self.robots[i].robot_path = robotPath_left[:(_index)]  # [1, 2, 3, 4]
            self.robots[i].point_index = self.robots[i].robot_path - 1
            robotPath_left = robotPath_left[(_index + 1) :]  # [5, 6]
        self.robots[-1].robot_path = robotPath_left
        self.robots[-1].point_index = self.robots[-1].robot_path - 1

        max_len = max([len(self.robots[i]) for i in range(self.robot_count)])

        for i in range(self.robot_count):
            appendArray = np.ones(max_len - len(self.robots[i]), dtype="int32") * -1
            self.robots[i].point_index = np.hstack(
                (self.robots[i].point_index, appendArray, -1)
            )  # [4, 5, -1, -1]

    def intForOneSeq(self, numOfCheckPoint, q_1_best, q_2_best):
        if q_1_best.ndim == 1:
            int_q = np.expand_dims(q_1_best, axis=0)
        if numOfCheckPoint == 0:
            int_q = np.vstack((int_q, q_2_best))
        else:
            offset_q = (q_2_best - q_1_best) / numOfCheckPoint
            for i in range(numOfCheckPoint):
                int_q = np.vstack((int_q, int_q[i, :] + offset_q))
        int_q = np.delete(int_q, 0, axis=0)
        return int_q

    def coor2OptiAngle(self, vv: Coord, q_1_best, robot: Robot):
        vv = self.rc.robot2world(vv, robot.position)
        q_2 = self.rc.userIK(vv, self.direct_array)
        len_q = q_2.shape[0]
        idx_notOutRange = np.zeros((0))
        for i in range(len_q):
            q_2_test = q_2[i, :]
            q_2_test = q_2_test[None, :]
            q_2_outRange = self.rc.cvAxisRange(q_2_test, self.axisRange)
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
            q_2_best = self.rc.greedySearch(q_1_best, q_2_output)
        return q_2_best

    def angleOffset(self, loc, q_1_best):
        q_2_best = []
        angleOffset = []
        for rb in range(self.robot_count):
            if loc[rb] == -1:
                q_2_best.append(self.orgPos)
            else:
                vv_a = Coord(self.px[loc[rb]], self.py[loc[rb]], self.pz[loc[rb]])
                q_2_best.append(self.coor2OptiAngle(vv_a, q_1_best[rb], self.robots[rb]))
                isQaNan = np.isnan(q_2_best[rb])
                if np.any(isQaNan):
                    return None
            angleOffset.append(np.degrees(np.abs(q_2_best[rb] - q_1_best[rb])))

        return angleOffset, q_2_best

    def genRobotsInt(self, q_1_best, loc):
        _angleOffset = self.angleOffset(loc, q_1_best)
        if _angleOffset is None:
            return None
        angOffset, q_2_best = _angleOffset[0], _angleOffset[1]

        int_q = []
        for rb in range(self.robot_count):
            seqCheckNum = int(np.max(angOffset[rb]))
            int_q.append(self.intForOneSeq(seqCheckNum, q_1_best[rb], q_2_best[rb]))

        return int_q, q_2_best, angOffset

    def interpolation(self, chromosome):
        self.set_robotsPath(chromosome)

        isFirstLoop = True
        totalAngle = 0
        numOfJoints = self.orgPos.size
        totalInt_qa, totalInt_qb = np.zeros((0, numOfJoints)), np.zeros((0, numOfJoints))

        len_pointIndex = len(self.robots[0].point_index)
        totalInt_q = [np.zeros((0, numOfJoints))] * 4
        for i in range(len_pointIndex):
            if isFirstLoop:
                q_1_best = [self.orgPos] * self.robot_count
                isFirstLoop = False
            else:
                q_1_best = q_2_best

            loc = [self.robots[rb].point_index[i] for rb in range(self.robot_count)]
            _genTwoRobotInt = self.genRobotsInt(q_1_best, loc)
            if _genTwoRobotInt is None:
                return None
            int_q, q_2_best, angOffset = _genTwoRobotInt
            for rb in range(self.robot_count):
                totalInt_q[rb] = np.vstack((totalInt_q[rb], int_q[rb]))
            totalAngle = totalAngle + sum(angOffset)  # 兩隻手臂相加

        for rb in range(self.robot_count):
            totalInt_q[rb] = np.delete(
                totalInt_q[rb],
                np.where(
                    np.all(
                        totalInt_q[rb] < self.orgPos + 0.000001
                        and totalInt_q[rb] > self.orgPos + 0.000001,
                        axis=1,
                    )
                ),
                axis=0,
            )

        int_count = [np.shape(totalInt_q[rb])[0] for rb in range(self.robot_count)]
        min_int_count = min(int_count)
        for rb in range(self.robot_count):
            totalInt_q[rb] = totalInt_q[rb][0:min_int_count, :]
        return totalInt_q, totalAngle

    def slicingCheckPoint(self, totalInt_q, levelOfSlicing, preInd):
        self.allOneSide = False
        slicing_count = 0
        int_count = totalInt_q[0].shape[0]
        if int_count == 0:
            self.allOneSide = True
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
        for rb in range(self.robot_count):
            totalInt_q[rb] = totalInt_q[rb][slicingInd, :]
        if np.all(checkSlicing == np.arange(int_count)):
            return (
                totalInt_q,
                True,
                slicingInd,
            )
        return totalInt_q, False, slicingInd

    def intOfStep(self, chromosome):
        _interpolation = self.interpolation(chromosome)
        if _interpolation is None:
            return None
        totalInt_q, totalAngle = _interpolation
        int_count = totalInt_q[0].shape[0]
        len_path = np.min([len(self.robots[rb]) for rb in range(self.robot_count)])
        if self.step + 1 == self.num_slicing:
            return _interpolation
        split_num = int_count // len_path // (self.step + 1)
        if split_num == 0:
            split_num = 1
        for rb in range(self.robot_count):
            totalInt_q[rb] = totalInt_q[rb][::split_num, :]
        return totalInt_q, totalAngle

    def isOutOfRange(self, totalInt_q):
        isOutOfRange_q = [
            self.rc.cvAxisRange(totalInt_q[rb], self.axisRange)
            for rb in range(self.robot_count)
        ]

        if any(isOutOfRange_q):
            return True
        return False

    def isCollision(self, totalInt_q):
        int_count = np.shape(totalInt_q[0])[0]
        for rb in range(0, self.robot_count):
            for rb_next in range(rb + 1, self.robot_count):
                for i in range(int_count):
                    if self.rc.cvCollision(
                        totalInt_q[rb][i, :], totalInt_q[rb_next][i, :]
                    ):
                        return True
        return False

    # @np_cache
    def scoreOfTwoRobot(self, chromosome, logging):
        # chromosome = np.array(hashable_chromosome)
        _interpolation = self.interpolation(chromosome)
        if _interpolation is not None:
            totalInt_q, totalAngle = _interpolation

            if self.isOutOfRange(totalInt_q):
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
                    if self.allOneSide:
                        collisionScore = 0
                        msg = "Save, but all points on one side!"
                        print(msg)
                        logging.save_status(msg)
                        break
                    preInd = np.hstack((preInd, checkPoint[3]))
                    if checkPoint[2] is not True:
                        if self.isCollision(checkPoint[0], checkPoint[1]):
                            collisionScore = 1000000
                            collisionScore = collisionScore + (
                                baseScore / (levelOfSlicing * 2)
                            )
                            msg = "Collision!"
                            print(msg)
                            logging.save_status(msg)
                            break
                    else:
                        if self.isCollision(checkPoint[0], checkPoint[1]):
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
        points_eachRobot = self.points_count // self.robot_count
        robots_path_len = [
            len(self.robots[rb]) - points_eachRobot for rb in range(self.robot_count)
        ]
        score_unif = np.sum(np.abs(robots_path_len))
        return [score_dist, score_unif]

    def scoreOfTwoRobot_step(self, chromosome, logging):
        # chromosome = np.array(hashable_chromosome)
        _interpolation = self.intOfStep(chromosome)
        if _interpolation is not None:
            totalInt_q, totalAngle = _interpolation

            if self.isOutOfRange(totalInt_q):
                collisionScore = 90000000
                msg = "Out of Range!"
                print(msg)
                logging.save_status(msg)
            else:
                if self.isCollision(totalInt_q):
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
            msg = "Out of Range!"
            print(msg)
            logging.save_status(msg)

        score_dist = totalAngle + collisionScore
        points_eachRobot = self.points_count // self.robot_count
        robots_path_len = [
            len(self.robots[rb]) - points_eachRobot for rb in range(self.robot_count)
        ]
        score_unif = np.sum(np.abs(robots_path_len))
        return [score_dist, score_unif]


if __name__ == "__main__":
    folderName = ["210519_2_Result", "210519_Result", "210522_Result"]
    importedPoint = np.genfromtxt("./output_point.csv", delimiter=",")
    chrom = np.genfromtxt(f"./[Result]/{folderName[0]}/Phen.csv", delimiter=",")
    orgPos = np.array([np.radians(90), np.radians(45), np.radians(-45)])
    px = importedPoint[:, 0]
    py = importedPoint[:, 1]
    pz = importedPoint[:, 2]
    axisRange = np.array([[-170, 170], [-30, 120], [-204, 69]])
    fun = ChromoCalcV3()
    test = fun.scoreOfTwoRobot(chrom, 0, orgPos, px, py, pz, axisRange)
