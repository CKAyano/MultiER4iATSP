import numpy as np
from numpy.lib.function_base import append
from robotCalc_pygeos import RobotCalc_pygeos, Coord
from robotInfo import Robot, Position

# from caching import np_cache


class ChromoCalcV3:
    def __init__(self, config, points, step, num_slicing):
        self.points_range = config["points_range"]
        self.baseX_offset = sum(self.points_range[0])
        self.linkWidth = config["linkWidth"]
        self.orgPos = np.array(np.radians(config["originalPosture"]))
        self.axisRange = np.array(config["axisRange"])
        self.direct_array = np.array(config["direct_array"])
        self.rc = RobotCalc_pygeos(self.baseX_offset, self.linkWidth)
        self.px = points[:, 0]
        self.py = points[:, 1]
        self.pz = points[:, 2]
        self.outRange = False
        self.allOneSide = False
        self.chromoIndex = None
        self.middle_x = self.baseX_offset / 2
        self.step = step
        self.num_slicing = num_slicing
        self.robots = []
        position = [Position.LEFT, Position.RIGHT, Position.UP, Position.DOWN]
        for i in range(config["robot_count"]):
            self.robots = append(Robot(i + 1, position[i]))

    def adjChromo(self, chromosome, pop):
        def needPreAdj(pointIndex: int, whichRobot: str):
            vv_robot = Coord(
                self.px[pointIndex], self.py[pointIndex], self.pz[pointIndex]
            )
            try:
                if whichRobot == "b":
                    vv_robot = self.rc.robot2world(vv_robot)
                elif whichRobot == "a":
                    pass
                else:
                    raise RuntimeError("There's only 'a' or 'b' for 'whichRobot'")
            except RuntimeError as e:
                print(repr(e))
                raise
            q_robot = self.rc.userIK(vv_robot, self.direct_array)
            isOutRange = []
            for i in range(q_robot.shape[0]):
                isOutRange.append(self.rc.cvAxisRange(q_robot[i, :], self.axisRange))
            if np.all(isOutRange):
                return True
            return False

        zeroGeneIndex = np.where(chromosome == 0)
        zeroGeneIndex = zeroGeneIndex[0][0]
        robotPath_a = chromosome[:(zeroGeneIndex)]
        robotPath_b = chromosome[(zeroGeneIndex + 1) :]
        pointIndex_a = robotPath_a - 1
        pointIndex_b = robotPath_b - 1
        robotPath_a_needMove = np.zeros(0, dtype=int)
        robotPath_b_needMove = np.zeros(0, dtype=int)
        for i in range(len(pointIndex_a)):
            if needPreAdj(pointIndex_a[i], "a"):
                robotPath_a_needMove = np.hstack((robotPath_a_needMove, i))
        for i in range(len(pointIndex_b)):
            if needPreAdj(pointIndex_b[i], "b"):
                robotPath_b_needMove = np.hstack((robotPath_b_needMove, i))
        robotPath_a_after = robotPath_a.copy()
        robotPath_b_after = robotPath_b.copy()
        robotPath_a_after = np.delete(robotPath_a_after, robotPath_a_needMove)
        robotPath_b_after = np.delete(robotPath_b_after, robotPath_b_needMove)
        robotPath_a_after = np.hstack(
            (robotPath_a_after, robotPath_b[robotPath_b_needMove])
        )
        robotPath_b_after = np.hstack(
            (robotPath_b_after, robotPath_a[robotPath_a_needMove])
        )
        pop.Phen[self.chromoIndex, :] = np.hstack(
            (robotPath_a_after, 0, robotPath_b_after)
        )
        pop.Chrom[self.chromoIndex, :] = np.hstack(
            (robotPath_a_after, 0, robotPath_b_after)
        )

    def splitChromo(self, chromosome):
        if chromosome.ndim >= 2:
            chromosome = np.squeeze(chromosome)
        # chromosome = population[chromoIndex, :]  # [1, 2, 3, 4, 0, 5, 6]
        zeroGeneIndex = np.where(chromosome == 0)  # index of 0 = 4
        zeroGeneIndex = zeroGeneIndex[0][0]
        robotPath_a = chromosome[:(zeroGeneIndex)]  # [1, 2, 3, 4]
        robotPath_b = chromosome[(zeroGeneIndex + 1) :]  # [5, 6]
        pointIndex_a = robotPath_a - 1  # [0, 1, 2, 3]
        pointIndex_b = robotPath_b - 1  # [4, 5]
        if len(pointIndex_a) > len(pointIndex_b):
            appendArray = np.ones(len(pointIndex_a) - len(pointIndex_b)) * -1
            pointIndex_b = np.hstack((pointIndex_b, appendArray))  # [4, 5, -1, -1]
        elif len(pointIndex_b) > len(pointIndex_a):
            appendArray = np.ones(len(pointIndex_b) - len(pointIndex_a)) * -1
            pointIndex_a = np.hstack((pointIndex_a, appendArray))  # [0, 1, 2, 3]

        appendArray = np.array([-1])
        pointIndex_a = np.hstack((pointIndex_a, appendArray))
        pointIndex_b = np.hstack((pointIndex_b, appendArray))

        pointIndex_a = pointIndex_a.astype(int)  # [0, 1, 2, 3, -1]
        pointIndex_b = pointIndex_b.astype(int)  # [4, 5, -1, -1, -1]
        return [pointIndex_a, pointIndex_b, robotPath_a, robotPath_b]

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

    def coor2OptiAngle(self, vv: Coord, q_1_best, whichRobot: str):
        try:
            if whichRobot == "b":
                vv = self.rc.robot2world(vv)
            elif whichRobot == "a":
                pass
            else:
                raise RuntimeError("There's only 'a' or 'b' for 'whichRobot'")
        except RuntimeError as e:
            print(repr(e))
            raise
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

    def angleOffset(self, loc_a, loc_b, qa_1_best, qb_1_best):
        self.outRange = False
        if loc_a == -1:
            qa_2_best = self.orgPos
        else:
            vv_a = Coord(self.px[loc_a], self.py[loc_a], self.pz[loc_a])
            qa_2_best = self.coor2OptiAngle(vv_a, qa_1_best, "a")
            isQaNan = np.isnan(qa_2_best)
            if np.any(isQaNan):
                self.outRange = True
                return

        if loc_b == -1:
            qb_2_best = self.orgPos
        else:
            vv_b = Coord(self.px[loc_b], self.py[loc_b], self.pz[loc_b])
            qb_2_best = self.coor2OptiAngle(vv_b, qb_1_best, "b")
            isQbNan = np.isnan(qb_2_best)
            if np.any(isQbNan):
                self.outRange = True
                return
        angleOffset_a = np.abs(qa_2_best - qa_1_best)
        angleOffset_b = np.abs(qb_2_best - qb_1_best)
        angleOffset_a = np.degrees(angleOffset_a)
        angleOffset_b = np.degrees(angleOffset_b)
        return [angleOffset_a, angleOffset_b, qa_2_best, qb_2_best]

    def genTwoRobotInt(self, qa_1_best, qb_1_best, loc_a, loc_b):
        _angleOffset = self.angleOffset(loc_a, loc_b, qa_1_best, qb_1_best)
        if self.outRange:
            return
        angOffset_a, angOffset_b = _angleOffset[0], _angleOffset[1]
        qa_2_best, qb_2_best = _angleOffset[2], _angleOffset[3]
        seqCheckNum_a = int(np.max(angOffset_a))
        seqCheckNum_b = int(np.max(angOffset_b))
        int_qa = self.intForOneSeq(seqCheckNum_a, qa_1_best, qa_2_best)
        int_qb = self.intForOneSeq(seqCheckNum_b, qb_1_best, qb_2_best)
        return [int_qa, int_qb, qa_2_best, qb_2_best, angOffset_a, angOffset_b]

    def interpolation(self, chromosome):
        pointIndex = self.splitChromo(chromosome)

        isFirstLoop = True
        totalAngle = 0
        numOfJoints = self.orgPos.size
        totalInt_qa, totalInt_qb = np.zeros((0, numOfJoints)), np.zeros((0, numOfJoints))
        for loc_a, loc_b in zip(pointIndex[0], pointIndex[1]):
            if isFirstLoop:
                qa_1_best = self.orgPos
                qb_1_best = self.orgPos
                isFirstLoop = False
            else:
                qa_1_best = qa_2_best
                qb_1_best = qb_2_best
            _genTwoRobotInt = self.genTwoRobotInt(qa_1_best, qb_1_best, loc_a, loc_b)
            if self.outRange:
                return pointIndex
            int_qa, int_qb = _genTwoRobotInt[0], _genTwoRobotInt[1]
            qa_2_best, qb_2_best = _genTwoRobotInt[2], _genTwoRobotInt[3]
            angOffset_a, angOffset_b = _genTwoRobotInt[4], _genTwoRobotInt[5]
            totalInt_qa = np.vstack((totalInt_qa, int_qa))
            totalInt_qb = np.vstack((totalInt_qb, int_qb))
            totalAngle = totalAngle + angOffset_a + angOffset_b  # 兩隻手臂相加

        totalInt_qa = np.delete(
            totalInt_qa, np.where(np.all(totalInt_qa == self.orgPos, axis=1)), axis=0
        )
        totalInt_qb = np.delete(
            totalInt_qb, np.where(np.all(totalInt_qb == self.orgPos, axis=1)), axis=0
        )
        numsOfIntPoint_a = np.shape(totalInt_qa)[0]
        numsOfIntPoint_b = np.shape(totalInt_qb)[0]
        if numsOfIntPoint_a > numsOfIntPoint_b:
            totalInt_qa = totalInt_qa[0:numsOfIntPoint_b, :]
        if numsOfIntPoint_b > numsOfIntPoint_a:
            totalInt_qb = totalInt_qb[0:numsOfIntPoint_a, :]
        return [totalInt_qa, totalInt_qb, totalAngle, pointIndex]

    def slicingCheckPoint(self, totalInt_qa, totalInt_qb, levelOfSlicing, preInd):
        self.allOneSide = False
        numOfSlicing = 0
        numOfInt = totalInt_qa.shape[0]
        if numOfInt == 0:
            self.allOneSide = True
            return
        for i in range(levelOfSlicing):
            numOfSlicing = numOfSlicing + np.power(2, i)
        spacing = numOfInt // (numOfSlicing + 1)
        slicingInd = np.arange(0, numOfInt, spacing)
        lastEleFlag = numOfInt % (numOfSlicing + 1)
        if lastEleFlag == 0 and levelOfSlicing == 1:
            slicingInd = np.hstack((slicingInd, numOfInt - 1))
        checkSlicing = slicingInd.copy()
        for i in preInd:
            slicingInd = np.delete(slicingInd, np.where(slicingInd == i))
        if np.all(checkSlicing == np.arange(numOfInt)):
            return (
                totalInt_qa[slicingInd, :],
                totalInt_qb[slicingInd, :],
                True,
                slicingInd,
            )
        return totalInt_qa[slicingInd, :], totalInt_qb[slicingInd, :], False, slicingInd

    def intOfStep(self, chromosome):
        _interpolation = self.interpolation(chromosome)
        if self.outRange:
            return _interpolation
        [totalInt_qa, totalInt_qb, totalAngle, pointIndex] = _interpolation
        num_int = totalInt_qa.shape[0]
        len_path = np.min([len(pointIndex[2]), len(pointIndex[3])])
        if self.step + 1 == self.num_slicing:
            return _interpolation
        split_num = num_int // len_path // (self.step + 1)
        if split_num == 0:
            split_num = 1
        return [
            totalInt_qa[::split_num, :],
            totalInt_qb[::split_num, :],
            totalAngle,
            pointIndex,
        ]

    def isOutOfRange(self, totalInt_qa, totalInt_qb):
        isOutOfRange_qa = self.rc.cvAxisRange(totalInt_qa, self.axisRange)
        isOutOfRange_qb = self.rc.cvAxisRange(totalInt_qb, self.axisRange)
        if isOutOfRange_qa or isOutOfRange_qb:
            return True
        return False

    def isCollision(self, totalInt_qa, totalInt_qb):
        numsOfIntPoint = np.shape(totalInt_qa)[0]
        for i in range(numsOfIntPoint):
            if self.rc.cvCollision(totalInt_qa[i, :], totalInt_qb[i, :]):
                return True
        return False

    # @np_cache
    def scoreOfTwoRobot(self, chromosome, logging):
        # chromosome = np.array(hashable_chromosome)
        _interpolation = self.interpolation(chromosome)
        if not self.outRange:
            totalInt_qa, totalInt_qb = _interpolation[0], _interpolation[1]
            totalAngle = np.max(_interpolation[2])
            pointIndex = _interpolation[3]

            if self.isOutOfRange(totalInt_qa, totalInt_qb):
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
                        totalInt_qa, totalInt_qb, levelOfSlicing, preInd
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
            pointIndex = _interpolation
            totalAngle = 0
            collisionScore = 90000000
            msg = "Out of Range!"
            print(msg)
            logging.save_status(msg)

        score_dist = totalAngle + collisionScore
        score_unif = np.abs(len(pointIndex[2]) - len(pointIndex[3]))
        return [score_dist, score_unif]

    def scoreOfTwoRobot_step(self, chromosome, logging):
        # chromosome = np.array(hashable_chromosome)
        _interpolation = self.intOfStep(chromosome)
        if not self.outRange:
            totalInt_qa, totalInt_qb = _interpolation[0], _interpolation[1]
            totalAngle = np.max(_interpolation[2])
            pointIndex = _interpolation[3]

            if self.isOutOfRange(totalInt_qa, totalInt_qb):
                collisionScore = 90000000
                msg = "Out of Range!"
                print(msg)
                logging.save_status(msg)
            else:
                if self.isCollision(totalInt_qa, totalInt_qb):
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
            pointIndex = _interpolation
            totalAngle = 0
            collisionScore = 90000000
            msg = "Out of Range!"
            print(msg)
            logging.save_status(msg)

        score_dist = totalAngle + collisionScore
        score_unif = np.abs(len(pointIndex[2]) - len(pointIndex[3]))
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
