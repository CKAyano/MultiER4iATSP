from chromoCalcV3 import ChromoCalcV3
import numpy as np
import geatpy as ea
from status_logging import Collision_status
import datetime
import yaml


class Problem_config:
    def __init__(self, filepath) -> None:
        self.read_config(filepath)

    def read_config(self, filepath):
        with open(filepath, "r") as config_file:
            self.config = yaml.load(config_file)


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, step, num_slicing, config_path, M=2):
        config = Problem_config(config_path).config

        robot_count = config["robot_count"]
        points_range = config["points_range"]
        linkWidth = config["linkWidth"]

        points = np.genfromtxt("output_point.csv", delimiter=",")
        points_count = points.shape[0]

        name = "MyProblem"  # 初始化name（函数名称，可以随意设置）
        Dim = points_count + robot_count - 1  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [-(robot_count - 2)] * Dim  # 决策变量下界
        ub = [points_count] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.step = step
        if step == 0:
            self.isFirstImp = True
            file = open(f"./Result/info.txt", "w")
            file.write(
                f'{"start time:":<25}\
                {datetime.datetime.now().strftime("%y%m%d-%H%M%S")}\n'
            )
            file.write(f'{"points range:":<25}{points_range}\n')
            file.write(f'{"number of points:":<25}{points_count}\n')
            file.write(f'{"link width:":<25}{linkWidth}\n')
            file.close()
        else:
            self.isFirstImp = False
        self.logging = Collision_status(self.step)
        self.ccv3 = ChromoCalcV3(config, points, step, num_slicing)
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        # population = pop.Chrom
        nind = pop.sizes

        score_dist = np.ones((nind, 1))
        score_unif = np.ones((nind, 1))
        for chromoIndex in range(nind):
            # aim1 總加工時間最短
            # chromosome = population[chromoIndex, :]
            self.ccv3.chromoIndex = chromoIndex
            if self.isFirstImp:
                self.ccv3.adjChromo(pop.Chrom[chromoIndex, :], pop)
            score_all = self.ccv3.scoreOfTwoRobot_step(
                pop.Chrom[chromoIndex, :], self.logging
            )
            score_dist[chromoIndex] = score_all[0]
            # aim2 手臂點分佈最平均
            score_unif[chromoIndex] = score_all[1]
            # print(f'\t{ccv3.scoreOfTwoRobot.cache_info()}')
        self.isFirstImp = False
        pop.CV = score_dist - 1000000
        pop.ObjV = np.hstack([score_dist, score_unif])


if __name__ == "__main__":
    config = Problem_config("./config.yml")
    print(config.config)
