from chromoCalcV3 import ChromoCalcV3
import numpy as np
import geatpy as ea
from status_logging import Collision_status
import datetime
import yaml
from robotInfo import Config


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, step, num_slicing, config_path, M=2):
        config = Config(config_path)

        robots_count = config.robots_count
        points_range = config.points_range
        link_width = config.link_width

        points = np.genfromtxt("output_point.csv", delimiter=",")
        points_count = points.shape[0]

        name = "MyProblem"  # 初始化name（函数名称，可以随意设置）
        Dim = points_count + robots_count - 1  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [-(robots_count - 2)] * Dim  # 决策变量下界
        ub = [points_count] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.step = step
        if step == 0:
            self.is_firstImp = True
            file = open(f"./Result/info.txt", "w")
            file.write(
                f'{"start time:":<25}'
                + f'{datetime.datetime.now().strftime("%y%m%d-%H%M%S")}\n'
            )
            file.write(f'{"number of robots:":<25}{robots_count}\n')
            file.write(f'{"points range:":<25}{points_range}\n')
            file.write(f'{"number of points:":<25}{points_count}\n')
            file.write(f'{"link width:":<25}{link_width}\n')
            file.close()
        else:
            self.is_firstImp = False
        self.logging = Collision_status(self.step)
        self.ccv3 = ChromoCalcV3(config, points, step, num_slicing)
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        # population = pop.Chrom
        nind = pop.sizes

        score_dist = np.ones((nind, 1))
        score_unif = np.ones((nind, 1))
        for chromo_index in range(nind):
            # aim1 總加工時間最短
            # chromosome = population[chromoIndex, :]
            self.ccv3.chromo_index = chromo_index
            if self.is_firstImp:
                self.ccv3.adj_chromo(pop.Chrom[chromo_index, :], pop)
            score_all = self.ccv3.score_step(pop.Chrom[chromo_index, :], self.logging)
            score_dist[chromo_index] = score_all[0]
            # aim2 手臂點分佈最平均
            score_unif[chromo_index] = score_all[1]
            # print(f'\t{ccv3.scoreOfTwoRobot.cache_info()}')
        self.is_firstImp = False
        pop.CV = np.hstack((score_dist - 1000000, score_unif - 10000))
        pop.ObjV = np.hstack((score_dist, score_unif))
