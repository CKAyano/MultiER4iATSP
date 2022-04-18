from chromoCalcV3 import ChromoCalcV3
import numpy as np
import geatpy as ea
from status_logging import Collision_status
import datetime
import yaml
from robotInfo import Config
from pathlib import Path


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, step, gen_step_count, config_path, feasibleSol_list, M=2):
        config = Config(config_path)

        robots_count = config.robots_count
        replace_chromo = config.replace_chromo
        adj_chromo = config.adj_chromo
        custom_initChrom = config.custom_initChrom
        points_range = config.points_range
        link_width = config.link_width
        interp_mode = config.interp_mode
        interp_step_freq = config.interp_step_freq
        mean_motion_velocity_deg = np.degrees(config.mean_motion_velocity_rad)

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
        Path("./Result/Chrom_per1000").mkdir(parents=True, exist_ok=True)
        if step == 0:
            self.is_firstImp = True
            with open("./Result/info.txt", "w") as file:
                # file = open(f"./Result/info.txt", "w")
                file.write(f'{"start time:":<25}' + f'{datetime.datetime.now().strftime("%y%m%d-%H%M%S")}\n')
                file.write(f'{"number of robots:":<25}{robots_count}\n')
                file.write(f'{"replace chromosome:":<25}{replace_chromo}\n')
                file.write(f'{"adjust chromosome:":<25}{adj_chromo}\n')
                file.write(f'{"fix init chromosome:":<25}{custom_initChrom}\n')
                file.write(f'{"points range:":<25}{points_range}\n')
                file.write(f'{"number of points:":<25}{points_count}\n')
                file.write(f'{"link width:":<25}{link_width}\n')
                file.write(f'{"interp mode:":<25}{interp_mode}\n')
                file.write(f'{"interp step freq:":<25}{interp_step_freq}\n')
                file.write(f'{"mean motion velocity:":<25}{mean_motion_velocity_deg}\n')

            # file.close()
        else:
            self.is_firstImp = False
        self.logging = Collision_status(self.step)
        self.ccv3 = ChromoCalcV3(config, points, step, gen_step_count, feasibleSol_list)
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        # population = pop.Chrom
        self.ccv3.feasibleSol_count = 0
        nind = pop.sizes

        score_dist = np.ones((nind, 1))
        score_unif = np.ones((nind, 1))
        for chromo_id in range(nind):
            # aim1 總加工時間最短
            if self.is_firstImp:
                if self.ccv3.config.adj_chromo:
                    self.ccv3.adj_chromo(pop.Chrom[chromo_id, :], chromo_id, pop)
            score_all = self.ccv3.score_step(pop.Chrom[chromo_id, :], self.logging)
            score_dist[chromo_id] = score_all[0]
            # aim2 手臂點分佈最平均
            score_unif[chromo_id] = score_all[1]
            # print(f'\t{ccv3.scoreOfTwoRobot.cache_info()}')
        self.is_firstImp = False
        self.ccv3.feasibleSol_list.append(self.ccv3.feasibleSol_count)
        pop.CV = np.hstack((score_dist - 1000000, score_unif - 10000))
        pop.ObjV = np.hstack((score_dist, score_unif))
