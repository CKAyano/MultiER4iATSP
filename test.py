from chromoCalcV3 import ChromoCalcV3
from ga_Problem import Problem_config
import numpy as np

config = Problem_config("./config.yml").config
points = np.genfromtxt("output_point.csv", delimiter=",")
ccv3 = ChromoCalcV3(config, points, 0, 1)
chromo = np.array([2, 3, 4, -1, 1, 0, 5, 6, -2])
ccv3.set_robotsPath(chromo)
print(ccv3.robots)
