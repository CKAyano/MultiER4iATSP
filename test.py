from chromoCalcV3 import Trajectory
from robotCalc_pygeos import RobotCalc_pygeos
import numpy as np
from robot_configuration import PumaKinematics

from robot_configuration import Config, Coord


def main() -> None:
    angle_start = np.radians(np.array([20, 30, 40]))
    print(angle_start)
    angle_end = np.radians(np.array([40, 60, 80]))
    print(angle_end)

    mean_ang_v = np.radians(100)
    # print(f"{mean_ang_v:.4f}")
    _, s_all = Trajectory.get_trajectory(
        angle_start, angle_end, mean_ang_v, step_period=1 / 20, is_test=True
    )
    s_all_out = s_all[1:, :]
    print(s_all)


def ik_test():
    config = Config("./CONFIG.yml")
    rb = RobotCalc_pygeos(config)
    p = Coord(300, 20, 0)
    # [[   3.8141   28.0415   28.0358   -0.       33.9227  176.1859]
    #  [   3.8141   28.0415   28.0358  180.      -33.9227   -3.8141]
    #  [   3.8141  151.9585  159.8545   -0.      138.1869  176.1859]
    #  [   3.8141  151.9585  159.8545  180.     -138.1869   -3.8141]
    #  [-176.1859 -151.9585   28.0358 -180.      146.0773  176.1859]
    #  [-176.1859 -151.9585   28.0358    0.     -146.0773   -3.8141]
    #  [-176.1859  -28.0415  159.8545 -180.       41.8131  176.1859]
    #  [-176.1859  -28.0415  159.8545    0.      -41.8131   -3.8141]]
    ik = np.round(np.degrees(rb.userIK(p)), 4)
    print(ik)


def puma_test():
    puma = PumaKinematics()
    # p = Coord(600, 149.09, 200)
    # p = Coord(500, 240, 230)
    # p = Coord(540, 210, 260)
    # p = Coord(180, -400, 400)
    p = Coord(-180, 400, -200)

    puma._validate(p, [0, 0, -3.14159265])


if __name__ == "__main__":
    # main()
    # ik_test()
    puma_test()
