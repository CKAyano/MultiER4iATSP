from chromoCalcV3 import Trajectory
from robotCalc_pygeos import RobotCalc_pygeos
import numpy as np

from robotInfo import Config, Coord


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
    ik = rb.userIK(p)
    print(ik)


if __name__ == "__main__":
    # main()
    ik_test()
