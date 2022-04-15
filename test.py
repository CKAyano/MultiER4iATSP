from chromoCalcV3 import Trajectory
import numpy as np


def main() -> None:
    angle_start = np.radians(np.array([20, 30, 40]))
    print(angle_start)
    angle_end = np.radians(np.array([40, 60, 80]))
    print(angle_end)

    mean_ang_v = np.radians(100)
    # print(f"{mean_ang_v:.4f}")
    s_all = Trajectory.get_trajectory(angle_start, angle_end, mean_ang_v, step_freq=1 / 20, is_test=True)
    print(s_all)


if __name__ == "__main__":
    main()
