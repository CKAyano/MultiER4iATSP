import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _randomize_from_range(points_count, lb, ub):
    first_half_points_count = int(points_count / 2)
    leftover_points_count = points_count - first_half_points_count
    mid_bound = (lb + ub) / 2
    p_1 = lb + np.random.rand(first_half_points_count) * (mid_bound - lb)
    p_2 = mid_bound + np.random.rand(leftover_points_count) * (ub - mid_bound)
    p = np.append(p_1, p_2)
    np.random.shuffle(p)
    return p[:, None]


def get_random_points(points_count) -> None:
    with open("./CONFIG.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    points_range = config["points_range"]
    px_lb = points_range[0][0]
    px_ub = points_range[0][1]
    py_lb = points_range[1][0]
    py_ub = points_range[1][1]
    pz_lb = points_range[2][0]
    pz_ub = points_range[2][1]

    px = _randomize_from_range(points_count, px_lb, px_ub)
    py = _randomize_from_range(points_count, py_lb, py_ub)
    pz = _randomize_from_range(points_count, pz_lb, pz_ub)
    points = np.hstack((px, py, pz))
    return points


def draw_manuf_points(points):
    fig = plt.figure()
    ax = Axes3D(fig)
    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]
    ax.plot(
        px, py, pz, linestyle="", marker="o", markerfacecolor="orangered", markersize="10",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    points_counts = 100
    points = get_random_points(points_counts)
    draw_manuf_points(points)
    np.savetxt("./output_point.csv")
