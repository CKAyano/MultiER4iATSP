import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(prog=".randomize_manuf_points.py", description="generate random points")
    parser.add_argument("--nums", "-n", default="100", type=int, required=True, help="number of points")
    parser.add_argument("--draw", "-d", default="True", type=str2bool, required=False, help="draw points?")
    parser.add_argument("--save", "-s", default="True", type=str2bool, required=False, help="save points?")

    return parser.parse_args()


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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
    set_axes_equal(ax)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    # points_count = 100
    points_count = args.nums
    is_draw = args.draw
    is_save = args.save
    points = get_random_points(points_count)
    if is_draw:
        draw_manuf_points(points)
    if is_save:
        np.savetxt("./output_point.csv", points, delimiter=",")
