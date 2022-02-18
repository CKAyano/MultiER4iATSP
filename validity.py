from argparse import ArgumentError
from email import header
from enum import Enum, auto
from statistics import mean, stdev
import numpy as np
from chromoCalcV3 import ChromoCalcV3
import robotCalc_Geo3D as rcg
from robotCalc_pygeos import RobotCalc_pygeos
import Geometry3Dmaster.Geometry3D as gm
from pathlib import Path
from typing import List, Optional
from robotInfo import Robot, Config
from status_logging import Collision_status
import os
import imageio
import natsort
import matplotlib.pyplot as plt
import pandas as pd
import datetime


class DrawRobots:
    def __init__(self, folderName: str) -> None:
        self.config = Config(f"./[Result]/{folderName}/config.yml")
        self.points = np.genfromtxt(f"./[Result]/{folderName}/output_point.csv", delimiter=",")
        self.folderName = folderName
        self.config.link_width = self.config.link_width / 2
        self.rc = RobotCalc_pygeos(self.config)
        self.robots_color = ["k", "r", "g", "b"]
        # self.ccv3 = ChromoCalcV3(self.config, self.points, 0, 1)

    def cph(self, point):
        point1 = gm.geometry.Point(point[0][0], point[0][1], point[0][2])
        point2 = gm.geometry.Point(point[1][0], point[1][1], point[1][2])
        point3 = gm.geometry.Point(point[2][0], point[2][1], point[2][2])
        point4 = gm.geometry.Point(point[3][0], point[3][1], point[3][2])
        point5 = gm.geometry.Point(point[4][0], point[4][1], point[4][2])
        point6 = gm.geometry.Point(point[5][0], point[5][1], point[5][2])
        point7 = gm.geometry.Point(point[6][0], point[6][1], point[6][2])
        point8 = gm.geometry.Point(point[7][0], point[7][1], point[7][2])
        point9 = gm.geometry.Point(point[8][0], point[8][1], point[8][2])
        point10 = gm.geometry.Point(point[9][0], point[9][1], point[9][2])
        point11 = gm.geometry.Point(point[10][0], point[10][1], point[10][2])
        point12 = gm.geometry.Point(point[11][0], point[11][1], point[11][2])
        point13 = gm.geometry.Point(point[12][0], point[12][1], point[12][2])
        point14 = gm.geometry.Point(point[13][0], point[13][1], point[13][2])
        point15 = gm.geometry.Point(point[14][0], point[14][1], point[14][2])
        point16 = gm.geometry.Point(point[15][0], point[15][1], point[15][2])
        cpg1 = gm.geometry.ConvexPolygon((point1, point2, point3, point4))
        cpg2 = gm.geometry.ConvexPolygon((point1, point2, point6, point5))
        cpg3 = gm.geometry.ConvexPolygon((point4, point3, point7, point8))
        cpg4 = gm.geometry.ConvexPolygon((point2, point3, point7, point6))
        cpg5 = gm.geometry.ConvexPolygon((point1, point4, point8, point5))
        cpg6 = gm.geometry.ConvexPolygon((point5, point6, point7, point8))
        cpg7 = gm.geometry.ConvexPolygon((point9, point10, point11, point12))
        cpg8 = gm.geometry.ConvexPolygon((point9, point10, point14, point13))
        cpg9 = gm.geometry.ConvexPolygon((point12, point11, point15, point16))
        cpg10 = gm.geometry.ConvexPolygon((point10, point11, point15, point14))
        cpg11 = gm.geometry.ConvexPolygon((point9, point12, point16, point13))
        cpg12 = gm.geometry.ConvexPolygon((point13, point14, point15, point16))
        cph1 = gm.geometry.ConvexPolyhedron((cpg1, cpg2, cpg3, cpg4, cpg5, cpg6))
        cph2 = gm.geometry.ConvexPolyhedron((cpg7, cpg8, cpg9, cpg10, cpg11, cpg12))
        return cph1, cph2

    def cph_robots(self, q_best: List[np.ndarray], intPoint) -> bool:
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1)
        cphs = []
        for rb in range(self.config.robots_count):
            points = self.rc.get_link_points(q_best[rb][intPoint, :], ccv3.robots[rb])
            _cph = self.cph(points)
            cphs.append(_cph)
        return cphs

    def draw(self, q_best: List[np.ndarray], intPoint, save_path: str):
        cphs = self.cph_robots(q_best, intPoint)
        r = gm.render.Renderer()

        for rb in range(self.config.robots_count):
            r.add((cphs[rb][0], self.robots_color[rb], 1), normal_length=0)
            r.add((cphs[rb][1], self.robots_color[rb], 1), normal_length=0)
        # r.show()
        r.savefigure(save_path)

    def chrom_to_png(self, is_log: Optional[bool] = None):
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1)
        chrom = np.genfromtxt(f"./[Result]/{self.folderName}/Chrom.csv", delimiter=",", dtype="int32")
        for chromoInd in range(chrom.shape[0]):
            if is_log:
                Path(f"./ValidityFigure/{self.folderName}/ChromID_{chromoInd}/log").mkdir(
                    parents=True, exist_ok=True
                )
                logging = Collision_status(0, f"./ValidityFigure/{self.folderName}/ChromID_{chromoInd}/log")
            totalInt_q, _ = ccv3.interpolation(chrom[chromoInd, :])
            points_count = np.shape(totalInt_q[0])[0]
            Path(f"./ValidityFigure/{self.folderName}/ChromID_{chromoInd}/figure").mkdir(
                parents=True, exist_ok=True
            )
            for intPoint in range(points_count):
                path = (
                    f"./ValidityFigure/{self.folderName}/" + f"ChromID_{chromoInd}/figure/figure_{intPoint}"
                )
                try:
                    self.draw(totalInt_q, intPoint, path)
                except Exception as e:
                    print(f"can't draw, because {e}")
                    # raise
                    continue
        if is_log:
            _ = ccv3.score_step(chrom[chromoInd, :], logging)

    def png_to_gif(self, fps=24):
        chrom_count = len(os.listdir(f"./ValidityFigure/{self.folderName}"))
        for i in range(chrom_count):
            png_dir = f"./ValidityFigure/{self.folderName}/ChromID_{i}/figure"
            images = []
            png_list = os.listdir(png_dir)
            sortedList = natsort.natsorted(png_list)
            for file_name in sortedList:
                if file_name.endswith(".png"):
                    file_path = os.path.join(png_dir, file_name)
                    images.append(imageio.imread(file_path))
            path = f"./ValidityFigure/{self.folderName}/ChromID_{i}/animate.gif"
            imageio.mimsave(path, images, format="GIF", fps=fps)

    def draw_manuf_route(self, is_connect: Optional[bool] = None):
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1)
        chroms = np.genfromtxt(f"./[Result]/{self.folderName}/Chrom.csv", delimiter=",", dtype="int32")
        chrom_count = len(os.listdir(f"./ValidityFigure/{self.folderName}"))

        px = self.points[:, 0]
        py = self.points[:, 1]
        for c in range(chrom_count):
            chrom = chroms[c, :]
            ccv3.set_robotsPath(chrom)
            for rb in range(self.config.robots_count):
                path_index = ccv3.robots[rb].robot_path - 1
                plt.plot(
                    px[path_index], py[path_index], f"{self.robots_color[rb]}o",
                )

                if is_connect:
                    path_index_shift = np.hstack((path_index[1:], path_index[0]))
                    for i, next_i in zip(path_index, path_index_shift):
                        plt.arrow(
                            px[i],
                            py[i],
                            px[next_i] - px[i],
                            py[next_i] - py[i],
                            head_width=3,
                            length_includes_head=True,
                            color="red",
                        )

            plt.savefig(f"./ValidityFigure/{self.folderName}/ChromID_{c}/manuf_route.png")
            plt.show()


def c_measurement(obj_a_all: np.ndarray, obj_b_all: np.ndarray) -> float:
    # obj_a_count = obj_a_all.shape[0]
    obj_b_count = obj_b_all.shape[0]
    dominate_count = 0
    for obj_b in obj_b_all:
        for obj_a in obj_a_all:
            is_dominate = ChromoCalcV3._dominates(obj_a, obj_b)
            if is_dominate:
                dominate_count += 1
                break
    preferment_value = dominate_count / obj_b_count
    return preferment_value


class DMType(Enum):
    SP = auto()
    OS = auto()
    DM = auto()


def manhattan_distance(sorted_objV) -> List:
    d = []
    for i in range(sorted_objV.shape[0]):
        di = []
        for i_other in range(sorted_objV.shape[0]):
            if i != i_other:
                sum_diff = 0
                for obj_num in range(sorted_objV.shape[1]):
                    temp = np.abs(sorted_objV[i, obj_num] - sorted_objV[i_other, obj_num])
                    sum_diff = sum_diff + temp
                di.append(sum_diff)
        d.append(min(di))
    return d


def spacing_metric(objV):
    objV = sort_obj_value(objV)
    dist = manhattan_distance(objV)
    dist_count = len(dist)
    dist_np = np.array(dist)
    dist_mean = np.mean(dist)
    sp = np.sqrt(np.sum(np.square(dist_np - dist_mean)) / (dist_count - 1))
    return sp


def utopia_point_value(objV) -> List:
    try:
        if objV.shape[1] != 2:
            raise ArgumentError(
                f"number of objectives should be 2 for this method, but now is {objV.shape[1]}"
            )
    except ArgumentError as e:
        print(repr(e))
        raise
    objV = sort_obj_value(objV)
    return [objV[0, 0], objV[-1, 1]]


def nadir_point_value(objV):
    try:
        if objV.shape[1] != 2:
            raise ArgumentError(
                f"number of objectives should be 2 for this method, but now is {objV.shape[1]}"
            )
    except ArgumentError as e:
        print(repr(e))
        raise
    objV = sort_obj_value(objV)
    return [objV[-1, 0], objV[0, 1]]


def overall_pareto_spread(objV, range_obj):
    objV = sort_obj_value(objV)
    f_pb = range_obj[0]
    f_pg = range_obj[1]

    os_h = []
    for obj_num in range(objV.shape[1]):
        os_h.append(
            np.abs(np.max(objV[:, obj_num]) - np.min(objV[:, obj_num]))
            / np.abs(f_pb[obj_num] - f_pg[obj_num])
        )
    os = np.prod(os_h)
    return os


def distribution_metric(objV, range_obj):
    def distance(_sorted_objV):
        d = []
        for obj_num in range(_sorted_objV.shape[1]):
            d.append(np.abs(np.diff(_sorted_objV[:, obj_num])))
        return d

    objV = sort_obj_value(objV)
    d = distance(objV)
    f_pb = range_obj[0]
    f_pg = range_obj[1]
    dm_h = []
    for obj_num in range(objV.shape[1]):
        upper = np.std(d[obj_num]) / np.mean(d[obj_num])
        bottom = np.abs(np.max(objV[:, obj_num]) - np.min(objV[:, obj_num])) / np.abs(
            f_pb[obj_num] - f_pg[obj_num]
        )
        dm_h.append(upper / bottom)
    dm = sum(dm_h) / objV.shape[0]
    return dm


def sort_obj_value(objV):
    objV = np.unique(objV, axis=0)
    objV = objV[objV[:, 0].argsort()]
    return objV


def find_utopia_nadir(*main_folder):
    dir_list = []
    for folder in main_folder:
        dir_list.append(os.listdir(folder))
    utopia_list = []
    nadir_list = []
    for i, dir_folder in enumerate(dir_list):
        for dir in dir_folder:
            objV = np.genfromtxt(f"{main_folder[i]}/{dir}/ObjV.csv", delimiter=",")
            utopia_list.append(utopia_point_value(objV))
            nadir_list.append(nadir_point_value(objV))
    utopia_np = np.array(utopia_list)
    nadir_np = np.array(nadir_list)
    utopia_value = np.min(utopia_np, axis=0)
    nadir_value = np.max(nadir_np, axis=0)
    return utopia_value, nadir_value


def main_distribution_metric(method: DMType):
    noRep_path = "./[Result]/noRep_10000Gen_noSlice"
    repRandom_path = "./[Result]/rep_10000Gen_noSlice"
    repReverse_path = "./[Result]/mode_reverse"

    noRep_folders = os.listdir(noRep_path)
    repRandom_folders = os.listdir(repRandom_path)
    repReverse_folders = os.listdir(repReverse_path)

    if method == DMType.SP:
        metric_method = spacing_metric
    elif method == DMType.OS:
        metric_method = overall_pareto_spread
        utopia_nadir = find_utopia_nadir(noRep_path, repRandom_path, repReverse_path)
    elif method == DMType.DM:
        metric_method = distribution_metric
        utopia_nadir = find_utopia_nadir(noRep_path, repRandom_path, repReverse_path)

    folders = [noRep_folders, repRandom_folders, repReverse_folders]
    folders_path = [noRep_path, repRandom_path, repReverse_path]
    metric_all_folder = []
    metric_mean = []
    metric_std = []
    for i in range(3):
        metric = []
        for dir in folders[i]:
            objV = np.genfromtxt(f"{folders_path[i]}/{dir}/ObjV.csv", delimiter=",")
            if method == DMType.SP:
                metric.append(metric_method(objV))
            else:
                metric.append(metric_method(objV, utopia_nadir))
        metric_all_folder.append(metric)
        metric_mean.append(np.mean(metric))
        metric_std.append(np.std(metric))
    return metric_all_folder, metric_mean, metric_std


def main():
    folderName = "50_points/211111-215630"
    dr = DrawRobots(folderName)
    dr.chrom_to_png()
    dr.png_to_gif()
    dr.draw_manuf_route()


def main_CMeasurement():
    def plot_vs(obj_rep, obj_noRep, value_rep, value_noRep, rep_filename, noRep_filename):
        (p1,) = plt.plot(obj_rep[:, 0], obj_rep[:, 1], "ro")
        (p2,) = plt.plot(obj_noRep[:, 0], obj_noRep[:, 1], "bo")
        plt.xlabel("total angle changes of every robots")
        plt.ylabel("std of every robots' angle changes")
        plt.legend([p1, p2], [f"rep, vs no_rep={value_rep}", f"no_rep, vs rep={value_noRep}"])
        plt.title(f"{rep_filename}(rep) vs {noRep_filename}(noRep)")

    folder_self_path = "./[Result]/noRep_10000Gen_noSlice"
    folder_other_path = "./[Result]/mode_reverse"

    folder_self_list = os.listdir(folder_self_path)
    folder_other_list = os.listdir(folder_other_path)
    datetime_now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    path_output = f"./[Result]/c_measurement/{datetime_now}"
    Path(f"{path_output}/figure").mkdir(parents=True, exist_ok=True)
    for myself in folder_self_list:
        self_vs_other = np.zeros((0, 1))
        other_vs_self = np.zeros((0, 1))
        obj_self_path = f"{folder_self_path}/{myself}/ObjV.csv"
        obj_self = np.genfromtxt(obj_self_path, delimiter=",")
        obj_self = np.unique(obj_self, axis=0)
        # if obj_noRep.shape[0] >= 20:
        for other in folder_other_list:
            obj_other_path = f"{folder_other_path}/{other}/ObjV.csv"
            obj_other = np.genfromtxt(obj_other_path, delimiter=",")
            obj_other = np.unique(obj_other, axis=0)
            # if obj_rep.shape[0] >= 20:
            c_value_self = c_measurement(obj_self, obj_other)
            c_value_other = c_measurement(obj_other, obj_self)
            self_vs_other = np.vstack((self_vs_other, c_value_self))
            other_vs_self = np.vstack((other_vs_self, c_value_other))

            plot_vs(obj_other, obj_self, c_value_other, c_value_self, other, myself)
            plt.savefig(f"{path_output}/figure/{other}_vs_{myself}.png")
            plt.close()
            plt.cla()
            plt.clf()
    self_vs_other_mean = np.mean(self_vs_other)
    self_vs_other_std = np.std(self_vs_other)
    other_vs_self_mean = np.mean(other_vs_self)
    other_vs_self_std = np.std(other_vs_self)
    mean_std = np.array([[self_vs_other_mean, other_vs_self_mean], [self_vs_other_std, other_vs_self_std]])
    fileoutput = np.hstack((self_vs_other, other_vs_self))
    fileoutput = np.vstack((fileoutput, mean_std))
    np.savetxt(f"{path_output}/result.csv", fileoutput, delimiter=",")


def main_metric_compare():
    method_list = [DMType.SP, DMType.OS, DMType.DM]
    sheet_name = ["SP", "OS", "DM"]
    writer = pd.ExcelWriter(
        f"./[Result]/metrics_compare/metrics{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}.xlsx",
        engine="xlsxwriter",
    )
    for i, md in enumerate(method_list):
        res = main_distribution_metric(md)
        print(res[1], res[2])
        df = pd.DataFrame(res[0]).T
        df.columns = ["noRep", "rep_random", "rep_reverse"]
        df.to_excel(writer, sheet_name=sheet_name[i])
        # print(res)
    writer.save()


if __name__ == "__main__":
    # main()
    main_CMeasurement()
    # objV = np.genfromtxt("./[Result]/mode_reverse/220128-174038/ObjV.csv", delimiter=",")
    # overall_pareto_spread(objV)
    # main_metric_compare()
