from argparse import ArgumentError
from email import header
from enum import Enum, auto
from operator import delitem
from statistics import mean, stdev
import numpy as np
from chromoCalcV3 import ChromoCalcV3
import robotCalc_Geo3D as rcg
from robotCalc_pygeos import RobotCalc_pygeos
import Geometry3Dmaster.Geometry3D as gm
from pathlib import Path
from typing import List, Optional
from robotInfo import Coord, Position, Robot, Config
from status_logging import Collision_status
import os
import imageio
import natsort
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk, filedialog
import tkfilebrowser as tkf
from mpl_toolkits.mplot3d import Axes3D


class DrawRobots:
    def __init__(self, folderName: str) -> None:
        if os.path.exists(f"{folderName}/config.yml"):
            self.config = Config(f"{folderName}/config.yml")
        else:
            self.config = Config(f"{folderName}/CONFIG.yml")
        self.points = np.genfromtxt(f"{folderName}/output_point.csv", delimiter=",")
        self.folderName = folderName
        self.config.link_width = self.config.link_width / 2
        self.rc = RobotCalc_pygeos(self.config)
        self.robots_color = ["k", "royalblue", "r", "g"]
        # self.ccv3 = ChromoCalcV3(self.config, self.points, 0, 1)
        self.datetime_now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

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

    def _cph_vs_seg_robots(self, q_best, int_point):
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1, [])
        cphs = []
        # for rb in range(self.config.robots_count):
        points = self.rc.get_link_points(q_best[0][int_point, :], ccv3.robots[0])
        cph1, cph2 = self.cph(points)

        vb_all = self.rc.userFK(q_best[1][int_point, :])
        vb_all = self.rc.robot2world_v_all(vb_all, ccv3.robots[1].position)

        gm_vb1 = gm.geometry.Point(vb_all.v2.xx, vb_all.v2.yy, vb_all.v2.zz)
        gm_vb3 = gm.geometry.Point(vb_all.v4.xx, vb_all.v4.yy, vb_all.v4.zz)
        gm_vb4 = gm.geometry.Point(vb_all.v5.xx, vb_all.v5.yy, vb_all.v5.zz)

        gmSegB1 = gm.geometry.Segment(gm_vb1, gm_vb3)
        gmSegB2 = gm.geometry.Segment(gm_vb3, gm_vb4)
        # cphs.append(_cph)
        return cph1, cph2, gmSegB1, gmSegB2

    def cph_robots(self, q_best: List[np.ndarray], intPoint) -> bool:
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1, [])
        cphs = []
        for rb in range(self.config.robots_count):
            points = self.rc.get_link_points(q_best[rb][intPoint, :], ccv3.robots[rb])
            _cph = self.cph(points)
            cphs.append(_cph)
        return cphs

    def _draw_cph_vs_seg(
        self, q_best: List[np.ndarray], intPoint, save_path: Optional[str] = None, axis=None, is_show=True
    ):
        cphs = self._cph_vs_seg_robots(q_best, intPoint)

        inter1 = gm.calc.intersection(cphs[0], cphs[2])
        inter2 = gm.calc.intersection(cphs[0], cphs[3])
        inter3 = gm.calc.intersection(cphs[1], cphs[2])
        inter4 = gm.calc.intersection(cphs[1], cphs[3])

        strInter1 = str(inter1)
        strInter2 = str(inter2)
        strInter3 = str(inter3)
        strInter4 = str(inter4)

        r = gm.render.Renderer()

        # for rb in range(self.config.robots_count):
        r.add((cphs[0], self.robots_color[0], 2), normal_length=0)
        r.add((cphs[1], self.robots_color[0], 2), normal_length=0)

        r.add((cphs[2], self.robots_color[1], 2), normal_length=0)
        r.add((cphs[3], self.robots_color[1], 2), normal_length=0)

        color_inter = "darkorange"
        if strInter1 == "None":
            pass
        else:
            r.add((inter1, color_inter, 8), normal_length=0)
        if strInter2 == "None":
            pass
        else:
            r.add((inter2, color_inter, 8), normal_length=0)
        if strInter3 == "None":
            pass
        else:
            r.add((inter3, color_inter, 8), normal_length=0)
        if strInter4 == "None":
            pass
        else:
            r.add((inter4, color_inter, 5), normal_length=0)

        if is_show:
            r.show(axis=axis, dpi=200)
        if save_path:
            r.savefigure(save_path, axis)

    def draw(self, q_best: List[np.ndarray], intPoint, save_path=None, axis=None, is_show=True):
        cphs = self.cph_robots(q_best, intPoint)
        r = gm.render.Renderer()

        for rb in range(self.config.robots_count):
            r.add((cphs[rb][0], self.robots_color[rb], 3), normal_length=0)
            r.add((cphs[rb][1], self.robots_color[rb], 3), normal_length=0)
        if is_show:
            r.show(dpi=200, axis=axis)
        if save_path:
            r.savefigure(save_path, axis)

    def chrom_to_png(self, is_log: Optional[bool] = None, axis=None):
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1, [])
        chrom = np.genfromtxt(f"{self.folderName}/Chrom.csv", delimiter=",", dtype="int32")
        for chromoInd in range(chrom.shape[0]):
            if is_log:
                Path(f"./ValidityFigure/{self.folderName[11:]}/ChromID_{chromoInd}/log").mkdir(
                    parents=True, exist_ok=True
                )
                logging = Collision_status(
                    0, f"./ValidityFigure/{self.folderName[11:]}/ChromID_{chromoInd}/log"
                )
            totalInt_q, _ = ccv3.interpolation(chrom[chromoInd, :])
            points_count = np.shape(totalInt_q[0])[0]
            Path(f"./ValidityFigure/{self.folderName[11:]}/ChromID_{chromoInd}/figure").mkdir(
                parents=True, exist_ok=True
            )
            for intPoint in range(points_count):
                path = (
                    f"./ValidityFigure/{self.folderName[11:]}/"
                    + f"ChromID_{chromoInd}/figure/figure_{intPoint}"
                )
                try:
                    self.draw(totalInt_q, intPoint, path, axis)
                except Exception as e:
                    print(f"can't draw, because {e}")
                    # raise
                    continue
        if is_log:
            _ = ccv3.score_step(chrom[chromoInd, :], logging)

    def png_to_gif(self, fps=24):
        chrom_count = len(os.listdir(f"./ValidityFigure/{self.folderName[11:]}"))
        for i in range(chrom_count):
            png_dir = f"./ValidityFigure/{self.folderName[11:]}/ChromID_{i}/figure"
            images = []
            png_list = os.listdir(png_dir)
            sortedList = natsort.natsorted(png_list)
            for file_name in sortedList:
                if file_name.endswith(".png"):
                    file_path = os.path.join(png_dir, file_name)
                    images.append(imageio.imread(file_path))
            path = f"./ValidityFigure/{self.folderName[11:]}/ChromID_{i}/animate.gif"
            imageio.mimsave(path, images, format="GIF", fps=fps)

    def draw_manuf_dist(self):
        points = np.genfromtxt(f"{self.folderName}/output_point.csv", delimiter=",")
        fig = plt.figure(dpi=200)
        ax = Axes3D(fig, elev=60)
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            linestyle="",
            marker="o",
            markerfacecolor="orangered",
            markersize="10",
        )
        # ax.set_xlim3d((0, 900))
        # ax.set_ylim3d((-450, 450))
        # ax.set_zlim3d((45, 55))
        self.set_axes_equal(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def draw_manuf_route(self, is_connect: Optional[bool] = None):
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1, [])
        chroms = np.genfromtxt(f"{self.folderName}/Chrom.csv", delimiter=",", dtype="int32")
        chroms = np.unique(chroms, axis=0)
        chrom_count = chroms.shape[0]

        px = self.points[:, 0]
        py = self.points[:, 1]

        for c in range(chrom_count):
            path_save = f"./ValidityFigure/{self.datetime_now}/manuf_path"
            Path(path_save).mkdir(parents=True, exist_ok=True)
            chrom = chroms[c, :]
            ccv3.set_robotsPath(chrom)
            for rb in range(self.config.robots_count):
                path_index = ccv3.robots[rb].robot_path - 1
                plt.plot(
                    px[path_index],
                    py[path_index],
                    marker="o",
                    markerfacecolor=f"{self.robots_color[rb]}",
                    markersize="10",
                    linestyle="",
                )

                if is_connect:
                    try:
                        path_index_shift = path_index[1:]
                        for i, next_i in zip(path_index, path_index_shift):
                            plt.arrow(
                                px[i],
                                py[i],
                                px[next_i] - px[i],
                                py[next_i] - py[i],
                                head_width=15,
                                length_includes_head=True,
                                color="red",
                            )
                    except Exception:
                        pass

            plt.savefig(f"{path_save}/ChromID_{c}")
            plt.cla()
            plt.clf()
            plt.close()
            # plt.show()

    def draw_pareto(self):
        path_save = f"./ValidityFigure/{self.datetime_now}"
        Path(path_save).mkdir(parents=True, exist_ok=True)
        objV = np.genfromtxt(f"{self.folderName}/ObjV.csv", delimiter=",")
        objV = IndexComparision._sort_obj_value(objV)
        plt.plot(objV[:, 0], objV[:, 1], "ro")
        plt.title("Pareto Front")
        plt.xlabel(r"$T$")
        plt.ylabel(r"$\sigma$")
        plt.savefig(f"{path_save}/pareto.png")
        plt.cla()
        plt.clf()
        plt.close()
        # plt.show()

    @staticmethod
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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


def main():
    folderName = "50_points/211111-215630"
    dr = DrawRobots(folderName)
    dr.chrom_to_png()
    dr.png_to_gif()
    dr.draw_manuf_route()


class IndexComparision:
    def _manhattan_distance(sorted_objV) -> List:
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

    def _spacing_metric(objV):
        objV = IndexComparision._sort_obj_value(objV)
        dist = IndexComparision._manhattan_distance(objV)
        dist_count = len(dist)
        dist_np = np.array(dist)
        dist_mean = np.mean(dist)
        sp = np.sqrt(np.sum(np.square(dist_np - dist_mean)) / (dist_count - 1))
        return sp

    def _utopia_point_value(objV) -> List:
        try:
            if objV.shape[1] != 2:
                raise ArgumentError(
                    f"number of objectives should be 2 for this method, but now is {objV.shape[1]}"
                )
        except ArgumentError as e:
            print(repr(e))
            raise
        objV = IndexComparision._sort_obj_value(objV)
        return [objV[0, 0], objV[-1, 1]]

    def _nadir_point_value(objV):
        try:
            if objV.shape[1] != 2:
                raise ArgumentError(
                    f"number of objectives should be 2 for this method, but now is {objV.shape[1]}"
                )
        except ArgumentError as e:
            print(repr(e))
            raise
        objV = IndexComparision._sort_obj_value(objV)
        return [objV[-1, 0], objV[0, 1]]

    def _overall_pareto_spread(objV, range_obj):
        objV = IndexComparision._sort_obj_value(objV)
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

    def _distribution_metric(objV, range_obj):
        def _distance(_sorted_objV):
            d = []
            for obj_num in range(_sorted_objV.shape[1]):
                d.append(np.abs(np.diff(_sorted_objV[:, obj_num])))
            return d

        def _sigma(s, d):
            return np.sum(np.square(d - np.mean(d))) / (s - 2)

        def _mu(s, d):
            return np.sum(d) / (s - 1)

        objV = IndexComparision._sort_obj_value(objV)
        d = _distance(objV)
        f_pb = range_obj[0]
        f_pg = range_obj[1]
        dm_h = []
        s = objV.shape[0]
        for obj_num in range(objV.shape[1]):
            # test = _sigma(s, d[obj_num])
            # upper = np.std(d[obj_num]) / np.mean(d[obj_num])
            upper = _sigma(s, d[obj_num]) / _mu(s, d[obj_num])
            bottom = np.abs(np.max(objV[:, obj_num]) - np.min(objV[:, obj_num])) / np.abs(
                f_pb[obj_num] - f_pg[obj_num]
            )
            dm_h.append(upper / bottom)
        dm = sum(dm_h) / s
        return dm

    def _sort_obj_value(objV):
        objV = np.unique(objV, axis=0)
        objV = objV[objV[:, 0].argsort()]
        return objV

    def _find_utopia_nadir(*main_folder):
        dir_list = []
        for folder in main_folder:
            dir_list.append(os.listdir(folder))
        utopia_list = []
        nadir_list = []
        for i, dir_folder in enumerate(dir_list):
            for dir in dir_folder:
                objV = np.genfromtxt(f"{main_folder[i]}/{dir}/ObjV.csv", delimiter=",")
                utopia_list.append(IndexComparision._utopia_point_value(objV))
                nadir_list.append(IndexComparision._nadir_point_value(objV))
        utopia_np = np.array(utopia_list)
        nadir_np = np.array(nadir_list)
        utopia_value = np.min(utopia_np, axis=0)
        nadir_value = np.max(nadir_np, axis=0)
        return utopia_value, nadir_value

    def _remove_private_folder_from_list(folder_list):
        for folder in folder_list:
            if folder[:2] == "__":
                folder_list.remove(folder)
        return folder_list

    def main_distribution_metric(method: DMType, *folders_paths, un):
        # noRep_path = "./[Result]/noRep_10000Gen_noSlice"
        # repRandom_path = "./[Result]/rep_10000Gen_noSlice"
        # repReverse_path = "./[Result]/mode_reverse"

        folders_lists = []
        for path in folders_paths:
            path_list = os.listdir(path)
            path_list = IndexComparision._remove_private_folder_from_list(path_list)
            folders_lists.append(path_list)

        if method == DMType.SP:
            metric_method = IndexComparision._spacing_metric
        elif method == DMType.OS:
            metric_method = IndexComparision._overall_pareto_spread
            # utopia_nadir = IndexComparision._find_utopia_nadir(*folders_paths)
        elif method == DMType.DM:
            metric_method = IndexComparision._distribution_metric
            # utopia_nadir = IndexComparision._find_utopia_nadir(*folders_paths)

        folders = folders_lists
        folders_path = [ph for ph in folders_paths]
        metric_all_folder = []
        metric_mean = []
        metric_std = []
        for i in range(len(folders)):
            metric = []
            for dir in folders[i]:
                objV = np.genfromtxt(f"{folders_path[i]}/{dir}/ObjV.csv", delimiter=",")
                if method == DMType.SP:
                    metric.append(metric_method(objV))
                else:
                    metric.append(metric_method(objV, un))
            metric_all_folder.append(metric)
            metric_mean.append(np.mean(metric))
            metric_std.append(np.std(metric))
        return metric_all_folder, metric_mean, metric_std

    def main_cIndex(self_path: str, other_path: str, is_plot: bool = False) -> np.ndarray:
        def plot_vs(obj_rep, obj_noRep, value_rep, value_noRep, rep_filename, noRep_filename):
            (p1,) = plt.plot(obj_rep[:, 0], obj_rep[:, 1], "ro")
            (p2,) = plt.plot(obj_noRep[:, 0], obj_noRep[:, 1], "bo")
            plt.xlabel("total angle changes of every robots")
            plt.ylabel("std of every robots' angle changes")
            plt.legend([p1, p2], [f"rep, vs no_rep={value_rep}", f"no_rep, vs rep={value_noRep}"])
            plt.title(f"{rep_filename}(rep) vs {noRep_filename}(noRep)")

        folder_self_path = self_path
        folder_other_path = other_path

        folder_self_list = os.listdir(folder_self_path)
        folder_self_list = IndexComparision._remove_private_folder_from_list(folder_self_list)
        folder_other_list = os.listdir(folder_other_path)
        folder_other_list = IndexComparision._remove_private_folder_from_list(folder_other_list)
        datetime_now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        path_output = f"./[Result]/c_measurement/{datetime_now}"
        Path(f"{path_output}/figure").mkdir(parents=True, exist_ok=True)

        win_self_count = 0
        win_other_count = 0
        equal_count = 0
        self_vs_other = np.zeros((0, 1))
        other_vs_self = np.zeros((0, 1))
        for myself, other in zip(folder_self_list, folder_other_list):
            obj_self_path = f"{folder_self_path}/{myself}/ObjV.csv"
            obj_self = np.genfromtxt(obj_self_path, delimiter=",")
            obj_self = np.unique(obj_self, axis=0)
            # if obj_noRep.shape[0] >= 20:
            obj_other_path = f"{folder_other_path}/{other}/ObjV.csv"
            obj_other = np.genfromtxt(obj_other_path, delimiter=",")
            obj_other = np.unique(obj_other, axis=0)
            # if obj_rep.shape[0] >= 20:
            c_value_self = c_measurement(obj_self, obj_other)
            c_value_other = c_measurement(obj_other, obj_self)
            self_vs_other = np.vstack((self_vs_other, c_value_self))
            other_vs_self = np.vstack((other_vs_self, c_value_other))

            if c_value_self > c_value_other:
                win_self_count += 1
            if c_value_self < c_value_other:
                win_other_count += 1
            if c_value_self == c_value_other:
                equal_count += 1
            if is_plot:
                plot_vs(obj_other, obj_self, c_value_other, c_value_self, other, myself)
                plt.savefig(f"{path_output}/figure/{other}_vs_{myself}.png")
                plt.close()
                plt.cla()
                plt.clf()
        compare_index = np.hstack((self_vs_other, other_vs_self))
        win_count = [win_self_count, win_other_count, equal_count]
        # np.savetxt(f"{path_output}/c_result.csv", fileoutput, delimiter=",")
        return compare_index, win_count

    def main_save_metric_comparision():
        method_list = [DMType.SP, DMType.OS, DMType.DM]
        sheet_name = ["SP", "OS", "DM"]
        writer = pd.ExcelWriter(
            f"./[Result]/metrics_compare/{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}.xlsx",
            engine="xlsxwriter",
        )
        for i, md in enumerate(method_list):
            res = IndexComparision.main_distribution_metric(md)
            print(res[1], res[2])
            df = pd.DataFrame(res[0]).T
            df.columns = ["noRep", "rep_random", "rep_reverse"]
            df.to_excel(writer, sheet_name=sheet_name[i])
            # print(res)
        writer.save()


class ResultAnalyzeApp:
    def __init__(self) -> None:

        self.root = tk.Tk()
        self.root.title("HighSchool AI Project")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.folders_name = ["111aa", "222", "333", "444"]
        self.root.geometry("1280x720")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # test = self.get_files_path()

        # --------show folder list--------
        df_col = ["path"]
        self.tree = ttk.Treeview(self.root, show="headings", columns=df_col)
        # counter = len(self.folders_name)
        # rowLabels = self.folders_name.index
        for j in range(len(df_col)):
            self.tree.column(df_col[j], width=1150)
            self.tree.heading(df_col[j], text=df_col[j])
        for i, d in enumerate(self.folders_name):
            self.tree.insert("", len(self.folders_name) - 1 + i, values=[d])
        self.tree.pack(side=tk.BOTTOM, padx=20)

        # # --------path button--------

        frame_sel_path = tk.Frame(self.root)
        self.button_sel_path = tk.Button(
            frame_sel_path, text="select paths", width=15, command=self.select_files_path
        )
        self.button_sel_path.pack(side=tk.RIGHT, pady=50)
        self.button_del_path = tk.Button(frame_sel_path, text="delete paths", width=15, command=self.delete)
        self.button_del_path.pack(side=tk.RIGHT, pady=50)
        frame_sel_path.pack()

        frame_operations = tk.Frame(self.root)
        self.button_SP = tk.Button(frame_operations, text="SP", width=15, command=self.select_files_path)
        self.button_OS = tk.Button(frame_operations, text="OS", width=15, command=self.select_files_path)
        self.button_DM = tk.Button(frame_operations, text="DM", width=15, command=self.select_files_path)

        self.video_loop()
        # self.root.mainloop()

    def video_loop(self) -> None:
        # self.paths.config(text=f"{self.folders_name}")

        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds

    def select_files_path(self):
        dirs = tkf.askopendirnames()
        for i, d in enumerate(dirs):
            self.tree.insert("", len(self.folders_name) - 1 + i, values=[d])
        dirs = list(dirs)
        self.folders_name.extend(dirs)

    def delete(self) -> None:
        selected_item = self.tree.selection()
        for i in range(len(selected_item)):
            selected_values = self.tree.item(selected_item[i])["values"]
            selected_values = selected_values[0]
            self.folders_name.remove(str(selected_values))
            self.tree.delete(selected_item[i])

    def sp(self):
        IndexComparision.main_distribution_metric()

    def destructor(self) -> None:
        print("[INFO] closing...")
        self.root.destroy()


class Single_Robot:
    def save_pareto() -> None:
        objv_path_1 = "./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909"
        objv_path_2 = "./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220316-114439"
        if os.path.exists(objv_path_1):
            objv_1 = np.genfromtxt(f"{objv_path_1}/ObjV.csv", delimiter=",")
            objv_2 = np.genfromtxt(f"{objv_path_2}/ObjV.csv", delimiter=",")
            try:
                (p1,) = plt.plot(objv_1[:, 0], objv_1[:, 1], ".r")
                (p2,) = plt.plot(objv_2[:, 0], objv_2[:, 1], ".b")
                plt.title("Pareto Front")
                plt.xlabel("Total Travel Distance")
                plt.ylabel("Total Angle Changes of Robot")
                plt.legend([p1, p2], ["multi_solutions", "single_solution"])
                plt.savefig(f"{objv_path_2}/pareto_2type.png")
                plt.close()
                plt.cla()
                plt.clf()
            except Exception:
                pass

        else:
            with open("./Result/pareto.txt", "w") as file:
                file.write("No solution")

    def first_sol_export():
        config = Config("./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/config.yml")
        chromo_path = "./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/Chrom.csv"
        points_path = "./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/output_point.csv"
        chromo = np.genfromtxt(chromo_path, delimiter=",", dtype="int32")
        points = np.genfromtxt(points_path, delimiter=",")
        first_sol = chromo[0, :]
        idx_first_sol = first_sol - 1
        points_first_sol = points[idx_first_sol]
        rc = RobotCalc_pygeos(config)
        rb = Robot(0, Position.LEFT)
        q_1_best = config.org_pos
        q_best = np.zeros((0, 6))
        for pt in points_first_sol:
            vv_pt = Coord(pt[0], pt[1], pt[2])
            q_2_best = rc.coord2bestAngle(vv_pt, q_1_best, rb)
            q_best = np.vstack((q_best, q_2_best))
            q_1_best = q_2_best
        np.savetxt(
            "./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/best_joints.csv",
            q_best,
            delimiter=",",
        )
        np.savetxt(
            "./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/points_first_sol.csv",
            points_first_sol,
            delimiter=",",
        )

    def ik_list():
        config = Config("./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/config.yml")
        rc = RobotCalc_pygeos(config)
        points_path = "./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/output_point.csv"
        points = np.genfromtxt(points_path, delimiter=",")
        rb = Robot(0, Position.LEFT)
        first_three_idx = [0, 2, 4, 6]
        ik_all = np.empty((0, 7))
        for i, pt in enumerate(points):
            vv = Coord(pt[0], pt[1], pt[2])
            id_pt_list = np.hstack((int(i + 1), pt))
            vv_ik = rc.userIK(vv)
            vv_ik_first_three = vv_ik[first_three_idx, 0:3]
            id_pt_array = np.repeat(id_pt_list[None, :], 4, axis=0)
            id_pt_ik = np.hstack((id_pt_array, vv_ik_first_three))
            ik_all = np.vstack((ik_all, id_pt_ik))
        np.savetxt(
            "./[Result]/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/points_all_ik.csv",
            ik_all,
            delimiter=",",
        )


def index_test():
    f_1 = "./[Result]/Robot_2/points_count100/noStep/Gen10000/replace/Hamming10/poly_traj"
    f_2 = "./[Result]/Robot_2/points_count100/noStep/Gen10000/replace/Hamming20/poly_traj"
    f_3 = "./[Result]/Robot_2/points_count100/noStep/Gen10000/replace/Hamming30/poly_traj"
    f_4 = "./[Result]/Robot_2/points_count100/noStep/Gen10000/replace/Hamming40/poly_traj"
    f_5 = "./[Result]/Robot_2/points_count100/noStep/Gen10000/no_replace/poly_traj"
    un = IndexComparision._find_utopia_nadir(f_1, f_2, f_3, f_4, f_5)

    folder_self_path = "./[Result]/Robot_2/points_count100/noStep/Gen10000/no_replace/poly_traj"
    folder_other_path = "./[Result]/Robot_2/points_count100/noStep/Gen10000/replace/Hamming30/poly_traj"
    res_c = IndexComparision.main_cIndex(folder_self_path, folder_other_path)
    res_dm = IndexComparision.main_distribution_metric(DMType.DM, folder_self_path, folder_other_path, un=un)
    print(res_c[1])
    print(res_dm[1])
    print(res_dm[2])


def draw_figure():
    # dr = DrawRobots("./[Result]/Robot_2/noStep/Gen10000/random/Hamming10/poly_traj/220422-222226")
    # dr = DrawRobots(
    #     "./[Result]/Robot_2/points_count25/noStep/Gen10000/random/Hamming5/poly_traj/220513-211237"
    # )
    # dr = DrawRobots(
    #     "./[Result]/Robot_2/points_count50/noStep/Gen10000/random/Hamming10/poly_traj/220513-110428"
    # )
    # dr = DrawRobots("./[Result]/Robot_2/noStep/Gen10000/random/Hamming20/poly_traj/220421-180516")
    # dr = DrawRobots("./[Result]/Robot_2/noStep/Gen10000/no_replace/poly_traj/220419-203253")
    dr = DrawRobots(
        "./[Result]/Robot_4/points_count100/noStep/Gen5000/replace/Hamming20/poly_traj/220606-150353"
    )
    # dr.draw_manuf_route(is_connect=True)
    dr.draw_pareto()


def draw_robot():
    # dr = DrawRobots("./[Result]/Robot_4/noStep/Gen10000/random/Hamming30/220312-151028")
    dr = DrawRobots(
        "./[Result]/Robot_2/points_count100/noStep/Gen10000/replace/Hamming10/poly_traj/220522-085023"
    )

    q_best_1 = np.radians(np.array([[0, 0, 0, 0, 0, 0]]))
    q_best_2 = np.radians(np.array([[0, 0, 0, 0, 0, 0]]))
    q_best_3 = np.radians(np.array([[0, 0, 0, 0, 0, 0]]))
    q_best_4 = np.radians(np.array([[0, 0, 0, 0, 0, 0]]))

    # dr.draw([q_best_1, q_best_2, q_best_3, q_best_4], 0, axis=[[-100, 1000], [-550, 550], [-350, 350]])
    dr.draw([q_best_1, q_best_2], 0, axis=[[-100, 1000], [-550, 550], [-350, 350]])


if __name__ == "__main__":
    # dr.config.baseX_offset -= 200
    # q_best_1 = np.radians(np.array([[0, 60, -20, 0, -90, 0]]))
    # q_best_2 = np.radians(np.array([[0, 60, -20, 0, -90, 0]]))
    # q_best_1 = np.radians(np.array([[0, 0, 0, 0, -90, 0]]))
    # q_best_2 = np.radians(np.array([[0, 0, 0, 0, -90, 0]]))
    # q_best_1 = np.radians(np.array([[-20, 60, -20, 0, -90, 0]]))
    # q_best_2 = np.radians(np.array([[-20, 60, -20, 0, -90, 0]]))

    # dr._draw_cph_vs_seg([q_best_1, q_best_2], 0, "1", axis=[[0, 700], [-350, 350], [-350, 350]])
    # try:
    #     dr.chrom_to_png(axis=[[0, 900], [-450, 450], [-450, 450]])
    # except Exception:
    #     pass

    # config = Config("./[Result]/Robot_2/noStep/Gen10000/random/Hamming10/220320-121932/CONFIG.yml")
    # points = np.genfromtxt(
    #     "./[Result]/Robot_2/noStep/Gen10000/random/Hamming10/220320-121932/output_point.csv", delimiter=","
    # )
    # ccv3 = ChromoCalcV3(config, points, 0, 1, [])
    # chrom = np.genfromtxt(
    #     f"./[Result]/Robot_2/noStep/Gen10000/random/Hamming10/220320-121932/Chrom.csv",
    #     delimiter=",",
    #     dtype="int32",
    # )
    # test = ccv3.interpolation_step(chrom[3, :])
    # print()
    # draw_figure()
    index_test()
    # draw_robot()
