from argparse import ArgumentError
from enum import Enum, auto
import numpy as np
from chromoCalcV3 import ChromoCalcV3
from robotCalc_pygeos import RobotCalc_pygeos
import Geometry3Dmaster.Geometry3D as gm
from pathlib import Path
from typing import List, Optional
from robot_configuration import Coord, Position, Robot, Config
from status_logging import Collision_status
import os
import imageio
import natsort
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from mpl_toolkits.mplot3d import Axes3D


class DrawRobots:
    def __init__(self, folderName: str, folder_name=None) -> None:
        if os.path.exists(f"{folderName}/config.yml"):
            self.config = Config(f"{folderName}/config.yml")
        else:
            self.config = Config(f"{folderName}/CONFIG.yml")
        self.points = np.genfromtxt(f"{folderName}/output_point.csv", delimiter=",")
        self.folderName = folderName
        self.rc = RobotCalc_pygeos(self.config)
        for i in range(len(self.rc.robot_kine.links_width)):
            self.rc.robot_kine.links_width[i] /= 2
        self.robots_color = ["k", "royalblue", "r", "g"]
        # self.ccv3 = ChromoCalcV3(self.config, self.points, 0, 1)
        if folder_name is None:
            self.folder_name = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        else:
            self.folder_name = folder_name

    @staticmethod
    def cph(points):
        point1 = gm.geometry.Point(points[0][0], points[0][1], points[0][2])
        point2 = gm.geometry.Point(points[1][0], points[1][1], points[1][2])
        point3 = gm.geometry.Point(points[2][0], points[2][1], points[2][2])
        point4 = gm.geometry.Point(points[3][0], points[3][1], points[3][2])
        point5 = gm.geometry.Point(points[4][0], points[4][1], points[4][2])
        point6 = gm.geometry.Point(points[5][0], points[5][1], points[5][2])
        point7 = gm.geometry.Point(points[6][0], points[6][1], points[6][2])
        point8 = gm.geometry.Point(points[7][0], points[7][1], points[7][2])
        cpg1 = gm.geometry.ConvexPolygon((point1, point2, point3, point4))
        cpg2 = gm.geometry.ConvexPolygon((point1, point2, point6, point5))
        cpg3 = gm.geometry.ConvexPolygon((point4, point3, point7, point8))
        cpg4 = gm.geometry.ConvexPolygon((point2, point3, point7, point6))
        cpg5 = gm.geometry.ConvexPolygon((point1, point4, point8, point5))
        cpg6 = gm.geometry.ConvexPolygon((point5, point6, point7, point8))
        return gm.geometry.ConvexPolyhedron((cpg1, cpg2, cpg3, cpg4, cpg5, cpg6))

    def _get_robot_cph(self, q, robot):
        link_count = len(self.rc.robot_kine.collision_links)
        va_all = self.rc.userFK(q)
        va_all = self.rc.robot2world_v_all(va_all, robot.position)
        cph_all = tuple()
        for i in range(link_count):
            v_f = va_all.__dict__.get(self.rc.robot_kine.collision_links[i][0])
            v_e = va_all.__dict__.get(self.rc.robot_kine.collision_links[i][1])
            points = self.rc._get_link_points_by_joints_position(
                q, v_f, v_e, self.rc.robot_kine.links_width[i]
            )
            cph = self.cph(points)
            cph_all += (cph,)
        return cph_all

    def _cph_vs_seg_robots(self, q_best, int_point):
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1, [])
        cphs = []
        link_count = len(self.rc.robot_kine.collision_links)
        va_all = self.rc.userFK(q_best[0][int_point, :])
        va_all = self.rc.robot2world_v_all(va_all, ccv3.robots[0].position)
        vb_all = self.rc.userFK(q_best[1][int_point, :])
        vb_all = self.rc.robot2world_v_all(vb_all, ccv3.robots[1].position)
        cphs_all = tuple()
        segs_all = tuple()
        for i in range(link_count):
            va_f = va_all.__dict__.get(self.rc.robot_kine.collision_links[i][0])
            va_e = va_all.__dict__.get(self.rc.robot_kine.collision_links[i][1])
            vb_f = vb_all.__dict__.get(self.rc.robot_kine.collision_links[i][0])
            vb_e = vb_all.__dict__.get(self.rc.robot_kine.collision_links[i][1])
            points = self.rc._get_link_points_by_joints_position(
                q_best[0][int_point, :], va_f, va_e, self.rc.robot_kine.links_width[i]
            )
            vb_points_f = gm.geometry.Point(vb_f.xx, vb_f.yy, vb_f.zz)
            vb_points_e = gm.geometry.Point(vb_e.xx, vb_e.yy, vb_e.zz)
            gm_seg = gm.geometry.Segment(vb_points_f, vb_points_e)
            cph = self.cph(points)
            cphs_all += (cph,)
            segs_all += (gm_seg,)

        return cphs_all, segs_all

    def cph_robots(self, q_best: List[np.ndarray], intPoint) -> bool:
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1, [])
        cphs_rb_all = []
        for rb in range(self.config.robots_count):
            # points = self.rc.get_robot_polygons(q_best[rb][intPoint, :], ccv3.robots[rb])
            cph_rb = self._get_robot_cph(q_best[rb][intPoint, :], ccv3.robots[rb])
            cphs_rb_all.append(cph_rb)
        return cphs_rb_all

    def _draw_cph_vs_seg(
        self, q_best: List[np.ndarray], intPoint, save_path: Optional[str] = None, axis=None, is_show=True
    ):
        cphs, segs = self._cph_vs_seg_robots(q_best, intPoint)

        inters = []
        for cph in cphs:
            for seg in segs:
                int = gm.calc.intersection(cph, seg)
                inters.append(int)

        inters_str = [str(int) for int in inters]

        r = gm.render.Renderer()

        # for rb in range(self.config.robots_count):
        for cph in cphs:
            r.add((cph, self.robots_color[0], 2), normal_length=0)
        # r.add((cphs[1], self.robots_color[0], 2), normal_length=0)
        for seg in segs:
            r.add((seg, self.robots_color[1], 2), normal_length=0)
        # r.add((cphs[3], self.robots_color[1], 2), normal_length=0)

        color_inter = "darkorange"
        for i, int_str in enumerate(inters_str):
            if int_str == "None":
                pass
            else:
                r.add((inters[i], color_inter, 8), normal_length=0)

        if is_show:
            r.show(axis=axis, dpi=200)
        if save_path:
            r.savefigure(save_path, axis)

    def draw(self, q_best: List[np.ndarray], intPoint, save_path=None, axis=None, is_show=True):
        cphs_rb_all = self.cph_robots(q_best, intPoint)
        r = gm.render.Renderer()

        for i, cph_rb in enumerate(cphs_rb_all):
            for cph in cph_rb:
                r.add((cph, self.robots_color[i], 3), normal_length=0)
            # r.add((cphs[rb][1], self.robots_color[rb], 3), normal_length=0)
        if is_show:
            r.show(dpi=200, axis=axis)
        if save_path:
            r.savefigure(save_path, axis)

    def chrom_to_png(self, is_log: Optional[bool] = None, axis=None):
        ccv3 = ChromoCalcV3(self.config, self.points, 0, 1, [])
        chrom = np.genfromtxt(f"{self.folderName}/Chrom.csv", delimiter=",", dtype="int32")
        for chromoInd in range(chrom.shape[0]):
            if is_log:
                Path(f"./ValidityFigure/{self.folder_name}/ChromID_{chromoInd}/log").mkdir(
                    parents=True, exist_ok=True
                )
                logging = Collision_status(0, f"./ValidityFigure/{self.folder_name}/ChromID_{chromoInd}/log")
            totalInt_q, _ = ccv3.interpolation(chrom[chromoInd, :])
            points_count = np.shape(totalInt_q[0])[0]
            Path(f"./ValidityFigure/{self.folder_name}/ChromID_{chromoInd}/figure").mkdir(
                parents=True, exist_ok=True
            )
            for intPoint in range(points_count):
                path = (
                    f"./ValidityFigure/{self.folder_name}/" + f"ChromID_{chromoInd}/figure/figure_{intPoint}"
                )
                try:
                    self.draw(totalInt_q, intPoint, path, axis, is_show=False)
                except Exception as e:
                    print(f"can't draw, because {e}")
                    # raise
                    continue
        if is_log:
            _ = ccv3.score_step(chrom[chromoInd, :], logging)

    def png_to_gif(self, fps=20):
        chrom_count = len(os.listdir(f"./ValidityFigure/{self.folder_name}"))
        for i in range(chrom_count):
            png_dir = f"./ValidityFigure/{self.folder_name}/ChromID_{i}/figure"
            images = []
            png_list = os.listdir(png_dir)
            sortedList = natsort.natsorted(png_list)
            for file_name in sortedList:
                if file_name.endswith(".png"):
                    file_path = os.path.join(png_dir, file_name)
                    images.append(imageio.imread(file_path))
            path = f"./ValidityFigure/{self.folder_name}/ChromID_{i}/animate.gif"
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
            path_save = f"./ValidityFigure/{self.folder_name}/manuf_path"
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

    def draw_pareto(self, xlabel: str = None, ylabel: str = None, dpi: float = None):
        path_save = f"./ValidityFigure/{self.folder_name}"
        Path(path_save).mkdir(parents=True, exist_ok=True)
        objV = np.genfromtxt(f"{self.folderName}/ObjV.csv", delimiter=",")
        objV = IndexComparision._sort_obj_value(objV)
        plt.figure(figsize=(4, 3))
        if dpi:
            plt.figure(figsize=(4, 3), dpi=dpi)
        plt.plot(objV[:, 0], objV[:, 1], "ro")
        plt.title("Pareto Front")
        if xlabel:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(r"$T$")
        if ylabel:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(r"$\sigma$")
        # if fontsize:
        #     plt.rcParams.update({"font.size": fontsize})
        plt.tight_layout()
        plt.savefig(f"{path_save}/pareto.png", bbox_inches="tight")
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
        # noRep_path = "./all_results/noRep_10000Gen_noSlice"
        # repRandom_path = "./all_results/rep_10000Gen_noSlice"
        # repReverse_path = "./all_results/mode_reverse"

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
        path_output = f"./all_results/c_measurement/{datetime_now}"
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
            f"./all_results/metrics_compare/{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}.xlsx",
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


class Single_Robot:
    def save_pareto() -> None:
        objv_path_1 = "./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909"
        objv_path_2 = "./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220316-114439"
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
        config = Config("./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/config.yml")
        chromo_path = "./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/Chrom.csv"
        points_path = "./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/output_point.csv"
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
            "./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/best_joints.csv",
            q_best,
            delimiter=",",
        )
        np.savetxt(
            "./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/points_first_sol.csv",
            points_first_sol,
            delimiter=",",
        )

    def ik_list():
        config = Config("./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/config.yml")
        rc = RobotCalc_pygeos(config)
        points_path = "./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/output_point.csv"
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
            "./all_results/Robot_1/noStep/Gen10000/random/Hamming15/220308-061909/points_all_ik.csv",
            ik_all,
            delimiter=",",
        )


def index_test():
    f_1 = "./all_results/fanuc/Robot_2/points_count100/noStep/Gen10000/replace/Hamming5/poly_traj"
    f_2 = "./all_results/fanuc/Robot_2/points_count100/noStep/Gen10000/replace/Hamming10/poly_traj"
    f_3 = "./all_results/fanuc/Robot_2/points_count100/noStep/Gen10000/replace/Hamming20/poly_traj"
    f_4 = "./all_results/fanuc/Robot_2/points_count100/noStep/Gen10000/replace/Hamming30/poly_traj"
    f_5 = "./all_results/fanuc/Robot_2/points_count100/noStep/Gen10000/replace/Hamming40/poly_traj"
    f_6 = "./all_results/fanuc/Robot_2/points_count100/noStep/Gen10000/no_replace/poly_traj"
    un = IndexComparision._find_utopia_nadir(f_1, f_2, f_3, f_4, f_5, f_6)

    folder_self_path = "./all_results/fanuc/Robot_2/points_count100/noStep/Gen10000/no_replace/poly_traj"
    folder_other_path = (
        "./all_results/fanuc/Robot_2/points_count100/noStep/Gen10000/replace/Hamming30/poly_traj"
    )
    res_c = IndexComparision.main_cIndex(folder_self_path, folder_other_path)
    res_dm = IndexComparision.main_distribution_metric(DMType.DM, folder_self_path, folder_other_path, un=un)
    print(res_c[1])
    print(res_dm[1])
    print(res_dm[2])


def draw_figure():
    dr = DrawRobots(
        "./all_results/puma/Robot_2/points_count50/noStep/"
        + "Gen10000/replace/Hamming5/poly_traj/220706-233141",
        "220713-144954",
    )
    # dr = DrawRobots(
    #     "./all_results/puma/Robot_2/points_count50/noStep/"
    #     + "Gen10000/replace/Hamming5/poly_traj/220623-003729"
    # )
    # dr.draw_manuf_route(is_connect=True)
    # dr.draw_manuf_dist()
    dr.draw_pareto("Manufacturing Time", "Load Balance of Each Robot", 300)


def draw_robot():
    # dr = DrawRobots("./all_results/Robot_4/noStep/Gen10000/random/Hamming30/220312-151028")
    # dr = DrawRobots(
    #     "./all_results/fanuc/Robot_2/points_count100/"
    #     + "noStep/Gen10000/replace/Hamming10/poly_traj/220522-085023"
    # )
    dr = DrawRobots(
        "./all_results/puma/Robot_2/points_count100/noStep"
        + "/Gen10000/replace/Hamming10/poly_traj/220621-050842"
    )

    q_best_1 = np.radians(np.array([[1, 1, 1, 1, 1, 1]]))
    q_best_2 = np.radians(np.array([[1, 1, 1, 1, 1, 1]]))
    # q_best_1 = np.radians(np.array([[30, 30, -40, 0, 0, 0]]))
    # q_best_2 = np.radians(np.array([[40, 40, -50, 0, 0, 0]]))

    # dr.draw([q_best_1, q_best_2, q_best_3, q_best_4], 0, axis=[[-100, 1000], [-550, 550], [-350, 350]])
    dr.draw([q_best_1, q_best_2], 0, axis=[[-100, 1200], [-650, 650], [-650, 650]])


def draw_robot_polygons_vs_segs():
    dr = DrawRobots(
        "./all_results/fanuc/Robot_2/points_count100/"
        + "noStep/Gen10000/random/Hamming10/poly_traj/220422-222226"
    )
    dr.config.baseX_offset -= 200
    # q_best_1 = np.radians(np.array([[0, 60, -20, 0, -90, 0]]))
    # q_best_2 = np.radians(np.array([[0, 60, -20, 0, -90, 0]]))
    # q_best_1 = np.radians(np.array([[0, 0, 0, 0, -90, 0]]))
    # q_best_2 = np.radians(np.array([[0, 0, 0, 0, -90, 0]]))
    q_best_1 = np.radians(np.array([[-20, 60, -20, 0, -90, 0]]))
    q_best_2 = np.radians(np.array([[-20, 60, -20, 0, -90, 0]]))

    dr._draw_cph_vs_seg([q_best_1, q_best_2], 0, "1", axis=[[0, 700], [-350, 350], [-350, 350]])


def draw_route_gif():

    dr = DrawRobots(
        "./all_results/fanuc/Robot_2/points_count75/noStep"
        + "/Gen10000/replace/Hamming8/poly_traj/220617-031624",
        "fanuc_2-75",
    )

    dr.chrom_to_png(axis=[[0, 700], [-350, 350], [-350, 350]])
    dr.png_to_gif()


def get_robots_points_index():
    folder_path = (
        "./all_results/puma/Robot_2/points_count50/noStep/Gen10000/replace/Hamming5/poly_traj/220706-233141"
    )
    config = Config(f"{folder_path}/CONFIG.yml")
    points = np.genfromtxt(f"{folder_path}/output_point.csv", delimiter=",",)
    cc = ChromoCalcV3(config, points, 0, 1, [])
    chrom_all = np.genfromtxt(f"{folder_path}/Chrom.csv", delimiter=",", dtype=int)

    chrom = chrom_all[2, :]
    scores = cc.score_step(chrom)
    cc.set_robotsPath(chrom)
    # for chrom in chrom_all:
    #     scores = cc.score_step(chrom)
    #     cc.set_robotsPath(chrom)
    #     print(f"{scores[0]}, " f"{len(cc.robots[0].robot_path)}", f"{len(cc.robots[1].robot_path)}")
    points_rb = cc.get_points_from_robots()
    for pts in points_rb:
        pts_rounded = np.round(pts, 2)
        for i, pt in enumerate(pts_rounded):
            print(i + 1, f"({pt[0]}, {pt[1]}, {pt[2]})")
        print()
    print(f"{scores[0]}\n")
    print(f"{cc.robots[0].robot_path}")
    print(f"{len(cc.robots[0].robot_path)}\n")
    print(f"{cc.robots[1].robot_path}")
    print(f"{len(cc.robots[1].robot_path)}\n")


if __name__ == "__main__":
    draw_figure()
    # get_robots_points_index()
    # index_test()
    # draw_robot()
    # draw_robot_polygons_vs_segs()
    # draw_route_gif()
