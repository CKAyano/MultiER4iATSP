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
    obj_a_count = obj_a_all.shape[0]
    obj_b_count = obj_b_all.shape[0]
    dominate_count = 0
    for obj_a in obj_a_all:
        for obj_b in obj_b_count:
            is_dominate = ChromoCalcV3.dominates(obj_a, obj_b)
            if is_dominate:
                dominate_count += 1


def main():
    folderName = "50_points/211111-215630"
    dr = DrawRobots(folderName)
    dr.chrom_to_png()
    dr.png_to_gif()
    dr.draw_manuf_route()


if __name__ == "__main__":
    main()
