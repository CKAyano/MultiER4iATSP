import numpy as np
from chromoCalcV3 import ChromoCalcV3
import robotCalc_Geo3D as rcg
from pathlib import Path
from status_logging import Collision_status
from ga_Problem import Problem_config
import os
import imageio
import natsort
import matplotlib.pyplot as plt


def chrom_to_png(folderName):
    config = Problem_config(f"./[Result]/{folderName}/config.yml").config
    points = np.genfromtxt(f"./[Result]/{folderName}/output_point.csv", delimiter=",")
    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]
    points_range = config["points_range"]
    baseX_offset = sum(points_range[0])
    linkWidth = config["linkWidth"]
    originalPosture = np.array(np.radians(config["originalPosture"]))
    axisRange = np.array(config["axisRange"])
    direct_array = np.array(config["direct_array"])
    ccv3 = ChromoCalcV3(
        direct_array,
        baseX_offset,
        linkWidth,
        px,
        py,
        pz,
        originalPosture,
        axisRange,
        0,
        1,
    )

    chrom = np.genfromtxt(f"./[Result]/{folderName}/Chrom.csv", delimiter=",")
    for chromoInd in range(chrom.shape[0]):
        totalInt_qa, totalInt_qb, _, _ = ccv3.interpolation(chrom[chromoInd, :])
        numsOfIntPoints = np.shape(totalInt_qa)[0]
        Path(f"./ValidityFigure/{folderName}/ChromID_{chromoInd}/figure").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"./ValidityFigure/{folderName}/ChromID_{chromoInd}/log").mkdir(
            parents=True, exist_ok=True
        )
        logging = Collision_status(
            0, f"./ValidityFigure/{folderName}/ChromID_{chromoInd}/log"
        )
        for intPoint in range(numsOfIntPoints):
            path = f"./ValidityFigure/{folderName}/\
                   ChromID_{chromoInd}/figure/figure_{intPoint}"
            try:
                isCollision = rcg.cvCollision(
                    totalInt_qa[intPoint, :3], totalInt_qb[intPoint, :3], path
                )
            except Exception:
                continue
            # test = ccv3.rc.cvCollision(totalInt_qa[intPoint, :3],
            #     totalInt_qb[intPoint, :3])
        # test2 = ccv3.scoreOfTwoRobot_step(chrom[chromoInd, :], logging)


def png_to_gif(folderName, fps=24):
    chrom_count = len(os.listdir(f"./ValidityFigure/{folderName}"))
    for i in range(chrom_count):
        png_dir = f"./ValidityFigure/{folderName}/ChromID_{i}/figure"
        images = []
        png_list = os.listdir(png_dir)
        sortedList = natsort.natsorted(png_list)
        for file_name in sortedList:
            if file_name.endswith(".png"):
                file_path = os.path.join(png_dir, file_name)
                images.append(imageio.imread(file_path))
        path = f"./ValidityFigure/{folderName}/ChromID_{i}/animate.gif"
        imageio.mimsave(path, images, format="GIF", fps=fps)


def draw_manuf_route(folderName):
    chroms = np.genfromtxt(f"./[Result]/{folderName}/Chrom.csv", delimiter=",")
    chrom_count = len(os.listdir(f"./ValidityFigure/{folderName}"))
    points = np.genfromtxt(f"./[Result]/{folderName}/output_point.csv", delimiter=",")
    px = points[:, 0]
    py = points[:, 1]
    for c in range(chrom_count):
        chrom = chroms[c, :]
        robot_flag = np.where(chrom == 0)
        robot_flag = robot_flag[0][0]
        robotA_path = chrom.copy()[:robot_flag].astype(int) - 1
        robotA_path_shift = np.hstack((robotA_path[1:], robotA_path[0]))
        robotB_path = chrom.copy()[robot_flag + 1 :].astype(int) - 1
        robotB_path_shift = np.hstack((robotB_path[1:], robotB_path[0]))

        plt.plot(px[robotA_path], py[robotA_path], "ro")
        # for i, next_i in zip(robotA_path, robotA_path_shift):
        #     plt.arrow(
        #         px[i], py[i],
        #         px[next_i] - px[i],
        #         py[next_i] - py[i],
        #         head_width=3, length_includes_head=True, color='red')

        plt.plot(px[robotB_path], py[robotB_path], "bo")
        # for i, next_i in zip(robotB_path, robotB_path_shift):
        #     plt.arrow(
        #         px[i], py[i],
        #         px[next_i] - px[i],
        #         py[next_i] - py[i],
        #         head_width=3, length_includes_head=True, color='blue')

        plt.savefig(f"./ValidityFigure/{folderName}/ChromID_{c}/manuf_route.png")
        plt.show()


if __name__ == "__main__":
    folderName = "211012-204457"
    chrom_to_png(folderName)
    png_to_gif(folderName)
    draw_manuf_route(folderName)
