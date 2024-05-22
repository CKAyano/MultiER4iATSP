import datetime
import os
from pathlib import Path

import numpy as np

from chromoCalcV3 import ChromoCalcV3
from robot_configuration import Config


class Dump_IK:
    def __init__(self, folderName: str, folder_name=None) -> None:
        if os.path.exists(f"{folderName}/config.yml"):
            config = Config(f"{folderName}/config.yml")
        else:
            config = Config(f"{folderName}/CONFIG.yml")
        points = np.genfromtxt(f"{folderName}/output_point.csv", delimiter=",")
        self.folderName = folderName
        self.ccv3 = ChromoCalcV3(config, points, 0, 1, [])
        if folder_name is None:
            self.folder_name = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        else:
            self.folder_name = folder_name

    def _split_rbs(self, all_q):
        all_q_np = np.array(all_q)
        q_splited = []
        for rb in range(self.ccv3.config.robots_count):
            q_splited.append(all_q_np[:, rb, :])
        return q_splited

    def get_ik(self, chrom):
        self.ccv3.set_robotsPath(chrom)
        is_firstLoop = True

        len_pointIndex = len(self.ccv3.robots[0].point_index)
        q_2_best_rbs = None

        for i in range(len_pointIndex):
            if is_firstLoop:
                q_1_best_rbs = [self.ccv3.config.org_pos] * self.ccv3.config.robots_count
                all_q = [q_1_best_rbs]
                is_firstLoop = False
            else:
                all_q.append(q_2_best_rbs)
                q_1_best_rbs = q_2_best_rbs

            p_id_rbs = [self.ccv3.robots[rb].point_index[i] for rb in range(self.ccv3.config.robots_count)]

            q_2_best_rbs = self.ccv3._get_best_q_2(p_id_rbs, q_1_best_rbs)
        all_q_splited = self._split_rbs(all_q)
        return all_q_splited

    def get_ik_interpolation(self, chrom):
        int_q_rbs, _ = self.ccv3.interpolation(chrom)
        return int_q_rbs


def main() -> None:
    result_path = (
        "./all_results/fanuc/Robot_2/points_count25/noStep/Gen10000/replace/Hamming5/poly_traj/220609-053247"
    )
    dump = Dump_IK(result_path)
    chroms = np.genfromtxt(f"{result_path}/Chrom.csv", delimiter=",", dtype="int32")

    folder_path_ik = f"{result_path}/Dump/ik"
    folder_path_ik_with_int = f"{result_path}/Dump/ik_with_int"
    for i, chrom in enumerate(chroms):
        ik = dump.get_ik(chrom)
        ik_with_int = dump.get_ik_interpolation(chrom)
        for rb in range(dump.ccv3.config.robots_count):
            _path_ik = f"{folder_path_ik}/robot_{rb+1}"
            _path_ik_int = f"{folder_path_ik_with_int}/robot_{rb+1}"
            Path(_path_ik).mkdir(parents=True, exist_ok=True)
            Path(_path_ik_int).mkdir(parents=True, exist_ok=True)
            np.savetxt(f"{_path_ik}/chrom_{i+1}.csv", ik[rb], delimiter=",")
            np.savetxt(f"{_path_ik_int}/chrom_{i+1}.csv", ik_with_int[rb], delimiter=",")


if __name__ == "__main__":
    main()
