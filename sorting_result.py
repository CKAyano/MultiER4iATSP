import os
from typing import List, Tuple
import yaml
from robotInfo import Config
import numpy as np
from pathlib import Path


def sorting_result() -> None:
    all_result = os.listdir("./[Result]/")
    for res in all_result:
        path_res = f"./[Result]/{res}"
        if have_config_file(path_res):
            res_config = result_configs(path_res)


def have_config_file(path: str) -> bool:
    config_file_name = "CONFIG.yml"
    # files = os.listdir(path)
    if os.path.isfile(f"{path}/{config_file_name}"):
        return True
    return False
    # if config_file_name in files:


def result_configs(path: str) -> Tuple:
    try:
        cf = Config(f"{path}/CONFIG.yml")
        points = np.genfromtxt(f"{path}/output_point.csv", delimiter=",")
        points_count = points.shape[0]
        with open(f"{path}/GA_PARAM.yml", "r") as ga_param_file:
            ga_param = yaml.load(ga_param_file)
        GEN_LIST: List = ga_param["GEN_LIST"]
        NIND: int = ga_param["NIND"]
        if len(GEN_LIST) == 1:
            is_sliced = False
    except Exception:
        return None
    return cf.replace_chromo, GEN_LIST[-1], is_sliced, points_count, cf.replace_chromo_dist


def main() -> None:
    sorting_result()


if __name__ == "__main__":
    main()
