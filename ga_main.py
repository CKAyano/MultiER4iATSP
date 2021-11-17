from typing import List
import geatpy as ea
from moea_NSGA3_modified import moea_NSGA3_modified
from ga_Problem import MyProblem
import datetime
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import gc


CONFIG_PATH: str = "./config.yml"
# GEN_LIST: List = [2000, 500, 100]
GEN_LIST: List = [3000]
NIND: int = 50


def del_result_contents() -> None:
    folder = "./Result"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def save_feasibleSol_figure(feasibleSol_list):
    plot_x = [i + 1 for i in range(len(feasibleSol_list))]
    plt.plot(plot_x, feasibleSol_list, "-r")
    plt.title("Feasible Solution")
    plt.xlabel("Generation")
    plt.ylabel("Feasible Solution")
    plt.savefig("./Result/feasibleSol.png")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def save_pareto() -> None:
    objv_path = "./Result/ObjV.csv"
    objv = np.genfromtxt(objv_path, delimiter=",")
    plt.plot(objv[:, 1], objv[:, 0], ".r")
    plt.title("Pareto Front")
    plt.xlabel("std of every robots' angle changes")
    plt.ylabel("total angle changes of every robots")
    plt.savefig("./Result/pareto.png")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def save_status(passTime_sec) -> None:
    file = open(f"./Result/info.txt", "a")
    file.write(f'\n{"number of generation:":<25}{GEN_LIST}\n')
    file.write(f'{"number of chromosome:":<25}{NIND}\n')
    file.write(
        f'\n{"pass time:":<25}' + f"{datetime.timedelta(seconds=passTime_sec)}({passTime_sec} secs)\n"
    )
    file.close()
    shutil.copyfile("./output_point.csv", "./Result/output_point.csv")
    shutil.copyfile("./config.yml", "./Result/config.yml")
    shutil.copytree("./log", "./Result/log")


def main() -> None:
    del_result_contents()

    num_slicing = len(GEN_LIST)
    passTime_sec = 0
    feasibleSol_list = []
    for step in range(num_slicing):
        problem = MyProblem(step, num_slicing, CONFIG_PATH, feasibleSol_list)
        Encoding = "P"
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        population = ea.Population(Encoding, Field, NIND)
        print(Field)
        myAlgorithm = moea_NSGA3_modified(problem, population)
        myAlgorithm.MAXGEN = GEN_LIST[step]
        myAlgorithm.verbose = True
        myAlgorithm.drawing = 0
        # if step + 1 == num_slicing:
        #     myAlgorithm.drawing = 1
        [NDSet, population] = myAlgorithm.run()
        NDSet.save()
        if os.path.exists(f"./log/Step_{step}"):
            shutil.rmtree(f"./log/Step_{step}")
        shutil.copytree("./Result", f"./log/Step_{step}")
        passTime_sec += myAlgorithm.passTime
        feasibleSol_list = problem.ccv3.feasibleSol_list

    save_pareto()
    save_feasibleSol_figure(feasibleSol_list)
    save_status(passTime_sec)
    shutil.copytree("./Result", f"./[Result]/{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}")


def del_memory():
    for name in dir():
        del globals()[name]
    gc.collect()


if __name__ == "__main__":
    # for _ in range(6):
    main()
    # del_memory()
