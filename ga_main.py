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
# GEN_LIST: List = [3000, 800, 200]
GEN_LIST: List = [5000]
NIND: int = 50


def del_result_contents() -> None:
    folder = "./Result"
    for filename in os.listdir(folder):
        if filename != "info.txt":
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


def del_log_contents() -> None:
    folder = "./log"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def save_feasibleSol_figure(feasibleSol_list) -> None:
    plot_x = [i + 1 for i in range(len(feasibleSol_list))]
    fig, ax = plt.subplots()
    ax.plot(plot_x, feasibleSol_list, "-r")
    for i in GEN_LIST:
        ax.axvline(x=i, color="gray", alpha=0.6)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Feasible Solution")
    fig.suptitle("Feasible Solution")
    fig.savefig("./Result/feasibleSol.png")
    plt.close()
    plt.cla()
    plt.clf()


def save_pareto(objv_path) -> None:
    if os.path.exists(objv_path):
        objv = np.genfromtxt(objv_path, delimiter=",")
        plt.plot(objv[:, 1], objv[:, 0], ".r")
        plt.title("Pareto Front")
        plt.xlabel("std of every robots' angle changes")
        plt.ylabel("total angle changes of every robots")
        plt.savefig("./Result/pareto.png")
        plt.close()
        plt.cla()
        plt.clf()
    else:
        with open("./Result/pareto.txt", "w") as file:
            file.write("No solution")


def save_status(passTime_sec) -> None:
    with open("./Result/info.txt", "a") as file:
        file.write(f'\n{"number of generation:":<25}{GEN_LIST}\n')
        file.write(f'{"number of chromosome:":<25}{NIND}\n')
        file.write(
            f'\n{"pass time:":<25}' + f"{datetime.timedelta(seconds=passTime_sec)}({passTime_sec} secs)\n"
        )
    shutil.copyfile("./output_point.csv", "./Result/output_point.csv")
    shutil.copyfile("./config.yml", "./Result/config.yml")
    shutil.copytree("./log", "./Result/log")


def main() -> None:
    del_result_contents()
    del_log_contents()

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
        [NDSet, population] = myAlgorithm.run()
        NDSet.save()
        if os.path.exists(f"./log/Step_{step}"):
            shutil.rmtree(f"./log/Step_{step}")
        shutil.copytree("./Result", f"./log/Step_{step}")
        passTime_sec += myAlgorithm.passTime
        feasibleSol_list = problem.ccv3.feasibleSol_list
        if step != num_slicing - 1:
            del_result_contents()

    save_pareto("./Result/ObjV.csv")
    save_feasibleSol_figure(feasibleSol_list)
    save_status(passTime_sec)
    shutil.copytree("./Result", f"./[Result]/{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}")


if __name__ == "__main__":
    main()
