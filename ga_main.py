import geatpy as ea
from moea_NSGA3_modified import moea_NSGA3_modified
from ga_Problem import MyProblem
import datetime
import shutil
import os


if __name__ == "__main__":
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
    try:
        if os.path.isdir("./[Result]/Result"):
            raise RuntimeError('Rename "Result" folder')
    except RuntimeError as e:
        print(repr(e))
        raise

    CONFIG_PATH = "./config.yml"
    GEN_LIST = [2000, 500, 100]
    NIND = 50

    num_slicing = len(GEN_LIST)
    passTime_sec = 0
    for step in range(num_slicing):
        problem = MyProblem(step, num_slicing, CONFIG_PATH)
        Encoding = "P"
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        population = ea.Population(Encoding, Field, NIND)
        print(Field)
        myAlgorithm = moea_NSGA3_modified(problem, population)
        myAlgorithm.MAXGEN = GEN_LIST[step]
        myAlgorithm.verbose = True
        myAlgorithm.drawing = 0
        if step + 1 == num_slicing:
            myAlgorithm.drawing = 1
        [NDSet, population] = myAlgorithm.run()
        NDSet.save()
        if os.path.exists(f"./log/Step_{step}"):
            shutil.rmtree(f"./log/Step_{step}")
        shutil.copytree("./Result", f"./log/Step_{step}")
        passTime_sec += myAlgorithm.passTime

    print("time: %f 秒" % passTime_sec)
    file = open(f"./Result/info.txt", "a")
    file.write(f'\n{"number of generation:":<25}{GEN_LIST}\n')
    file.write(f'{"number of chromosome:":<25}{NIND}\n')
    file.write(
        f'\n{"pass time:":<25}\
        {datetime.timedelta(seconds=passTime_sec)}({passTime_sec} secs)\n'
    )
    file.close()
    shutil.move("./Pareto Front.svg", "./Result/Pareto Front.svg")
    shutil.copyfile("./output_point.csv", "./Result/output_point.csv")
    shutil.copyfile("./config.yml", "./Result/config.yml")
    shutil.copytree("./log", "./Result/log")
    shutil.copytree(
        "./Result", f"./[Result]/{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
    )
