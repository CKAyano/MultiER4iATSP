import datetime
import numpy as np


class Collision_status:
    def __init__(self, step, path=None) -> None:
        self.step = step
        self.filename_status = f"status_{self.step}"
        self.filename_chrom = f"chrom_{self.step}"
        if path is None:
            self.path_status = f"./log/{self.filename_status}.log"
            self.path_chrom = f"./log/{self.filename_chrom}.csv"
            open(f"./log/{self.filename_status}.log", "w").close()
            open(f"./log/{self.filename_chrom}.csv", "w").close()
        else:
            self.path_status = f"{path}/{self.filename_status}.log"
            self.path_chrom = f"{path}/{self.filename_chrom}.csv"
        open(self.path_status, "w").close()
        open(self.path_chrom, "w").close()

    def save_status(self, msg):
        with open(self.path_status, "a") as file:
            # file = open(self.path_status, "a")
            timestamp = str(datetime.datetime.now())
            file.write(f"{timestamp}\t{msg}\n")
        # file.close()

    def save_chrom(self, chrom):
        np.savetxt(self.path_chrom, chrom, delimiter=",")
