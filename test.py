from chromoCalcV3 import ChromoCalcV3
import numpy as np
from robotInfo import Config, Coord
from robotCalc_pygeos import RobotCalc_pygeos

config = Config("./config.yml")
rc = RobotCalc_pygeos(config)


rc.coord2bestAngle()
