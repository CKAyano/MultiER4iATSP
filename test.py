from chromoCalcV3 import ChromoCalcV3, Trajectory
from robotCalc_pygeos import RobotCalc_pygeos
import numpy as np
from robot_configuration import PumaKinematics, FanucKinematics
from robot_configuration import Config, Coord, Position, Robot
from validity import DrawRobots


def main() -> None:
    angle_start = np.radians(np.array([20, 30, 40]))
    print(angle_start)
    angle_end = np.radians(np.array([40, 60, 80]))
    print(angle_end)

    mean_ang_v = np.radians(100)
    # print(f"{mean_ang_v:.4f}")
    _, s_all = Trajectory.get_trajectory(
        angle_start, angle_end, mean_ang_v, step_period=1 / 20, is_test=True
    )
    s_all_out = s_all[1:, :]
    print(s_all)


def ik_test():
    config = Config("./CONFIG.yml")
    config.zyx_euler = np.array([0, 0, 0])
    rb = RobotCalc_pygeos(config)
    p = Coord(200, 100, 50)
    ik = rb.userIK(p)
    for i in ik:
        i_deg = np.round(np.degrees(i), 4)
        print(list(i_deg))
        # print("out of range\n" if rb.cv_joints_range(i) else "good\n")
    # fk = rb.userFK(np.array([0, 0, 0, 0, 0, 0]))
    # print(fk.end_effector.coordToNp())


def fanuc_test():

    coords = np.array(
        [
            [-373.208749836348, 228.955231973641, 196.024700331349],
            [-36.1572633758018, 22.1816734665755, 484.919647747134],
            [334.651504889447, -205.301223419573, 275.664162125156],
            [373.530673209224, -229.152724772368, -211.92051341263],
            [-390.739595247152, -197.55464537339, 196.024700331349],
            [-37.8556892433541, -19.1395173532534, 484.919647747134],
            [350.371189385811, 177.144719680597, 275.664162125156],
            [391.0766404222, 197.725052572561, -211.92051341263],
        ]
    )
    fanuc = FanucKinematics()
    i = 2
    p = Coord(coords[i, 0], coords[i, 1], coords[i, 2])
    fanuc._validate(p, [0, 0, np.pi])
    direction = np.radians(np.array([30, 30, 30]))
    ik = fanuc.inverse_kines(p, direction)
    for i in ik:
        i_deg = np.round(np.degrees(i), 4)
        print(list(i_deg))

    # fanuc._validate(p, np.array([0, 0, -3.14159265]))


def puma_test():
    puma = PumaKinematics()
    # p = Coord(600, 149.09, 200)
    # p = Coord(500, 240, 230)
    # p = Coord(540, 210, 260)
    # p = Coord(180, -400, 400)
    p = Coord(-180, 400, -200)
    ik = puma.inverse_kines(p, np.array([0, 0, -3.14159265]))
    for i in ik:
        i_deg = np.round(np.degrees(i), 4)
        print(list(i_deg))

    # puma._validate(p, [0, 0, -3.14159265])


def test_collision():
    dr = DrawRobots(
        "./all_results/fanuc/Robot_4/points_count100/"
        + "noStep/Gen5000/replace/Hamming20/poly_traj/220606-150353"
    )
    robots = []
    position = [Position.LEFT, Position.RIGHT, Position.UP, Position.DOWN]
    config = Config("./CONFIG.yml")
    for i in range(config.robots_count):
        robots.append(Robot(i, position[i]))
    q1 = np.array(
        [
            [
                -0.4687295299717741,
                1.18680336559566,
                -0.9551351965407516,
                3.237707442091745e-09,
                1.3391281593269622,
                -0.468729530743935,
            ]
        ]
    )
    q2 = np.array(
        [
            [
                0.27449472976025263,
                0.8792894195204223,
                -0.40296621020486306,
                3.798087576817292e-09,
                1.0944731165354042,
                0.27449472803526315,
            ]
        ]
    )
    q3 = np.array([[1.5707963267948966, 0.7853981633974483, 0.0, 0.0, 1.5707963267948966, 0.0]])
    q4 = np.array([[1.5707963267948966, 0.7853981633974483, 0.0, 0.0, 1.5707963267948966, 0.0]])
    rc = RobotCalc_pygeos(config)
    dr.rc = rc
    dr.config = config
    # dr.config.link_width /= 2
    for i in range(len(dr.rc.robot_kine.links_width)):
        dr.rc.robot_kine.links_width[i] /= 2
    dr.draw([q1, q2, q3, q4], 0)
    print(np.degrees(q1))
    print(np.degrees(q2))

    # print(rc.cv_collision(q1, q2, robots[0], robots[1]))


if __name__ == "__main__":
    # main()
    # ik_test()
    fanuc_test()
    # test_collision()
    # puma_test()
