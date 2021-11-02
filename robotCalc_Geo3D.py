import numpy as np
import Geometry3Dmaster.Geometry3D as gm
import time
import timeit
# import Geometry3D as gm

# d_1 = 330
d_1 = 0
a_2 = 260
a_3 = 20
d_4 = 290
linkWidth = 200
baseX_offset = 650


def userIK(vv):
    def angleAdj(ax):
        for ii in range(len(ax)):
            while ax[ii] > np.pi:
                ax[ii] = ax[ii] - np.pi * 2

            while ax[ii] < -np.pi:
                ax[ii] = ax[ii] + np.pi * 2
        return ax

    px = vv["xx"]
    py = vv["yy"]
    pz = vv["zz"]

    q1 = np.zeros(2)
    q2 = np.zeros(4)
    q3 = np.zeros(2)
    q23 = np.zeros(4)
    numq2 = -1

    q1[0] = np.arctan2(py, px) - np.arctan2(0, np.sqrt(np.square(px) + np.square(py)))
    q1[1] = np.arctan2(py, px) - np.arctan2(0, -np.sqrt(np.square(px) + np.square(py)))

    k = (
        np.square(d_1)
        - 2 * d_1 * pz
        + np.square(px)
        + np.square(py)
        + np.square(pz)
        - np.square(a_3)
        - np.square(d_4)
        - np.square(a_2)
    ) / (2 * a_2)

    q3[0] = np.arctan2(a_3, d_4) - np.arctan2(
        k, np.sqrt(np.square(a_3) + np.square(d_4) - np.square(k))
    )
    q3[1] = np.arctan2(a_3, d_4) - np.arctan2(
        k, -np.sqrt(np.square(a_3) + np.square(d_4) - np.square(k))
    )

    for jj in range(2):
        for ii in range(2):
            numq2 = numq2 + 1
            q23[numq2] = np.arctan2(
                (
                    d_1 * d_4
                    - d_4 * pz
                    + a_3 * px * np.cos(q1[jj])
                    - a_2 * d_1 * np.sin(q3[ii])
                    + a_3 * py * np.sin(q1[jj])
                    + a_2 * pz * np.sin(q3[ii])
                    + a_2 * px * np.cos(q1[jj]) * np.cos(q3[ii])
                    + a_2 * py * np.cos(q3[ii]) * np.sin(q1[jj])
                ),
                -(
                    a_3 * d_1
                    - a_3 * pz
                    - a_2 * pz * np.cos(q3[ii])
                    - d_4 * px * np.cos(q1[jj])
                    - d_4 * py * np.sin(q1[jj])
                    + a_2 * d_1 * np.cos(q3[ii])
                    + a_2 * py * np.sin(q1[jj]) * np.sin(q3[ii])
                    + a_2 * px * np.cos(q1[jj]) * np.sin(q3[ii])
                ),
            )
            q2[numq2] = q23[numq2] - q3[ii]

    q1 = angleAdj(q1)
    q2 = angleAdj(q2)
    q3 = angleAdj(q3)

    q1 = np.array([q1[0], q1[0], q1[1], q1[1]])
    q3 = np.array([q3[0], q3[1], q3[0], q3[1]])

    group_1 = np.hstack((q1[0], q2[0], q3[0]))
    group_2 = np.hstack((q1[1], q2[1], q3[1]))
    group_3 = np.hstack((q1[2], q2[2], q3[2]))
    group_4 = np.hstack((q1[3], q2[3], q3[3]))

    q = {"group_1": group_1, "group_2": group_2, "group_3": group_3, "group_4": group_4}
    return q


def greedySearch(q_f_best, q):
    q_array = np.vstack((q["group_1"], q["group_2"], q["group_3"], q["group_4"]))
    diff_q1 = np.absolute(q_array[:, 0] - q_f_best[0])
    diff_q2 = np.absolute(q_array[:, 1] - q_f_best[1])
    diff_q3 = np.absolute(q_array[:, 2] - q_f_best[2])
    jointDiff = np.vstack((diff_q1, diff_q2, diff_q3))
    jointMax_index = np.argmax(jointDiff, axis=0)
    for ii in range(4):
        if ii == 0:
            element_jointMax = jointDiff[jointMax_index[ii], ii]
        else:
            element_jointMax = np.hstack(
                (element_jointMax, jointDiff[jointMax_index[ii], ii])
            )
    bestGroup_index = np.argmin(element_jointMax, axis=0)
    q_best = q_array[bestGroup_index, :]

    return q_best


def userFK(q):

    c1 = np.cos(q[0])
    s1 = np.sin(q[0])
    c2 = np.cos(q[1])
    s2 = np.sin(q[1])
    c3 = np.cos(q[2])
    s3 = np.sin(q[2])
    c4 = np.cos(0)
    s4 = np.sin(0)

    axisM1 = np.array([[c1, 0, -s1, 0], [s1, 0, c1, 0], [0, -1, 0, d_1], [0, 0, 0, 1]])

    axisM2 = np.array(
        [[s2, c2, 0, a_2 * s2], [-c2, s2, 0, -a_2 * c2], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    axisM3 = np.array(
        [[c3, 0, -s3, a_3 * c3], [s3, 0, c3, a_3 * s3], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

    axisM4 = np.array([[c4, 0, s4, 0], [s4, 0, -c4, 0], [0, 1, 0, d_4], [0, 0, 0, 1]])

    fk1 = axisM1
    fk2 = fk1.dot(axisM2)
    fk3 = fk2.dot(axisM3)
    fk4 = fk3.dot(axisM4)

    v1 = {"xx": 0, "yy": 0, "zz": 0}
    v2 = {"xx": fk1[0, 3], "yy": fk1[1, 3], "zz": fk1[2, 3]}
    v3 = {"xx": fk2[0, 3], "yy": fk2[1, 3], "zz": fk2[2, 3]}
    v4 = {"xx": fk3[0, 3], "yy": fk3[1, 3], "zz": fk3[2, 3]}
    v5 = {"xx": fk4[0, 3], "yy": fk4[1, 3], "zz": fk4[2, 3]}
    # v1 = {'xx': 0, 'yy': 0, 'zz': 0}
    # v2 = {'xx': fk1[0, 3], 'yy': fk1[1, 3], 'zz': fk1[2, 3]}
    # v3 = {'xx': fk2[0, 3], 'yy': fk2[1, 3], 'zz': fk2[2, 3]}
    # v4 = {'xx': fk3[0, 3], 'yy': fk3[1, 3], 'zz': fk3[2, 3]}
    # v5 = {'xx': fk4[0, 3], 'yy': fk4[1, 3], 'zz': fk4[2, 3]}

    v_all = {"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5}

    # vv = v_all.get(vi)

    return v_all


def robot2world(vv_a):
    vv_b = {"xx": 0, "yy": 0, "zz": 0}
    vv_b["xx"] = -vv_a["xx"] + baseX_offset
    vv_b["yy"] = -vv_a["yy"]
    vv_b["zz"] = vv_a["zz"]
    return vv_b


def secRb2WorldCoor(v_all):

    v1 = robot2world(v_all["v1"])
    v2 = robot2world(v_all["v2"])
    v3 = robot2world(v_all["v3"])
    v4 = robot2world(v_all["v4"])
    v5 = robot2world(v_all["v5"])

    v_all = {"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5}

    return v_all


def cvAxisRange(q_best, axisRange):
    axisRange = np.radians(axisRange)
    isQNan = np.isnan(q_best)
    if np.any(isQNan):
        return True
    else:
        for i in range(3):
            condition_ql = q_best[:, i] < axisRange[i, 0]
            condition_qu = q_best[:, i] > axisRange[i, 1]
            if np.any(condition_ql) or np.any(condition_qu):
                return True
    return False


def cvCollision(qa_best, qb_best, path):

    # % ------------------------- qa ------------------------- % #
    point = generateLinkWidenPoint_2(qa_best)
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
    cph1 = gm.geometry.ConvexPolyhedron(
        (cpg1, cpg2, cpg3, cpg4, cpg5, cpg6)
    )
    cph2 = gm.geometry.ConvexPolyhedron(
        (cpg7, cpg8, cpg9, cpg10, cpg11, cpg12)
    )

    # % ------------------------- qb ------------------------- % #
    vb_all = userFK(qb_best)
    vb_all = secRb2WorldCoor(vb_all)
    # print(vb_all["v5"])
    gm_vb1 = gm.geometry.Point(
        vb_all["v2"]["xx"], vb_all["v2"]["yy"], vb_all["v2"]["zz"]
    )
    gm_vb3 = gm.geometry.Point(
        vb_all["v4"]["xx"], vb_all["v4"]["yy"], vb_all["v4"]["zz"]
    )
    gm_vb4 = gm.geometry.Point(
        vb_all["v5"]["xx"], vb_all["v5"]["yy"], vb_all["v5"]["zz"]
    )

    gmSegB1 = gm.geometry.Segment(gm_vb1, gm_vb3)
    gmSegB2 = gm.geometry.Segment(gm_vb3, gm_vb4)

    # % -------------------- intersection -------------------- % #
    try:
        inter1 = gm.calc.intersection(cph1, gmSegB1)
        inter2 = gm.calc.intersection(cph2, gmSegB1)
        inter3 = gm.calc.intersection(cph1, gmSegB2)
        inter4 = gm.calc.intersection(cph2, gmSegB2)
    except Exception:
        return True
    else:
        strInter1 = str(inter1)
        strInter2 = str(inter2)
        strInter3 = str(inter3)
        strInter4 = str(inter4)

        r = gm.render.Renderer()
        r.add((cph1, "r", 1), normal_length=0)
        r.add((cph2, "r", 1), normal_length=0)
        r.add((gmSegB1, "b", 1), normal_length=0)
        r.add((gmSegB2, "b", 1), normal_length=0)
        if strInter1 == "None":
            pass
        else:
            r.add((inter1, "g", 5), normal_length=0)
        if strInter2 == "None":
            pass
        else:
            r.add((inter2, "g", 5), normal_length=0)
        if strInter3 == "None":
            pass
        else:
            r.add((inter3, "g", 5), normal_length=0)
        if strInter4 == "None":
            pass
        else:
            r.add((inter4, "g", 5), normal_length=0)
        r.savefigure(path)

        if strInter1 == "None" and strInter2 == "None" \
           and strInter3 == "None" and strInter4 == "None":
            return False
        else:
            return True


def generateLinkWidenPoint_2(q_best):
    def calcNormalVec(v_all, v1, v2):
        v1_point = np.array(list(v_all.get(v1).values()))
        v2_point = np.array(list(v_all.get(v2).values()))
        if q_best[0] < np.pi/2+0.0001 and q_best[0] > np.pi/2-0.0001:
            normed_normalVec_1 = np.array([1, 0, 0])
            normed_normalVec_2 = np.array([0, 1, 0])
        elif q_best[0] < -np.pi/2+0.0001 and q_best[0] > -np.pi/2-0.0001:
            normed_normalVec_1 = np.array([1, 0, 0])
            normed_normalVec_2 = np.array([0, -1, 0])
        else:
            vector_1 = np.array([np.tan(q_best[0]), -1, 0])
            vector_2 = v2_point - v1_point
            vector_3 = np.array([1, np.tan(q_best[0]), 0])
            vecCross_1 = np.cross(vector_1, vector_2)
            length_vec_1 = np.linalg.norm(vecCross_1)
            normed_normalVec_1 = vecCross_1 / length_vec_1
            vecCross_2 = np.cross(vector_3, vector_2)
            length_vec_2 = np.linalg.norm(vecCross_2)
            # np.seterr(all='raise')
            normed_normalVec_2 = vecCross_2 / length_vec_2
            # np.seterr(all='warn')
        return v1_point, v2_point, normed_normalVec_1, normed_normalVec_2
    v_all = userFK(q_best)
    v2_point, v4_point, normalVec_1, normalVec_2 = calcNormalVec(v_all, 'v2', 'v4')
    _, v5_point, normalVec_3, normalVec_4 = calcNormalVec(v_all, 'v4', 'v5')
    p1 = v2_point + (-normalVec_1 + normalVec_2) * linkWidth/2
    p2 = v2_point + (normalVec_1 + normalVec_2) * linkWidth/2
    p3 = v2_point + (normalVec_1 - normalVec_2) * linkWidth/2
    p4 = v2_point + (-normalVec_1 - normalVec_2) * linkWidth/2
    p5 = v4_point + (-normalVec_1 + normalVec_2) * linkWidth/2
    p6 = v4_point + (normalVec_1 + normalVec_2) * linkWidth/2
    p7 = v4_point + (normalVec_1 - normalVec_2) * linkWidth/2
    p8 = v4_point + (-normalVec_1 - normalVec_2) * linkWidth/2

    p9 = v4_point + (-normalVec_3 + normalVec_4) * linkWidth/2
    p10 = v4_point + (normalVec_3 + normalVec_4) * linkWidth/2
    p11 = v4_point + (normalVec_3 - normalVec_4) * linkWidth/2
    p12 = v4_point + (-normalVec_3 - normalVec_4) * linkWidth/2
    p13 = v5_point + (-normalVec_3 + normalVec_4) * linkWidth/2
    p14 = v5_point + (normalVec_3 + normalVec_4) * linkWidth/2
    p15 = v5_point + (normalVec_3 - normalVec_4) * linkWidth/2
    p16 = v5_point + (-normalVec_3 - normalVec_4) * linkWidth/2
    point = (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16)
    return point


if __name__ == "__main__":
    qa_best = np.array([0, 45, -40])
    qa_best = np.radians(qa_best)
    qb_best = qa_best.copy()
    va = userFK(qa_best)
    # test = generateLinkWidenPoint_2(qa_best)
    # an_array = np.array(list(va.get('v3').values()))
    # qb_best = qa_best.copy()
    test = cvCollision(qa_best, qb_best)
