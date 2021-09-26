"""
Basic unit commitment to some mix-integer linear/quadratic programming problem
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
@date:6 Mar 2018
"""
from numpy import zeros, shape, ones, diag, concatenate, append, matlib
import matplotlib.pyplot as plt
from solvers.mixed_integer_quadratic_programming import mixed_integer_quadratic_programming as miqp


def problem_formulation(case):
    """
    :param case: The test case for unit commitment problem
    :return:
    """
    from unit_commitment.data_format.data_format import IG, PG
    from unit_commitment.test_cases.case118 import F_BUS, T_BUS
    from unit_commitment.test_cases.case118 import GEN_BUS, COST_C, COST_B, COST_A, PG_MAX, PG_MIN, I0, MIN_DOWN, \
        MIN_UP, RU, RD, COLD_START
    from unit_commitment.test_cases.case118 import BUS_ID, PD
    baseMVA, bus, gen, profile = case["baseMVA"], case["bus"], case["gen"], case["Load_profile"]

    # Modify the bus, gen and branch matrix
    bus[:, BUS_ID] = bus[:, BUS_ID] - 1
    gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1

    ng = shape(case['gen'])[0]  # number of schedule injections
    # Formulate a mixed integer quadratic programming problem
    # 1) Announce the variables
    # 1.1) boundary information
    T = case["Load_profile"].shape[0]
    lb = append(zeros((ng, 1)), gen[:, PG_MIN])
    ub = append(ones((ng, 1)), gen[:, PG_MAX])
    LB = matlib.repmat(lb, 1, T)
    UB = matlib.repmat(ub, 1, T)
    nx = LB.size
    NX = 2 * ng
    # 1.2) boundary information
    vtypes = []
    for i in range(T):
        vtypes += ["B"] * ng
        vtypes += ["C"] * ng
    # 1.3) objective information
    c = append(gen[:, COST_C], gen[:, COST_B])
    C = matlib.repmat(c, 1, T)
    q = append(zeros((ng, 1)), gen[:, COST_A])
    Q = matlib.repmat(q, 1, T)
    Q = diag(Q[0])
    # 2) Constraint set
    # 2.1) Power balance equation
    Aeq = zeros((T, nx))
    for i in range(T):
        Aeq[i, i * NX + ng:(i + 1) * NX] = 1
    beq = zeros((T, 1))
    for i in range(T):
        beq[i] = case["Load_profile"][i]
    # 2.2) Power range limitation
    Aineq = zeros((T * ng, nx))
    bineq = zeros((T * ng, 1))
    for i in range(T):
        for j in range(ng):
            Aineq[i + j, i * NX + j] = gen[j, PG_MIN]
            Aineq[i + j, i * NX + ng + j] = -1

    Aineq_temp = zeros((T * ng, nx))
    bineq_temp = zeros((T * ng, 1))
    for i in range(T):
        for j in range(ng):
            Aineq_temp[i + j, i * NX + j] = -gen[j, PG_MAX]
            Aineq_temp[i + j, i * NX + ng + j] = 1

    # plt.plot(LB[0])
    # plt.show()
    model = {}
    model["c"] = C[0]
    model["Q"] = Q
    model["Aeq"] = Aeq
    model["beq"] = beq
    model["lb"] = LB[0]
    model["ub"] = UB[0]
    model["Aineq"] = concatenate((Aineq, Aineq_temp), axis=0)
    model["bineq"] = append(bineq, bineq_temp)
    model["vtypes"] = vtypes
    return model


if __name__ == "__main__":
    from unit_commitment.test_cases import case118

    test_case = case118.case118()
    model = problem_formulation(test_case)
    (xx, obj, success) = miqp(c=model["c"], Q=model["Q"], Aeq=model["Aeq"], A=model["Aineq"], b=model["bineq"],
                              beq=model["beq"], xmin=model["lb"],
                              xmax=model["ub"], vtypes=model["vtypes"])
    plt.plot(xx)
    plt.show()
