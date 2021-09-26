"""
Augumented Lagrange decomposition for basic unit commitment problems
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
@date:21 Mar 2018

Note: The mathematical model is taken from the following references.
[1]
"""
from numpy import zeros, shape, ones, diag, concatenate, r_, arange, divide, linalg
import matplotlib.pyplot as plt
from solvers.mixed_integer_quadratic_programming import mixed_integer_quadratic_programming as miqp

from scipy.sparse import csr_matrix as sparse
from pypower import loadcase, ext2int


def problem_formulation(case, Distribution_factor, Cg, Cd):
    """
    :param case: The test case for unit commitment problem
    :return:
    """
    from unit_commitment.data_format.data_format import IG, PG
    from unit_commitment.test_cases.case118 import F_BUS, T_BUS, BR_X, RATE_A
    from unit_commitment.test_cases.case118 import GEN_BUS, COST_C, COST_B, COST_A, PG_MAX, PG_MIN, I0, MIN_DOWN, \
        MIN_UP, RU, RD, COLD_START
    from unit_commitment.test_cases.case118 import BUS_ID, PD
    baseMVA, bus, gen, branch, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"], case["Load_profile"]

    # Modify the bus, gen and branch matrix
    bus[:, BUS_ID] = bus[:, BUS_ID] - 1
    gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1
    branch[:, F_BUS] = branch[:, F_BUS] - 1
    branch[:, T_BUS] = branch[:, T_BUS] - 1

    ng = shape(case['gen'])[0]  # number of schedule injections
    nl = shape(case['branch'])[0]  ## number of branches
    nb = shape(case['bus'])[0]  ## number of branches

    u0 = [0] * ng  # The initial generation status
    for i in range(ng):
        u0[i] = int(gen[i, I0] > 0)
    # Formulate a mixed integer quadratic programming problem
    # 1) Announce the variables
    # [vt,wt,ut,Pt]:start-up,shut-down,status,generation level
    # 1.1) boundary information
    T = case["Load_profile"].shape[0]
    lb = []
    for i in range(ng):
        lb += [0] * T
        lb += [0] * T
        lb += [0] * T
        lb += [0] * T
    ub = []
    for i in range(ng):
        ub += [1] * T
        ub += [1] * T
        ub += [1] * T
        ub += [gen[i, PG_MAX]] * T
    nx = len(lb)
    NX = 4 * T  # The number of decision variables for each unit
    # 1.2) variable information
    vtypes = []
    for i in range(ng):
        vtypes += ["C"] * T
        vtypes += ["C"] * T
        vtypes += ["B"] * T
        vtypes += ["C"] * T
    # 1.3) objective information
    c = []
    q = []
    for i in range(ng):
        c += [gen[i, COLD_START]] * T
        c += [0] * T
        c += [gen[i, COST_C]] * T
        c += [gen[i, COST_B]] * T

        q += [0] * T
        q += [0] * T
        q += [0] * T
        q += [gen[i, COST_A]] * T

    Q = diag(q)

    # 2) Constraint set
    # 2.1) Power balance equation
    Aeq = zeros((T, nx))
    for i in range(T):
        for j in range(ng):
            Aeq[i, j * NX + 3 * T + i] = 1
    beq = [0] * T
    for i in range(T):
        beq[i] = case["Load_profile"][i]

    # 2.2) Status transformation of each unit
    Aeq_temp = zeros((T * ng, nx))
    beq_temp = [0] * T * ng
    for i in range(ng):
        for j in range(T):
            Aeq_temp[i * T + j, i * NX + j] = 1
            Aeq_temp[i * T + j, i * NX + j + T] = -1
            Aeq_temp[i * T + j, i * NX + j + 2 * T] = -1
            if j != 0:
                Aeq_temp[i * T + j, i * NX + j - 1 + 2 * T] = 1
            else:
                beq_temp[i * T + j] = -u0[i]

    Aeq = concatenate((Aeq, Aeq_temp), axis=0)
    beq += beq_temp
    # 2.3) Power range limitation
    Aineq = zeros((T * ng, nx))
    bineq = [0] * T * ng
    for i in range(ng):
        for j in range(T):
            Aineq[i * T + j, i * NX + 2 * T + j] = gen[i, PG_MIN]
            Aineq[i * T + j, i * NX + 3 * T + j] = -1

    Aineq_temp = zeros((T * ng, nx))
    bineq_temp = [0] * T * ng
    for i in range(ng):
        for j in range(T):
            Aineq_temp[i * T + j, i * NX + 2 * T + j] = -gen[i, PG_MAX]
            Aineq_temp[i * T + j, i * NX + 3 * T + j] = 1
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq += bineq_temp

    # 2.4) Start up and shut down time limitation
    UP_LIMIT = [0] * ng
    DOWN_LIMIT = [0] * ng
    for i in range(ng):
        UP_LIMIT[i] = T - int(gen[i, MIN_UP])
        DOWN_LIMIT[i] = T - int(gen[i, MIN_DOWN])
    # 2.4.1) Up limit
    Aineq_temp = zeros((sum(UP_LIMIT), nx))
    bineq_temp = [0] * sum(UP_LIMIT)

    for i in range(ng):
        for j in range(int(gen[i, MIN_UP]), T):
            Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gen[i, MIN_UP]), i * NX + j - int(gen[i, MIN_UP]):i * NX + j] = 1
            Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gen[i, MIN_UP]), i * NX + 2 * T + j] = -1
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq += bineq_temp
    # 2.4.2) Down limit
    Aineq_temp = zeros((sum(DOWN_LIMIT), nx))
    bineq_temp = [1] * sum(DOWN_LIMIT)
    for i in range(ng):
        for j in range(int(gen[i, MIN_DOWN]), T):
            Aineq_temp[sum(DOWN_LIMIT[0:i]) + j - int(gen[i, MIN_DOWN]),
            i * NX + T + j - int(gen[i, MIN_DOWN]):i * NX + T + j] = 1
            Aineq_temp[sum(DOWN_LIMIT[0:i]) + j - int(gen[i, MIN_DOWN]), i * NX + 2 * T + j] = 1
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq += bineq_temp

    # 2.5) Ramp constraints:
    # 2.5.1) Ramp up limitation
    Aineq_temp = zeros((ng * (T - 1), nx))
    bineq_temp = [0] * ng * (T - 1)
    for i in range(ng):
        for j in range(T - 1):
            Aineq_temp[i * (T - 1) + j, i * NX + 3 * T + j + 1] = 1
            Aineq_temp[i * (T - 1) + j, i * NX + 3 * T + j] = -1
            Aineq_temp[i * (T - 1) + j, i * NX + 2 * T + j] = -gen[i, RU]
            Aineq_temp[i * (T - 1) + j, i * NX + j] = -gen[i, PG_MAX]

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq += bineq_temp
    # 2.5.2) Ramp up limitation
    Aineq_temp = zeros((ng * (T - 1), nx))
    bineq_temp = [0] * ng * (T - 1)
    for i in range(ng):
        for j in range(T - 1):
            Aineq_temp[i * (T - 1) + j, i * NX + 3 * T + j + 1] = -1
            Aineq_temp[i * (T - 1) + j, i * NX + 3 * T + j] = 1
            Aineq_temp[i * (T - 1) + j, i * NX + 2 * T + j + 1] = -gen[i, RD]
            Aineq_temp[i * (T - 1) + j, i * NX + T + j + 1] = -gen[i, PG_MIN]
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq += bineq_temp
    # 2.6) Line flow limitation
    # Add the line flow limitation time by time
    Aineq_temp = zeros((nl * T, nx))
    bineq_temp = [0] * nl * T
    for i in range(T):
        index = [0] * ng
        for j in range(ng):
            index[j] = j * 4 * T + 3 * T + i
        Cx2g = sparse((ones(ng), (arange(ng), index)), (ng, nx))
        Aineq_temp[i * nl:(i + 1) * nl, :] = (Distribution_factor * Cg * Cx2g).todense()
        PD_bus = bus[:, PD] * case["Load_profile"][i]
        bineq_temp[i * nl:(i + 1) * nl] = branch[:, RATE_A] + Distribution_factor * Cd * PD_bus
        del index, Cx2g
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq += bineq_temp

    Aineq_temp = zeros((nl * T, nx))
    bineq_temp = [0] * nl * T
    for i in range(T):
        index = [0] * ng
        for j in range(ng):
            index[j] = j * 4 * T + 3 * T + i
        Cx2g = sparse((-ones(ng), (arange(ng), index)), (ng, nx))
        Aineq_temp[i * nl:(i + 1) * nl, :] = (Distribution_factor * Cg * Cx2g).todense()
        PD_bus = bus[:, PD] * case["Load_profile"][i]
        bineq_temp[i * nl:(i + 1) * nl] = branch[:, RATE_A] - Distribution_factor * Cd * PD_bus
        del index, Cx2g
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq += bineq_temp

    model = {}
    model["c"] = c
    model["Q"] = Q
    model["Aeq"] = Aeq
    model["beq"] = beq
    model["lb"] = lb
    model["ub"] = ub
    model["Aineq"] = Aineq
    model["bineq"] = bineq
    model["vtypes"] = vtypes
    return model


def solution_decomposition(xx, obj, success):
    """
    Decomposition of objective functions
    :param xx: Solution
    :param obj: Objective value
    :param success: Success or not
    :return:
    """
    T = 24
    ng = 54
    result = {}
    result["success"] = success
    result["obj"] = obj
    if success:
        v = zeros((ng, T))
        w = zeros((ng, T))
        Ig = zeros((ng, T))
        Pg = zeros((ng, T))
        for i in range(ng):
            v[i, :] = xx[4 * i * T:4 * i * T + T]
            w[i, :] = xx[4 * i * T + T:4 * i * T + 2 * T]
            Ig[i, :] = xx[4 * i * T + 2 * T:4 * i * T + 3 * T]
            Pg[i, :] = xx[4 * i * T + 3 * T:4 * i * T + 4 * T]
        result["vt"] = v
        result["wt"] = w
        result["Ig"] = Ig
        result["Pg"] = Pg
    else:
        result["vt"] = 0
        result["wt"] = 0
        result["Ig"] = 0
        result["Pg"] = 0

    return result


if __name__ == "__main__":
    from unit_commitment.test_cases import case118

    test_case = case118.case118()

    from pypower.case118 import case118
    from pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
    from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
    from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I
    from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
    from numpy import flatnonzero as find

    casedata = case118()
    mpc = loadcase.loadcase(casedata)
    mpc = ext2int.ext2int(mpc)
    baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]  #

    nb = shape(mpc['bus'])[0]  ## number of buses
    nl = shape(mpc['branch'])[0]  ## number of branches
    ng = shape(mpc['gen'])[0]  ## number of dispatchable injections

    ## Formualte the
    stat = branch[:, BR_STATUS]  ## ones at in-service branches
    b = stat / branch[:, BR_X]  ## series susceptance
    tap = ones(nl)  ## default tap ratio = 1
    i = find(branch[:, TAP])  ## indices of non-zero tap ratios
    tap[i] = branch[i, TAP]  ## assign non-zero tap ratios

    ## build connection matrix Cft = Cf - Ct for line and from - to buses
    f = branch[:, F_BUS]  ## list of "from" buses
    t = branch[:, T_BUS]  ## list of "to" buses
    i = r_[range(nl), range(nl)]  ## double set of row indices
    ## connection matrix
    Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))

    ## build Bf such that Bf * Va is the vector of real branch powers injected
    ## at each branch's "from" bus
    Bf = sparse((r_[b, -b], (i, r_[f, t])), shape=(nl, nb))  ## = spdiags(b, 0, nl, nl) * Cft

    ## build Bbus
    Bbus = Cft.T * Bf
    # The distribution factor
    Distribution_factor = sparse(linalg.solve(Bbus.toarray().transpose(), Bf.toarray().transpose()).transpose())

    Cg = sparse((ones(ng), (gen[:, GEN_BUS], arange(ng))),
                (nb, ng))  # Sparse index generation method is different from the way of matlab
    Cd = sparse((ones(nb), (bus[:, BUS_I], arange(nb))), (nb, nb))  # Sparse index load

    model = problem_formulation(test_case, Distribution_factor, Cg, Cd)
    (xx, obj, success) = miqp(c=model["c"], Q=model["Q"], Aeq=model["Aeq"], A=model["Aineq"], b=model["bineq"],
                              beq=model["beq"], xmin=model["lb"],
                              xmax=model["ub"], vtypes=model["vtypes"])
    sol = solution_decomposition(xx, obj, success)

    T = 24
    nx = 4 * T * ng
    # check the branch power flow
    branch_f2t = zeros((nl, T))
    branch_t2f = zeros((nl, T))
    for i in range(T):
        PD_bus = test_case["bus"][:, 1] * test_case["Load_profile"][i]
        branch_f2t[:, i] = Distribution_factor * (Cg * sol["Pg"][:, i] - Cd * PD_bus)
        branch_t2f[:, i] = -Distribution_factor * (Cg * sol["Pg"][:, i] - Cd * PD_bus)

    plt.plot(sol["Pg"])
    plt.show()
