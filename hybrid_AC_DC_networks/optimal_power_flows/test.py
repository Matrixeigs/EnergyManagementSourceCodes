"""
ADMM based distributed optimal power flow
The power flow modelling is based on the branch power flow

References:
    [1]Peng, Qiuyu, and Steven H. Low. "Distributed optimal power flow algorithm for radial networks, I: Balanced single phase case." IEEE Transactions on Smart Grid (2016).
    [2]
"""

from Two_stage_stochastic_optimization.power_flow_modelling import case33
from pypower import runopf
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate, where
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, diags


def run(mpc):
    """
    Gurobi based optimal power flow modelling and solution
    :param mpc: The input case of optimal power flow
    :return: obtained solution
    """
    # Data format
    from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
    from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
    from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
    from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
    from pypower.ext2int import ext2int

    mpc = ext2int(mpc)
    baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

    nb = shape(mpc['bus'])[0]  # number of buses
    nl = shape(mpc['branch'])[0]  # number of branches
    ng = shape(mpc['gen'])[0]  # number of dispatchable injections
    f = branch[:, F_BUS]  ## list of "from" buses
    t = branch[:, T_BUS]  ## list of "to" buses
    i = range(nl)  ## double set of row indices
    # Connection matrix
    Cf = sparse((ones(nl), (i, f)), (nl, nb))
    Ct = sparse((ones(nl), (i, t)), (nl, nb))
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
    Branch_R = branch[:, BR_R]
    Branch_X = branch[:, BR_X]
    Cf = Cf.T
    Ct = Ct.T
    # Obtain the boundary information

    Slmax = branch[:, RATE_A] / baseMVA
    Pij_l = -Slmax
    Qij_l = -Slmax
    Iij_l = zeros(nl)
    Vm_l = turn_to_power(bus[:, VMIN], 2)
    Pg_l = gen[:, PMIN] / baseMVA
    Qg_l = gen[:, QMIN] / baseMVA

    Pij_u = Slmax
    Qij_u = Slmax
    Iij_u = Slmax
    Vm_u = turn_to_power(bus[:, VMAX], 2)
    Pg_u = 2 * gen[:, PMAX] / baseMVA
    Qg_u = 2 * gen[:, QMAX] / baseMVA
    lx = concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l])
    ux = concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u])
    model = Model("OPF")
    # Define the decision variables
    x = {}
    nx = 3 * nl + nb + 2 * ng
    for i in range(nx):
        x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)

    # Add system level constraints
    Aeq_p = hstack([Ct - Cf, zeros((nb, nl)), -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg, zeros((nb, ng))])
    beq_p = bus[:, PD] / baseMVA
    # Add constraints for each sub system
    Aeq_q = hstack([zeros((nb, nl)), Ct - Cf, -diag(Ct * Branch_X) * Ct, zeros((nb, nb)), zeros((nb, ng)), Cg])
    beq_q = bus[:, QD] / baseMVA
    Aeq_KVL = hstack([-2 * diags(Branch_R), -2 * diags(Branch_X),
                      diags(turn_to_power(Branch_R, 2)) + diags(turn_to_power(Branch_X, 2)), Cf.T - Ct.T,
                      zeros((nl, 2 * ng))])
    beq_KVL = zeros(nl)

    Aeq = vstack([Aeq_p, Aeq_q, Aeq_KVL])
    Aeq = Aeq.todense()
    beq = concatenate([beq_p, beq_q, beq_KVL])
    neq = len(beq)

    for i in range(neq):
        expr = 0
        for j in range(nx):
            expr += x[j] * Aeq[i, j]
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

    for i in range(nl):
        model.addConstr(x[i] * x[i] + x[i + nl] * x[i + nl] <= x[i + 2 * nl] * x[f[i] + 3 * nl],
                        name='"rc{0}"'.format(i))

    obj = 0
    for i in range(ng):
        obj += gencost[i, 4] * x[i + 3 * nl + nb] * x[i + 3 * nl + nb] * baseMVA * baseMVA + gencost[i, 5] * x[
            i + 3 * nl + nb] * baseMVA + gencost[i, 6]

    model.setObjective(obj)
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.Params.DisplayInterval = 1
    model.optimize()

    xx = []
    for v in model.getVars():
        xx.append(v.x)

    obj = obj.getValue()

    Pij = xx[0:nl]
    Qij = xx[nl + 0:2 * nl]
    Iij = xx[2 * nl:3 * nl]
    Vi = xx[3 * nl:3 * nl + nb]
    Pg = xx[3 * nl + nb:3 * nl + nb + ng]
    Qg = xx[3 * nl + nb + ng: 3 * nl + nb + 2 * ng]



    # for i in range(nl):  # branch indexing exchange
    #     if branch[i, F_BUS] > branch[i, T_BUS]:
    #         temp = branch[i, F_BUS]
    #         branch[i, F_BUS] = branch[i, T_BUS]
    #         branch[i, T_BUS] = temp

    f = branch[:, F_BUS]  ## list of "from" buses
    t = branch[:, T_BUS]  ## list of "to" buses
    # i = range(nl)  ## double set of row indices
    area = ancestor_children_generation(f, t, range(nb))
    Pi = Cg * Pg -bus[:, PD]/ baseMVA
    Qi = Cg * Qg -bus[:, QD]/ baseMVA

    for i in range(nb):
        # If the bus is the root bus, only the children information is required.
        if len(area[i]["Ai"]) == 0:
            print(i)
            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Pij[area[i]["Cbranch"][0][j]]

            print(expr - Pi[i])

            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Qij[area[i]["Cbranch"][0][j]]

            print(expr - Qi[i])

        elif len(area[i]["Cbranch"]) == 0:  # This bus is the lead node
            print(i)
            print(Pij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_R[area[i]["Abranch"][0][0]] +
                    Pi[i])

            print(Qij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_X[area[i]["Abranch"][0][0]] +
                    Qi[i])

            print(Vi[int(area[i]["Ai"][0])] - Vi[i] - 2 * Branch_R[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] - 2 * Branch_X[area[i]["Abranch"][0][0]] * Qij[area[i]["Abranch"][0][0]] +
                                Iij[area[i]["Abranch"][0][0]] * (Branch_R[area[i]["Abranch"][0][0]] ** 2 + Branch_X[area[i]["Abranch"][0][0]] ** 2))

            print(
                Pij[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] + Qij[area[i]["Abranch"][0][0]] * Qij[
                    area[i]["Abranch"][0][0]] <= Vi[int(area[i]["Ai"][0])] *
                Iij[area[i]["Abranch"][0][0]])

        else:
            print(i)
            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Pij[area[i]["Cbranch"][0][j]]
            print(
                Pij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_R[area[i]["Abranch"][0][0]] +
                    Pi[i] - expr)

            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Qij[area[i]["Cbranch"][0][j]]

            print(Qij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_X[area[i]["Abranch"][0][0]] +
                    Qi[i] - expr)

            print(Vi[int(area[i]["Ai"][0])] - Vi[i] - 2 * Branch_R[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] - 2 * Branch_X[area[i]["Abranch"][0][0]] * Qij[area[i]["Abranch"][0][0]] +
                                Iij[area[i]["Abranch"][0][0]] * (Branch_R[area[i]["Abranch"][0][0]] ** 2 + Branch_X[area[i]["Abranch"][0][0]] ** 2)
                  )
            print(
                Pij[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] + Qij[area[i]["Abranch"][0][0]] * Qij[area[i]["Abranch"][0][0]] <= Vi[int(area[i]["Ai"][0])] *
                Iij[area[i]["Abranch"][0][0]])
    obj = 0
    for i in range(ng):
        print( Pg[i] - Pi[int(gen[i, GEN_BUS])] - bus[int(gen[i, GEN_BUS]), PD] / baseMVA)
        print( Qg[i] - Qi[int(gen[i, GEN_BUS])] - bus[int(gen[i, GEN_BUS]), QD] / baseMVA)
        print(int(gen[i, GEN_BUS]))
        obj += gencost[i, 4] * Pg[i] * Pg[i] * baseMVA * baseMVA + gencost[i, 5] * Pg[i] * baseMVA + gencost[i, 6]


    # Connection matrix
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
    Branch_R = branch[:, BR_R]
    Branch_X = branch[:, BR_X]

    # Obtain the boundary information

    Slmax = branch[:, RATE_A] / baseMVA

    Pij_l = -Slmax
    Qij_l = -Slmax
    Iij_l = zeros(nl)
    Vm_l = turn_to_power(bus[:, VMIN], 2)
    Pg_l = gen[:, PMIN] / baseMVA
    Qg_l = gen[:, QMIN] / baseMVA
    Pi_l = -bus[:, PD] / baseMVA + Cg * Pg_l / baseMVA
    Qi_l = -bus[:, QD] / baseMVA + Cg * Qg_l / baseMVA

    Pij_u = Slmax
    Qij_u = Slmax
    Iij_u = Slmax
    Vm_u = turn_to_power(bus[:, VMAX], 2)
    Pg_u = 2 * gen[:, PMAX] / baseMVA
    Qg_u = 2 * gen[:, QMAX] / baseMVA
    Pi_u = -bus[:, PD] / baseMVA + Cg * Pg_u # Boundary error
    Qi_u = -bus[:, QD] / baseMVA + Cg * Qg_u # Boundary error

    model = Model("OPF")
    # Define the decision variables, compact set
    Pij = {}
    Qij = {}
    Iij = {}
    Vi = {}
    Pg = {}
    Qg = {}
    Pi = {}
    Qi = {}

    for i in range(nl):
        Pij[i] = model.addVar(lb=Pij_l[i], ub=Pij_u[i], vtype=GRB.CONTINUOUS, name="Pij{0}".format(i))
        Qij[i] = model.addVar(lb=Qij_l[i], ub=Qij_u[i], vtype=GRB.CONTINUOUS, name="Qij{0}".format(i))
        Iij[i] = model.addVar(lb=Iij_l[i], ub=Iij_u[i], vtype=GRB.CONTINUOUS, name="Iij{0}".format(i))

    for i in range(nb):
        Vi[i] = model.addVar(lb=Vm_l[i], ub=Vm_u[i], vtype=GRB.CONTINUOUS, name="V{0}".format(i))

    for i in range(ng):
        Pg[i] = model.addVar(lb=Pg_l[i], ub=Pg_u[i], vtype=GRB.CONTINUOUS, name="Pg{0}".format(i))
        Qg[i] = model.addVar(lb=Qg_l[i], ub=Qg_u[i], vtype=GRB.CONTINUOUS, name="Qg{0}".format(i))
    for i in range(nb):
        Pi[i] = model.addVar(lb=Pi_l[i], ub=Pi_u[i], vtype=GRB.CONTINUOUS, name="Pi{0}".format(i))
        Qi[i] = model.addVar(lb=Qi_l[i], ub=Qi_u[i], vtype=GRB.CONTINUOUS, name="Qi{0}".format(i))
    # For each area, before decomposition
    # Add system level constraints
    for i in range(nb):
        # If the bus is the root bus, only the children information is required.
        if len(area[i]["Ai"]) == 0:
            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Pij[area[i]["Cbranch"][0][j]]

            model.addConstr(lhs = expr - Pi[i], sense=GRB.EQUAL, rhs=0)

            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Qij[area[i]["Cbranch"][0][j]]

            model.addConstr(lhs=expr - Qi[i], sense=GRB.EQUAL, rhs=0)

        elif len(area[i]["Cbranch"]) == 0:  # This bus is the lead node
            model.addConstr(
                lhs=Pij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_R[area[i]["Abranch"][0][0]] +
                    Pi[i], sense=GRB.EQUAL, rhs=0)
            model.addConstr(
                lhs=Qij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_X[area[i]["Abranch"][0][0]] +
                    Qi[i], sense=GRB.EQUAL, rhs=0)

            model.addConstr(lhs=Vi[int(area[i]["Ai"][0])] - Vi[i] - 2 * Branch_R[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] - 2 * Branch_X[area[i]["Abranch"][0][0]] * Qij[area[i]["Abranch"][0][0]] +
                                Iij[area[i]["Abranch"][0][0]] * (Branch_R[area[i]["Abranch"][0][0]] ** 2 + Branch_X[area[i]["Abranch"][0][0]] ** 2),
                            sense=GRB.EQUAL, rhs=0)

            model.addConstr(
                Pij[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] + Qij[area[i]["Abranch"][0][0]] * Qij[
                    area[i]["Abranch"][0][0]] <= Vi[area[i]["Ai"][0]] *
                Iij[area[i]["Abranch"][0][0]], name="rc{0}".format(i))
        else:
            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Pij[area[i]["Cbranch"][0][j]]
            model.addConstr(
                lhs=Pij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_R[area[i]["Abranch"][0][0]] +
                    Pi[i] - expr , sense=GRB.EQUAL, rhs=0)
            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Qij[area[i]["Cbranch"][0][j]]

            model.addConstr(
                lhs=Qij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_X[area[i]["Abranch"][0][0]] +
                    Qi[i] - expr, sense=GRB.EQUAL, rhs=0)

            model.addConstr(lhs=Vi[int(area[i]["Ai"][0])] - Vi[i] - 2 * Branch_R[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] - 2 * Branch_X[area[i]["Abranch"][0][0]] * Qij[area[i]["Abranch"][0][0]] +
                                Iij[area[i]["Abranch"][0][0]] * (Branch_R[area[i]["Abranch"][0][0]] ** 2 + Branch_X[area[i]["Abranch"][0][0]] ** 2),
                            sense=GRB.EQUAL, rhs=0)

            model.addConstr(
                Pij[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] + Qij[area[i]["Abranch"][0][0]] * Qij[area[i]["Abranch"][0][0]] <= Vi[area[i]["Ai"][0]] *
                Iij[area[i]["Abranch"][0][0]], name="rc{0}".format(i))
    obj = 0
    for i in range(ng):
        model.addConstr(lhs = Pg[i] - Pi[int(gen[i, GEN_BUS])], sense=GRB.EQUAL, rhs=bus[int(gen[i, GEN_BUS]), PD] / baseMVA)
        model.addConstr(lhs = Qg[i] - Qi[int(gen[i, GEN_BUS])], sense=GRB.EQUAL, rhs=bus[int(gen[i, GEN_BUS]), QD] / baseMVA)
        obj += gencost[i, 4] * Pg[i] * Pg[i] * baseMVA * baseMVA + gencost[i, 5] * Pg[i] * baseMVA + gencost[i, 6]


    model.setObjective(obj)
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.Params.DisplayInterval = 1
    model.optimize()

    Pij = []
    Qij = []
    Iij = []
    Vi = []
    Pg = []
    Qg = []
    Pi = []
    Qi = []

    for i in range(nl):
        Pij.append(model.getVarByName("Pij{0}".format(i)).X)
        Qij.append(model.getVarByName("Qij{0}".format(i)).X)
        Iij.append(model.getVarByName("Iij{0}".format(i)).X)

    for i in range(nb):
        Vi.append(model.getVarByName("V{0}".format(i)).X)
        Pi.append(model.getVarByName("Pi{0}".format(i)).X)
        Qi.append(model.getVarByName("Qi{0}".format(i)).X)

    for i in range(ng):
        Pg.append(model.getVarByName("Pg{0}".format(i)).X)
        Qg.append(model.getVarByName("Qg{0}".format(i)).X)

    obj = obj.getValue()


    primal_residual = []

    for i in range(nl):
        primal_residual.append(Pij[i] * Pij[i] + Qij[i] * Qij[i] - Iij[i] * Vi[int(f[i])])

    return obj, primal_residual


def turn_to_power(list, power=1):
    return [number ** power for number in list]


def ancestor_children_generation(branch_f, branch_t, index):
    """
    Ancestor and children information for each node
    :param branch_f:
    :param branch_t:
    :param index: Bus index
    :return: Area, ancestor bus, children buses, line among buses
    """
    Area = []
    for i in index:
        temp = {}
        temp["Index"] = i
        if i in branch_t:
            temp["Ai"] = branch_f[where(branch_t == i)] # For each bus, there exits only one ancestor bus, as one connected tree
            temp["Abranch"] = where(branch_t == i)
        else:
            temp["Ai"] = []
            temp["Abranch"] = []

        if i in branch_f:
            temp["Cbranch"] = where(branch_f == i)
        else:
            temp["Cbranch"] = []

        Area.append(temp)

    return Area


if __name__ == "__main__":
    from pypower import runopf

    mpc = case33.case33()  # Default test case
    (obj, residual) = run(mpc)

    result = runopf.runopf(case33.case33())

    gap = 100 * (result["f"] - obj) / obj

    print(gap)
    print(residual)
