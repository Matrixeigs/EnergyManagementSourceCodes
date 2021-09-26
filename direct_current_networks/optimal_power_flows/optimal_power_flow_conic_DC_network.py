"""

Optimal power flow based on branch power flow modelling for direct current power flows

Additional case33 is added to the test cases

Note: The proposed method has been verified
# Reference:
[1] Gan, Lingwen, and Steven H. Low. "Optimal power flow in direct current networks." IEEE Transactions on Power Systems 29.6 (2014): 2892-2904.
@author: Tianyang Zhao
@email: zhaoty@ntu.edu.sg
1) Note that, the hypothesis in Theorem 1 and 2 have been verified.
2) The upper boundary limitation of generators plays an important role in the exactness of relaxation.
"""

from distribution_system_optimization.test_cases import case33
from pypower import runopf
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate, power
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, diags
from pypower import case6ww, case9, case30, case118, case300


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

    nb = shape(mpc['bus'])[0]  ## number of buses
    nl = shape(mpc['branch'])[0]  ## number of branches
    ng = shape(mpc['gen'])[0]  ## number of dispatchable injections

    f = branch[:, F_BUS]  ## list of "from" buses
    t = branch[:, T_BUS]  ## list of "to" buses
    i = range(nl)  ## double set of row indices
    # Connection matrix
    Cf = sparse((ones(nl), (i, f)), (nl, nb))
    Ct = sparse((ones(nl), (i, t)), (nl, nb))
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
    Branch_R = branch[:, BR_R]
    # The branch resistance modification
    for i in range(nl):
        if Branch_R[i] <= 0:
            Branch_R[i] = max(Branch_R)

    Cf = Cf.T
    Ct = Ct.T
    # Obtain the boundary information

    Slmax = branch[:, RATE_A] / baseMVA
    Pij_l = -Slmax
    Iij_l = zeros(nl)
    Vm_l = power(bus[:, VMIN], 2)
    Pg_l = gen[:, PMIN] / baseMVA

    Pij_u = Slmax
    Iij_u = Slmax
    # Vm_u = [max(turn_to_power(bus[:, VMAX], 2))] * nb
    Vm_u = power(bus[:, VMAX], 2)
    Pg_u = 100 * gen[:, PMAX] / baseMVA
    # Pg_l = -Pg_u
    lx = concatenate([Pij_l, Iij_l, Vm_l, Pg_l])
    ux = concatenate([Pij_u, Iij_u, Vm_u, Pg_u])
    model = Model("OPF")
    # Define the decision variables
    x = {}
    nx = 2 * nl + nb + ng
    for i in range(nx):
        x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)

    # Add system level constraints
    Aeq_p = hstack([Ct - Cf, -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg])
    beq_p = bus[:, PD] / baseMVA
    # Add constraints for each sub system
    Aeq_KVL = hstack([-2 * diags(Branch_R), diags(power(Branch_R, 2)), Cf.T - Ct.T, zeros((nl, ng))])
    beq_KVL = zeros(nl)

    Aeq = vstack([Aeq_p, Aeq_KVL])
    Aeq = Aeq.todense()
    beq = concatenate([beq_p, beq_KVL])
    neq = len(beq)

    for i in range(neq):
        expr = 0
        for j in range(nx):
            expr += x[j] * Aeq[i, j]
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

    for i in range(nl):
        model.addConstr(x[i] * x[i] <= x[i + nl] * x[f[i] + 2 * nl], name='"rc{0}"'.format(i))

    obj = 0
    for i in range(ng):
        obj += gencost[i, 4] * x[i + 2 * nl + nb] * x[i + 2 * nl + nb] * baseMVA * baseMVA + gencost[i, 5] * x[
            i + 2 * nl + nb] * baseMVA + gencost[i, 6]
    # for i in range(nl):
    #     obj += 0.1 * (x[i + nl] * Branch_R[i])

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
    Iij = xx[nl:2 * nl]
    Vi = xx[2 * nl:2 * nl + nb]
    Pg = xx[2 * nl + nb:2 * nl + nb + ng]

    primal_residual = []

    for i in range(nl):
        primal_residual.append(Pij[i] * Pij[i] - Iij[i] * Vi[int(f[i])])

    return xx, obj, primal_residual


if __name__ == "__main__":
    from pypower import runopf

    # mpc = case33.case33()  # Default test case
    mpc = case30.case30()

    (xx, obj, residual) = run(mpc)

    result = runopf.runopf(case33.case33())

    gap = 100 * (result["f"] - obj) / obj

    print(gap)
    print(max(residual))
