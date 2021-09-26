"""
Two-stage unit commitment considering the uncertainty of loads
"""

from numpy import zeros, shape, ones, diag, concatenate, r_, arange, cumsum
import matplotlib.pyplot as plt
from solvers.mixed_integer_quadratic_programming import mixed_integer_quadratic_programming as miqp
import scipy.linalg as linalg
from scipy.sparse import csr_matrix as sparse
from solvers.mixed_integer_solvers_cplex import \
    mixed_integer_linear_programming as lp  # The solver for the second stage decison making


class TwoStageStochasticUnitCommitment():
    def __init__(self):
        self.name = "Stochastic unit commitment"

    def problem_formulation(self, case, beta=0.03):
        """
        :param case: Cases for unit commitment
        :return:
        """

        # Import the data format
        from unit_commitment.data_format.data_format_jointed_energy_reserve import ALPHA, BETA, IG, PG, RUG, RDG

        from unit_commitment.test_cases.case118 import F_BUS, T_BUS, BR_X, RATE_A
        from unit_commitment.test_cases.case118 import GEN_BUS, COST_C, COST_B, COST_A, PG_MAX, PG_MIN, I0, MIN_DOWN, \
            MIN_UP, RU, RD, COLD_START
        from unit_commitment.test_cases.case118 import BUS_ID, PD
        baseMVA, bus, gen, branch, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"], case[
            "Load_profile"]

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
        # [vt,wt,ut,Pt,Pug,Pdg]:start-up,shut-down,status,generation level
        # 1.1) boundary information, variable types
        T = case["Load_profile"].shape[0]
        nx = (RDG + 1) * T * ng

        lb = zeros((nx, 1))
        ub = zeros((nx, 1))
        vtypes = ["c"] * nx
        for i in range(T):
            for j in range(ng):
                lb[ALPHA * ng * T + i * ng + j] = 0
                lb[BETA * ng * T + i * ng + j] = 0
                lb[IG * ng * T + i * ng + j] = 0
                lb[PG * ng * T + i * ng + j] = 0
                lb[RUG * ng * T + i * ng + j] = 0
                lb[RDG * ng * T + i * ng + j] = 0

                ub[ALPHA * ng * T + i * ng + j] = 1
                ub[BETA * ng * T + i * ng + j] = 1
                ub[IG * ng * T + i * ng + j] = 1
                ub[PG * ng * T + i * ng + j] = gen[j, PG_MAX]
                ub[RUG * ng * T + i * ng + j] = gen[j, PG_MAX]
                ub[RDG * ng * T + i * ng + j] = gen[j, PG_MAX]

                vtypes[IG * ng * T + i * ng + j] = "B"

        # 1.2) objective information
        c = zeros((nx, 1))
        q = [0] * nx
        for i in range(T):
            for j in range(ng):
                c[ALPHA * ng * T + i * ng + j] = gen[j, COLD_START]
                c[IG * ng * T + i * ng + j] = gen[j, COST_C]
                c[PG * ng * T + i * ng + j] = gen[j, COST_B]

                q[PG * ng * T + i * ng + j] = gen[j, COST_A]

        # 2) Constraint set
        # 2.1) Power balance equation
        Aeq = zeros((T, nx))
        beq = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aeq[i, PG * ng * T + i * ng + j] = 1

            beq[i] = case["Load_profile"][i]

        # 2.2) Status transformation of each unit
        Aeq_temp = zeros((T * ng, nx))
        beq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aeq_temp[i * ng + j, ALPHA * ng * T + i * ng + j] = 1
                Aeq_temp[i * ng + j, BETA * ng * T + i * ng + j] = -1
                Aeq_temp[i * ng + j, IG * ng * T + i * ng + j] = -1
                if i != 0:
                    Aeq_temp[i * ng + j, IG * ng * T + (i - 1) * ng + j] = 1
                else:
                    beq_temp[i * ng + j] = -u0[i]

        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate([beq, beq_temp], axis=0)

        # 2.3) Power range limitation
        Aineq = zeros((T * ng, nx))
        bineq = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq[i * ng + j, IG * ng * T + i * ng + j] = gen[j, PG_MIN]
                Aineq[i * ng + j, PG * ng * T + i * ng + j] = -1
                Aineq[i * ng + j, RDG * ng * T + i * ng + j] = 1

        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, PG_MAX]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = 1
                Aineq_temp[i * ng + j, RUG * ng * T + i * ng + j] = 1

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.4) Start up and shut down time limitation
        UP_LIMIT = T - gen[:, MIN_UP]
        DOWN_LIMIT = T - gen[:, MIN_DOWN]
        UP_LIMIT_SUM = cumsum(UP_LIMIT)
        DOWN_LIMIT_SUM = cumsum(DOWN_LIMIT)

        # 2.4.1) Up limit
        Aineq_temp = zeros((int(sum(UP_LIMIT)), nx))
        bineq_temp = zeros((int(sum(UP_LIMIT)), 1))
        for i in range(ng):
            if i != 0:
                for j in range(int(UP_LIMIT[i])):
                    for k in range(int(gen[i, MIN_UP])):
                        Aineq_temp[int(UP_LIMIT_SUM[i - 1]) + j, ALPHA * ng * T + (k + j) * ng + i] = 1
                    Aineq_temp[int(UP_LIMIT_SUM[i - 1]) + j, IG * ng * T + (int(gen[i, MIN_UP]) - 1 + j) * ng + i] = -1
            else:
                for j in range(int(UP_LIMIT[i])):
                    for k in range(int(gen[i, MIN_UP])):
                        Aineq_temp[j, ALPHA * ng * T + (k + j) * ng + i] = 1
                    Aineq_temp[j, IG * ng * T + (int(gen[i, MIN_UP]) - 1 + j) * ng + i] = -1

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.4.2) Down limit
        Aineq_temp = zeros((int(sum(UP_LIMIT)), nx))
        bineq_temp = ones((int(sum(UP_LIMIT)), 1))
        for i in range(ng):
            if i != 0:
                for j in range(int(DOWN_LIMIT[i])):
                    for k in range(int(gen[i, MIN_DOWN])):
                        Aineq_temp[int(DOWN_LIMIT_SUM[i - 1]) + j, BETA * ng * T + (k + j) * ng + i] = 1
                    Aineq_temp[int(DOWN_LIMIT_SUM[i - 1]) + j, IG * ng * T + (int(gen[i, MIN_UP]) - 1 + j) * ng + i] = 1
            else:
                for j in range(int(DOWN_LIMIT[i])):
                    for k in range(int(gen[i, MIN_DOWN])):
                        Aineq_temp[j, BETA * ng * T + (k + j) * ng + i] = 1
                    Aineq_temp[j, IG * ng * T + (int(gen[i, MIN_DOWN]) - 1 + j) * ng + i] = 1

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.5) Ramp constraints:
        # 2.5.1) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(T - 1):
            for j in range(ng):
                Aineq_temp[i * ng + j, PG * ng * T + (i + 1) * ng + j] = 1
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = -1
                Aineq_temp[i * ng + j, ALPHA * ng * T + i * ng + j] = gen[j, RU] - gen[j, PG_MIN]
                bineq_temp[i * ng + j] = gen[j, RU]

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.5.2) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(T - 1):
            for j in range(ng):
                Aineq_temp[i * ng + j, PG * ng * T + (i + 1) * ng + j] = -1
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = 1
                Aineq_temp[i * ng + j, BETA * ng * T + i * ng + j] = gen[j, RD] - gen[j, PG_MIN]
                bineq_temp[i * ng + j] = gen[j, RD]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6) Line flow limitation
        # Add the line flow limitation time by time
        b = 1 / branch[:, BR_X]  ## series susceptance

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
        Cd = sparse((ones(nb), (bus[:, BUS_ID], arange(nb))), (nb, nb))  # Sparse index load

        Aineq_temp = zeros((nl * T, nx))
        bineq_temp = zeros((nl * T, 1))
        for i in range(T):
            Aineq_temp[i * nl:(i + 1) * nl, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = (
                    Distribution_factor * Cg).todense()
            PD_bus = bus[:, PD] * case["Load_profile"][i]
            bineq_temp[i * nl:(i + 1) * nl, :] = (branch[:, RATE_A] + Distribution_factor * PD_bus).reshape(nl, 1)
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        Aineq_temp = zeros((nl * T, nx))
        bineq_temp = zeros((nl * T, 1))
        for i in range(T):
            Aineq_temp[i * nl:(i + 1) * nl, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = -(
                    Distribution_factor * Cg).todense()
            PD_bus = bus[:, PD] * case["Load_profile"][i]
            bineq_temp[i * nl:(i + 1) * nl, :] = (branch[:, RATE_A] - Distribution_factor * PD_bus).reshape(nl, 1)
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6) Reserve constraints
        # 2.6.1) Up reserves
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RUG * ng * T + i * ng + j] = -1
            bineq_temp[i] = -case["Load_profile"][i] * beta
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6.2) Down reserves
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RDG * ng * T + i * ng + j] = -1
            bineq_temp[i] = -case["Load_profile"][i] * beta
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        model = {}
        model["c"] = c
        model["Q"] = diag(q)
        model["Aeq"] = Aeq
        model["beq"] = beq
        model["lb"] = lb
        model["ub"] = ub
        model["Aineq"] = Aineq
        model["bineq"] = bineq
        model["vtypes"] = vtypes
        model["Distribution_factor"] = Distribution_factor
        model["Cg"] = Cg
        model["Cd"] = Cd
        model["T"] = T
        model["ng"] = ng
        model["nl"] = nl

        return model

    def solution_decomposition(self, xx, obj, success):
        """
        Decomposition of objective functions
        :param xx: Solution
        :param obj: Objective value
        :param success: Success or not
        :return:
        """
        from unit_commitment.data_format.data_format_jointed_energy_reserve import ALPHA, BETA, IG, PG, RDG, RUG
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
            Rug = zeros((ng, T))
            Rdg = zeros((ng, T))

            for i in range(T):
                for j in range(ng):
                    v[j, i] = xx[ALPHA * ng * T + i * ng + j]
                    w[j, i] = xx[BETA * ng * T + i * ng + j]
                    Ig[j, i] = xx[IG * ng * T + i * ng + j]
                    Pg[j, i] = xx[PG * ng * T + i * ng + j]
                    Rug[j, i] = xx[RUG * ng * T + i * ng + j]
                    Rdg[j, i] = xx[RDG * ng * T + i * ng + j]

            result["vt"] = v
            result["wt"] = w
            result["Ig"] = Ig
            result["Pg"] = Pg
            result["Rug"] = Rug
            result["Rdg"] = Rdg

        else:
            result["vt"] = 0
            result["wt"] = 0
            result["Ig"] = 0
            result["Pg"] = 0
            result["Rug"] = 0
            result["Rdg"] = 0

        return result


if __name__ == "__main__":
    from unit_commitment.test_cases import case118

    test_case = case118.case118()
    unit_commitment = TwoStageStochasticUnitCommitment()
    model = unit_commitment.problem_formulation(test_case)

    (xx, obj, success) = miqp(c=model["c"], Q=model["Q"], Aeq=model["Aeq"], A=model["Aineq"], b=model["bineq"],
                              beq=model["beq"], xmin=model["lb"],
                              xmax=model["ub"], vtypes=model["vtypes"])

    sol = unit_commitment.solution_decomposition(xx, obj, success)
    ng = model["ng"]
    nl = model["nl"]
    T = model["T"]
    Distribution_factor = model["Distribution_factor"]
    Cg = model["Cg"]
    Cd = model["Cd"]

    nx = 6 * T * ng
    # check the branch power flow
    branch_f2t = zeros((nl, T))
    branch_t2f = zeros((nl, T))
    for i in range(T):
        PD_bus = test_case["bus"][:, 1] * test_case["Load_profile"][i]
        branch_f2t[:, i] = Distribution_factor * (Cg * sol["Pg"][:, i] - Cd * PD_bus)
        branch_t2f[:, i] = -Distribution_factor * (Cg * sol["Pg"][:, i] - Cd * PD_bus)

    plt.plot(sol["Pg"])
    plt.show()
