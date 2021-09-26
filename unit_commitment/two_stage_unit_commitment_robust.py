"""
Two Stage Robust Optimization for Unit Commitment Problem
The test case is the IEEE-6ww system.

@date: 13 June 2018
@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
"""

from numpy import zeros, shape, ones, diag, concatenate, r_, arange, array
import matplotlib.pyplot as plt
from solvers.mixed_integer_quadratic_programming import mixed_integer_quadratic_programming as miqp
import scipy.linalg as linalg
from scipy.sparse import csr_matrix as sparse

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN, RAMP_AGC
from pypower.idx_cost import STARTUP

from solvers.mixed_integer_quadratic_solver_cplex import mixed_integer_quadratic_programming as miqp

from unit_commitment.data_format.data_format_jointed_energy_reserve import ALPHA, BETA, IG, PG, RDG, RUG


class TwoStageUnitCommitmentRobust():
    """"""

    def __init__(self):
        self.name = "Two stage robust optimization"

    def problem_formulation(self, case, delta=0.03):
        """
        Input check for the unit commitment problem
        :param cases:
        :return:
        """
        baseMVA, bus, gen, branch, gencost, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"], case[
            "gencost"], case["Load_profile"]

        MIN_UP = -2
        MIN_DOWN = -3

        # Modify the bus, gen and branch matrix
        bus[:, BUS_I] = bus[:, BUS_I] - 1
        gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1
        branch[:, F_BUS] = branch[:, F_BUS] - 1
        branch[:, T_BUS] = branch[:, T_BUS] - 1

        ng = shape(case['gen'])[0]  # number of schedule injections
        nl = shape(case['branch'])[0]  ## number of branches
        nb = shape(case['bus'])[0]  ## number of branches
        self.ng = ng
        self.nb = nb
        self.nl = nl

        u0 = [0] * ng  # The initial generation status
        for i in range(ng):
            u0[i] = int(gencost[i, -1] > 0)
        # Formulate a mixed integer quadratic programming problem
        # 1) Announce the variables
        # [vt,wt,ut,Pt]:start-up,shut-down,status,generation level, up-reserve, down-reserve
        # 1.1) boundary information
        T = case["Load_profile"].shape[0]
        nx = (RDG + 1) * T * ng
        lb = zeros((nx, 1))
        ub = zeros((nx, 1))
        vtypes = ["c"] * nx
        self.T = T

        for i in range(T):
            for j in range(ng):
                # lower boundary
                lb[ALPHA * ng * T + i * ng + j] = 0
                lb[BETA * ng * T + i * ng + j] = 0
                lb[IG * ng * T + i * ng + j] = 0
                lb[PG * ng * T + i * ng + j] = 0
                lb[RUG * ng * T + i * ng + j] = 0
                lb[RDG * ng * T + i * ng + j] = 0
                # upper boundary
                ub[ALPHA * ng * T + i * ng + j] = 1
                ub[BETA * ng * T + i * ng + j] = 1
                ub[IG * ng * T + i * ng + j] = 1
                ub[PG * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RUG * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RDG * ng * T + i * ng + j] = gen[j, PMAX]
                # variable types
                vtypes[IG * ng * T + i * ng + j] = "D"

        c = zeros((nx, 1))
        q = zeros((nx, 1))
        for i in range(T):
            for j in range(ng):
                # cost
                c[ALPHA * ng * T + i * ng + j] = gencost[j, STARTUP]
                c[BETA * ng * T + i * ng + j] = 0
                c[IG * ng * T + i * ng + j] = gencost[j, 6]
                c[PG * ng * T + i * ng + j] = gencost[j, 5]
                c[RUG * ng * T + i * ng + j] = 0
                c[RDG * ng * T + i * ng + j] = 0

                q[PG * ng * T + i * ng + j] = gencost[j, 4]

        # 2) Constraint set
        # 2.1) Power balance equation
        Aeq = zeros((T, nx))
        beq = zeros((T, 1))
        for i in range(T):
            # For the hydro units
            for j in range(ng):
                Aeq[i, PG * ng * T + i * ng + j] = 1

            beq[i] = profile[i] * sum(bus[:, PD])

        # # 2.2) Status transformation of each unit
        Aeq_temp = zeros((T * ng, nx))
        beq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aeq_temp[i * ng + j, ALPHA * ng * T + i * ng + j] = -1
                Aeq_temp[i * ng + j, BETA * ng * T + i * ng + j] = 1
                Aeq_temp[i * ng + j, IG * ng * T + i * ng + j] = 1
                if i != 0:
                    Aeq_temp[i * ng + j, IG * ng * T + (i - 1) * ng + j] = -1
                else:
                    beq_temp[i * T + j] = 0

        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate((beq, beq_temp), axis=0)

        # 2.3) Power range limitation
        Aineq = zeros((T * ng, nx))
        bineq = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq[i * ng + j, ALPHA * ng * T + i * ng + j] = 1
                Aineq[i * ng + j, BETA * ng * T + i * ng + j] = 1
                bineq[i * ng + j] = 1

        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = gen[j, PMIN]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = -1
                Aineq_temp[i * ng + j, RDG * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, PMAX]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = 1
                Aineq_temp[i * ng + j, RUG * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.4) Start up and shut down time limitation
        UP_LIMIT = [0] * ng
        DOWN_LIMIT = [0] * ng
        for i in range(ng):
            UP_LIMIT[i] = T - int(gencost[i, MIN_UP])
            DOWN_LIMIT[i] = T - int(gencost[i, MIN_DOWN])
        # 2.4.1) Up limit
        Aineq_temp = zeros((sum(UP_LIMIT), nx))
        bineq_temp = zeros((sum(UP_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gencost[i, MIN_UP]), T):
                for k in range(j - int(gencost[i, MIN_UP]), j):
                    Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), ALPHA * ng * T + k * ng + i] = 1
                Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), IG * ng * T + j * ng + i] = -1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.4.2) Down limit
        Aineq_temp = zeros((sum(DOWN_LIMIT), nx))
        bineq_temp = ones((sum(DOWN_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gencost[i, MIN_DOWN]), T):
                for k in range(j - int(gencost[i, MIN_DOWN]), j):
                    Aineq_temp[
                        sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_DOWN]), BETA * ng * T + k * ng + i] = 1
                Aineq_temp[sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_DOWN]), IG * ng * T + j * ng + i] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.5) Ramp constraints:
        # 2.5.1) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, ALPHA * ng * T + (j + 1) * ng + i] = gen[i, RAMP_AGC] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_AGC]

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.5.2) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, BETA * ng * T + (j + 1) * ng + i] = gen[i, RAMP_AGC] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_AGC]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.6) Line flow limitation
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

        Aineq_temp = zeros((T * nl, nx))
        bineq_temp = zeros((T * nl, 1))
        for i in range(T):
            Aineq_temp[i * nl:(i + 1) * nl, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = -(
                    Distribution_factor * Cg).todense()
            bineq_temp[i * nl:(i + 1) * nl, :] = (
                    branch[:, RATE_A] - Distribution_factor * bus[:, PD] * profile[i]).reshape((nl, 1))
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        self.Distribution_factor = Distribution_factor
        self.Pd = bus[:, PD]
        self.profile = profile
        self.Cg = Cg
        Aineq_temp = zeros((T * nl, nx))
        bineq_temp = zeros((T * nl, 1))
        for i in range(T):
            Aineq_temp[i * nl:(i + 1) * nl, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = (
                    Distribution_factor * Cg).todense()
            bineq_temp[i * nl:(i + 1) * nl, :] = (
                    branch[:, RATE_A] + Distribution_factor * bus[:, PD] * profile[i]).reshape((nl, 1))
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.7)  Up and down reserve for the forecasting errors
        # Up reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RUG * ng * T + i * ng + j] = -1
            bineq_temp[i] -= delta * sum(bus[:, PD])

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # Down reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RDG * ng * T + i * ng + j] = -1
            bineq_temp[i] -= delta * sum(bus[:, PD])
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        model = {"c": c,
                 "q": q,
                 "lb": lb,
                 "ub": ub,
                 "A": Aineq,
                 "b": bineq,
                 "Aeq": Aeq,
                 "beq": beq,
                 "vtypes": vtypes}
        return model

    def problem_solving(self, model):
        """
        :return:
        """
        (xx, obj, success) = miqp(model["c"], model["q"], Aeq=model["Aeq"], beq=model["beq"],
                                  A=model["A"],
                                  b=model["b"], xmin=model["lb"], xmax=model["ub"],
                                  vtypes=model["vtypes"], objsense="min")
        xx = array(xx).reshape((len(xx), 1))
        return xx

    def result_check(self, sol):
        """

        :param sol: The solution of mathematical
        :return:
        """
        T = self.T
        ng = self.ng
        nl = self.nl
        nb = self.nb
        alpha = zeros((ng, T))
        beta = zeros((ng, T))
        ig = zeros((ng, T))
        pg = zeros((ng, T))
        rug = zeros((ng, T))
        rdg = zeros((ng, T))

        for i in range(T):
            for j in range(ng):
                alpha[j, i] = sol[ALPHA * ng * T + i * ng + j]
                beta[j, i] = sol[BETA * ng * T + i * ng + j]
                ig[j, i] = sol[IG * ng * T + i * ng + j]
                pg[j, i] = sol[PG * ng * T + i * ng + j]
                rug[j, i] = sol[RUG * ng * T + i * ng + j]
                rdg[j, i] = sol[RDG * ng * T + i * ng + j]
        pf = zeros((nl, T))
        Distribution_factor = self.Distribution_factor
        Pd = self.Pd
        profile = self.profile
        Cg = self.Cg
        for i in range(T):
            pf[:, i] = Distribution_factor * (Cg * pg[:, i] - Pd * profile[i])

        solution = {"ALPHA": alpha,
                    "BETA": beta,
                    "IG": ig,
                    "PG": pg,
                    "RUG": rug,
                    "RDG": rdg, }

        return solution


if __name__ == "__main__":
    # Import the test cases
    from unit_commitment.test_cases.case6 import case6

    two_stage_unit_commitment_robust = TwoStageUnitCommitmentRobust()
    profile = array(
        [1.75, 1.65, 1.58, 1.54, 1.55, 1.60, 1.73, 1.77, 1.86, 2.07, 2.29, 2.36, 2.42, 2.44, 2.49, 2.56, 2.56, 2.47,
         2.46, 2.37, 2.37, 2.33, 1.96, 1.96])
    case_base = case6()
    case_base["Load_profile"] = profile

    model = two_stage_unit_commitment_robust.problem_formulation(case_base)
    sol = two_stage_unit_commitment_robust.problem_solving(model)
    sol = two_stage_unit_commitment_robust.result_check(sol)
