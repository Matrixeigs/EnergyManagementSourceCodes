"""
Unit Commitment Problem Considering the Failures of Transmission Lines

References:
    [1]Flexible Operation of Batteries in Power System Scheduling With Renewable Energy

@date: 17 June 2018
@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
"""

from numpy import zeros, shape, ones, diag, concatenate, r_, arange, array, eye
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from scipy.sparse import csr_matrix as sparse

from pypower.idx_brch import F_BUS, T_BUS, BR_X, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, PD, BUS_I
from pypower.idx_gen import GEN_BUS, PG, PMAX, PMIN, RAMP_AGC, RAMP_10, RAMP_30
from pypower.idx_cost import STARTUP

from solvers.mixed_integer_quadratic_solver_cplex import mixed_integer_quadratic_programming as miqp

from unit_commitment.data_format.data_format_bess import ALPHA, BETA, IG, PG, RS, RU, RD, THETA, PL, ICS, PCS, PDC, \
    EESS, RBD, RBS, RBU, NG, NESS


class UnitCommitmentBattery():
    """"""

    def __init__(self):
        self.name = "Unit commitment with battery"

    def problem_formulation(self, case, delta=0.03, delta_r=0.02, battery=None, alpha_s=0.5, alpha_r=0.5):
        """
        Input check for the unit commitment problem
        :param cases:
        :return:
        """
        baseMVA, bus, gen, branch, gencost, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"], case[
            "gencost"], case["Load_profile"]
        MIN_UP = -2
        MIN_DOWN = -3

        # To manage the bess models
        if battery is not None:
            ness = len(battery)
            index = zeros(ness)
            for i in range(ness):
                index[i] = battery[i]["BUS"]
        else:
            ness = 0
            index = zeros(ness)

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
        self.ness = ness

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = r_[range(nl), range(nl)]  ## double set of row indices

        ## connection matrix
        Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], arange(ng))),
                    (nb, ng))
        Ce = sparse((ones(ness), (index, arange(ness))),
                    (nb, ness))

        u0 = [0] * ng  # The initial generation status
        for i in range(ng):
            u0[i] = int(gencost[i, -1] > 0)
        # Formulate a mixed integer quadratic programming problem
        # 1) Announce the variables
        # [vt,wt,ut,Pt,Rs,ru,rd]:start-up,shut-down,status,generation level, spinning reserve, up regulation reserve, down regulation reserve
        # 1.1) boundary information
        T = case["Load_profile"].shape[0]
        nx = NG * T * ng + NESS * ness * T + nb * T + nl * T
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
                lb[RS * ng * T + i * ng + j] = 0
                lb[RU * ng * T + i * ng + j] = 0
                lb[RD * ng * T + i * ng + j] = 0
                # upper boundary
                ub[ALPHA * ng * T + i * ng + j] = 1
                ub[BETA * ng * T + i * ng + j] = 1
                ub[IG * ng * T + i * ng + j] = 1
                ub[PG * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RS * ng * T + i * ng + j] = gen[j, RAMP_10]
                ub[RU * ng * T + i * ng + j] = gen[j, RAMP_AGC]
                ub[RD * ng * T + i * ng + j] = gen[j, RAMP_AGC]
                # variable types
                vtypes[IG * ng * T + i * ng + j] = "B"
        for i in range(T):
            for j in range(ness):
                # lower boundary
                lb[NG * ng * T + ICS * ness * T + i * ness + j] = 0
                lb[NG * ng * T + PCS * ness * T + i * ness + j] = 0
                lb[NG * ng * T + PDC * ness * T + i * ness + j] = 0
                lb[NG * ng * T + EESS * ness * T + i * ness + j] = battery[j]["EMIN"]
                lb[NG * ng * T + RBS * ness * T + i * ness + j] = 0
                lb[NG * ng * T + RBU * ness * T + i * ness + j] = 0
                lb[NG * ng * T + RBD * ness * T + i * ness + j] = 0
                # upper boundary
                ub[NG * ng * T + ICS * ness * T + i * ness + j] = 1
                ub[NG * ng * T + PCS * ness * T + i * ness + j] = battery[j]["PCH_MAX"]
                ub[NG * ng * T + PDC * ness * T + i * ness + j] = battery[j]["PDC_MAX"]
                ub[NG * ng * T + EESS * ness * T + i * ness + j] = battery[j]["EMAX"]
                ub[NG * ng * T + RBS * ness * T + i * ness + j] = battery[j]["PCH_MAX"] + battery[j]["PDC_MAX"]
                ub[NG * ng * T + RBU * ness * T + i * ness + j] = battery[j]["PCH_MAX"] + battery[j]["PDC_MAX"]
                ub[NG * ng * T + RBD * ness * T + i * ness + j] = battery[j]["PCH_MAX"] + battery[j]["PDC_MAX"]
                # variable types
                vtypes[NG * ng * T + ICS * ness * T + i * ness + j] = "B"
                if i == T - 1:
                    lb[NG * ng * T + EESS * ness * T + i * ness + j] = battery[j]["E0"]
                    ub[NG * ng * T + PDC * ness * T + i * ness + j] = battery[j]["E0"]

        # The bus angle
        for i in range(T):
            for j in range(nb):
                lb[NG * ng * T + NESS * ness * T + i * nb + j] = -360
                ub[NG * ng * T + NESS * ness * T + i * nb + j] = 360
                if bus[j, BUS_TYPE] == REF:
                    lb[NG * ng * T + NESS * ness * T + i * nb + j] = 0
                    ub[NG * ng * T + NESS * ness * T + i * nb + j] = 0
        # The power flow
        for i in range(T):
            for j in range(nl):
                lb[NG * ng * T + NESS * ness * T + T * nb + i * nl + j] = -branch[j, RATE_A]
                ub[NG * ng * T + NESS * ness * T + T * nb + i * nl + j] = branch[j, RATE_A]

        c = zeros((nx, 1))
        q = zeros((nx, 1))
        for i in range(T):
            for j in range(ng):
                # cost
                c[ALPHA * ng * T + i * ng + j] = gencost[j, STARTUP]
                c[IG * ng * T + i * ng + j] = gencost[j, 6]
                c[PG * ng * T + i * ng + j] = gencost[j, 5]

                q[PG * ng * T + i * ng + j] = gencost[j, 4]

        # 2) Constraint set
        # 2.1) Power balance equation, for each node
        Aeq = zeros((T * nb, nx))
        beq = zeros((T * nb, 1))
        for i in range(T):
            # For the unit
            Aeq[i * nb:(i + 1) * nb, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = Cg.todense()
            # For the battery energy storage systems
            Aeq[i * nb:(i + 1) * nb,
            NG * ng * T + PCS * ness * T + i * ness:NG * ng * T + PCS * ness * T + (i + 1) * ness] = -Ce.todense()
            Aeq[i * nb:(i + 1) * nb,
            NG * ng * T + PDC * ness * T + i * ness:NG * ng * T + PDC * ness * T + (i + 1) * ness] = Ce.todense()
            # For the transmission lines
            Aeq[i * nb:(i + 1) * nb,
            NG * ng * T + NESS * ness * T + T * nb + i * nl: NG * ng * T + NESS * ness * T + T * nb + (i + 1) * nl] = -(
                Cft.transpose()).todense()

            beq[i * nb:(i + 1) * nb, 0] = profile[i] * bus[:, PD]

        # 2.2) Status transformation of each unit
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

        # 2.3) Transmission line flows
        Aeq_temp = zeros((T * nl, nx))
        beq_temp = zeros((T * nl, 1))
        X = zeros((nl, nl))
        for i in range(nl):
            X[i, i] = 1 / branch[i, BR_X]

        for i in range(T):
            # For the unit
            Aeq_temp[i * nl:(i + 1) * nl,
            NG * ng * T + NESS * ness * T + T * nb + i * nl:NG * ng * T + NESS * ness * T + T * nb + (
                    i + 1) * nl] = -eye(nl)
            Aeq_temp[i * nl:(i + 1) * nl,
            NG * ng * T + NESS * ness * T + i * nb:NG * ng * T + NESS * ness * T + (i + 1) * nb] = X.dot(Cft.todense())

        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate((beq, beq_temp), axis=0)
        # 2.4) Energy status transfer
        Aeq_temp = zeros((T * ness, nx))
        beq_temp = zeros((T * ness, 1))
        for i in range(T):
            for j in range(ness):
                Aeq_temp[i * ness + j, NG * ng * T + PCS * ness * T + i * ness + j] = battery[j]["EFF_CH"]
                Aeq_temp[i * ness + j, NG * ng * T + PDC * ness * T + i * ness + j] = -1 / battery[j]["EFF_CH"]
                Aeq_temp[i * ness + j, NG * ng * T + EESS * ness * T + i * ness + j] = -1
                if i == 0:
                    beq_temp[i * ness + j] = -battery[j]["E0"]
                else:
                    Aeq_temp[i * ness + j, NG * ng * T + EESS * ness * T + (i - 1) * ness + j] = 1
        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate((beq, beq_temp), axis=0)

        # 2.5) Power range limitation
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
                Aineq_temp[i * ng + j, RD * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, PMAX]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = 1
                Aineq_temp[i * ng + j, RU * ng * T + i * ng + j] = 1
                Aineq_temp[i * ng + j, RS * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.6) Start up and shut down time limitation
        UP_LIMIT = [0] * ng
        DOWN_LIMIT = [0] * ng
        for i in range(ng):
            UP_LIMIT[i] = T - int(gencost[i, MIN_UP])
            DOWN_LIMIT[i] = T - int(gencost[i, MIN_DOWN])
        # 2.6.1) Up limit
        Aineq_temp = zeros((sum(UP_LIMIT), nx))
        bineq_temp = zeros((sum(UP_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gencost[i, MIN_UP]), T):
                for k in range(j - int(gencost[i, MIN_UP]), j):
                    Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), ALPHA * ng * T + k * ng + i] = 1
                Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), IG * ng * T + j * ng + i] = -1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6.2) Down limit
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

        # 2.7) Ramp constraints:
        # 2.7.1) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, ALPHA * ng * T + (j + 1) * ng + i] = gen[i, RAMP_30] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_30]

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # # 2.7.2) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, BETA * ng * T + (j + 1) * ng + i] = gen[i, RAMP_30] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_30]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8)  Reserve constraints
        # 2.8.1) Rs<=Ig*RAMP_10
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, RAMP_10]
                Aineq_temp[i * ng + j, RS * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8.2) ru<=Ig*RAMP_AGC
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, RAMP_AGC]
                Aineq_temp[i * ng + j, RU * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8.3) rd<=Ig*RAMP_AGC
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, RAMP_AGC]
                Aineq_temp[i * ng + j, RD * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8.4) Pcs<=Ics*Pcs_max
        Aineq_temp = zeros((T * ness, nx))
        bineq_temp = zeros((T * ness, 1))
        for i in range(T):
            for j in range(ness):
                Aineq_temp[i * ness + j, NG * ng * T + ICS * ness * T + i * ness + j] = -battery[j]["PCH_MAX"]
                Aineq_temp[i * ness + j, NG * ng * T + PCS * ness * T + i * ness + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8.5) Pcs<=(1-Ics)*Pdc_max
        Aineq_temp = zeros((T * ness, nx))
        bineq_temp = zeros((T * ness, 1))
        for i in range(T):
            for j in range(ness):
                Aineq_temp[i * ness + j, NG * ng * T + ICS * ness * T + i * ness + j] = battery[j]["PDC_MAX"]
                Aineq_temp[i * ness + j, NG * ng * T + PDC * ness * T + i * ness + j] = 1
                bineq_temp[i * ness + j] = battery[j]["PDC_MAX"]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8.5) Pess_dc-Pess_ch+Rbs+rbu<=Pess_dc_max
        Aineq_temp = zeros((T * ness, nx))
        bineq_temp = zeros((T * ness, 1))
        for i in range(T):
            for j in range(ness):
                Aineq_temp[i * ness + j, NG * ng * T + PCS * ness * T + i * ness + j] = -1
                Aineq_temp[i * ness + j, NG * ng * T + PDC * ness * T + i * ness + j] = 1
                Aineq_temp[i * ness + j, NG * ng * T + RBS * ness * T + i * ness + j] = 1
                Aineq_temp[i * ness + j, NG * ng * T + RBU * ness * T + i * ness + j] = 1
                bineq_temp[i * ness + j] = battery[j]["PDC_MAX"]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8.6) Pess_ch-Pess_dc+rbd<=Pess_ch_max
        Aineq_temp = zeros((T * ness, nx))
        bineq_temp = zeros((T * ness, 1))
        for i in range(T):
            for j in range(ness):
                Aineq_temp[i * ness + j, NG * ng * T + PCS * ness * T + i * ness + j] = 1
                Aineq_temp[i * ness + j, NG * ng * T + PDC * ness * T + i * ness + j] = -1
                Aineq_temp[i * ness + j, NG * ng * T + RBD * ness * T + i * ness + j] = 1
                bineq_temp[i * ness + j] = battery[j]["PCH_MAX"]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8.7) alpha_s*Rbs+alpha_r*rbu<=eff_dc*(E-Emin)
        Aineq_temp = zeros((T * ness, nx))
        bineq_temp = zeros((T * ness, 1))
        for i in range(T):
            for j in range(ness):
                Aineq_temp[i * ness + j, NG * ng * T + EESS * ness * T + i * ness + j] = -battery[j]["EFF_DC"]
                Aineq_temp[i * ness + j, NG * ng * T + RBS * ness * T + i * ness + j] = alpha_s
                Aineq_temp[i * ness + j, NG * ng * T + RBU * ness * T + i * ness + j] = alpha_r
                bineq_temp[i * ness + j] = -battery[j]["EFF_DC"] * battery[j]["EMIN"]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8.8) alpha_r*rbd<=(E_max-E)/eff_ch
        Aineq_temp = zeros((T * ness, nx))
        bineq_temp = zeros((T * ness, 1))
        for i in range(T):
            for j in range(ness):
                Aineq_temp[i * ness + j, NG * ng * T + EESS * ness * T + i * ness + j] = 1
                Aineq_temp[i * ness + j, NG * ng * T + RBD * ness * T + i * ness + j] = alpha_r * battery[j]["EFF_CH"]
                bineq_temp[i * ness + j] = battery[j]["EMAX"] / battery[j]["EFF_CH"]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.9)  Up and down reserve for the forecasting errors
        # Spinning reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RS * ng * T + i * ng + j] = -1
            for j in range(ness):
                Aineq_temp[i, NG * ng * T + RBS * ness * T + i * ness + j] = -1

            bineq_temp[i] -= delta * profile[i] * sum(bus[:, PD])
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # Up reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RU * ng * T + i * ng + j] = -1
            for j in range(ness):
                Aineq_temp[i, NG * ng * T + RBU * ness * T + i * ness + j] = -1
            bineq_temp[i] -= delta_r * profile[i] * sum(bus[:, PD])
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # Down reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RD * ng * T + i * ng + j] = -1
            for j in range(ness):
                Aineq_temp[i, NG * ng * T + RBD * ness * T + i * ness + j] = -1
            bineq_temp[i] -= delta_r * profile[i] * sum(bus[:, PD])
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

        :param model: Formulated mathematical models
        :return:
        """
        (xx, obj, success) = miqp(model["c"], model["q"], Aeq=model["Aeq"], beq=model["beq"],
                                  A=model["A"],
                                  b=model["b"], xmin=model["lb"], xmax=model["ub"],
                                  vtypes=model["vtypes"], objsense="min")
        xx = array(xx).reshape((len(xx), 1))
        return xx, obj

    def result_check(self, sol):
        """

        :param sol: The solution of mathematical
        :return:
        """
        T = self.T
        ng = self.ng
        nl = self.nl
        nb = self.nb
        ness = self.ness

        alpha = zeros((ng, T))
        beta = zeros((ng, T))
        ig = zeros((ng, T))
        pg = zeros((ng, T))
        Rs = zeros((ng, T))
        ru = zeros((ng, T))
        rd = zeros((ng, T))

        theta = zeros((nb, T))
        pf = zeros((nl, T))

        for i in range(T):
            for j in range(ng):
                alpha[j, i] = sol[ALPHA * ng * T + i * ng + j]
                beta[j, i] = sol[BETA * ng * T + i * ng + j]
                ig[j, i] = sol[IG * ng * T + i * ng + j]
                pg[j, i] = sol[PG * ng * T + i * ng + j]
                Rs[j, i] = sol[RS * ng * T + i * ng + j]
                ru[j, i] = sol[RU * ng * T + i * ng + j]
                rd[j, i] = sol[RD * ng * T + i * ng + j]

        ics = zeros((ness, T))
        pcs = zeros((ness, T))
        pdc = zeros((ness, T))
        eess = zeros((ness, T))
        rbs = zeros((ness, T))
        rbu = zeros((ness, T))
        rbd = zeros((ness, T))

        for i in range(T):
            for j in range(ness):
                ics[j, i] = sol[NG * ng * T + ICS * ness * T + i * ness + j]
                pcs[j, i] = sol[NG * ng * T + PCS * ness * T + i * ness + j]
                pdc[j, i] = sol[NG * ng * T + PDC * ness * T + i * ness + j]
                eess[j, i] = sol[NG * ng * T + EESS * ness * T + i * ness + j]
                rbs[j, i] = sol[NG * ng * T + RBS * ness * T + i * ness + j]
                rbu[j, i] = sol[NG * ng * T + RBU * ness * T + i * ness + j]
                rbd[j, i] = sol[NG * ng * T + RBD * ness * T + i * ness + j]

        for i in range(T):
            for j in range(nb):
                theta[j, i] = sol[NG * ng * T + NESS * ness * T + i * nb + j]

        for i in range(T):
            for j in range(nl):
                pf[j, i] = sol[NG * ng * T + NESS * ness * T + T * nb + i * nl + j]

        solution = {"ALPHA": alpha,
                    "BETA": beta,
                    "IG": ig,
                    "PG": pg,
                    "RS": Rs,
                    "RU": ru,
                    "RD": rd, }

        return solution


if __name__ == "__main__":
    # Import the test cases
    from unit_commitment.test_cases.case6 import case6

    BESS = []
    bess = {
        "BUS": 1,
        "E0": 1,
        "EMIN": 0.1,
        "EMAX": 2,
        "PCH_MAX": 2,
        "PDC_MAX": 2,
        "EFF_DC": 0.9,
        "EFF_CH": 0.9,
        "COST": 2,
    }
    BESS.append(bess)
    bess = {
        "BUS": 2,
        "E0": 1,
        "EMIN": 0.1,
        "EMAX": 2,
        "PCH_MAX": 0,
        "PDC_MAX": 0,
        "EFF_DC": 0.9,
        "EFF_CH": 0.9,
        "COST": 2,
    }
    BESS.append(bess)

    unit_commitment_battery = UnitCommitmentBattery()
    profile = array(
        [1.75, 1.65, 1.58, 1.54, 1.55, 1.60, 1.73, 1.77, 1.86, 2.07, 2.29, 2.36, 2.42, 2.44, 2.49, 2.56, 2.56, 2.47,
         2.46, 2.37, 2.37, 2.33, 1.96, 1.96])
    case_base = case6()

    case_base["Load_profile"] = profile

    model = unit_commitment_battery.problem_formulation(case_base, battery=BESS)

    (sol, obj) = unit_commitment_battery.problem_solving(model)
    sol = unit_commitment_battery.result_check(sol)
