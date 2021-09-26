"""
Two stage stochastic unit commitment for jointed wind  hydro dispatch
This algorithm is solved using multi-cuts benders decomposition
@author: Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
"""

from numpy import zeros, shape, ones, diag, concatenate, r_, arange, array, eye, random
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp
from scipy.sparse import csr_matrix as sparse
from solvers.benders_solvers import linear_programming as lp
from pypower.idx_brch import F_BUS, T_BUS, BR_X, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, PD, BUS_I
from pypower.idx_gen import GEN_BUS, PG, PMAX, PMIN, RAMP_AGC, RAMP_10, RAMP_30
from pypower.idx_cost import STARTUP
import os
from multiprocessing import Pool
from unit_commitment.data_format.data_format_contigency import ALPHA, BETA, IG, PG, RS, RU, RD, THETA, PL, NG


class TwoStageStochasticUnitCommitment():

    def __init__(self):
        self.name = "Two-stage stochastic unit commitment"

    def problem_formulation_first_stage(self, case, delta=0.10, delta_r=0.03):
        baseMVA, bus, gen, branch, gencost, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"], case[
            "gencost"], case["Load_profile"]
        MIN_UP = -3

        # Modify the bus, gen and branch matrix
        bus[:, BUS_I] = bus[:, BUS_I] - 1
        gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1
        branch[:, F_BUS] = branch[:, F_BUS] - 1
        branch[:, T_BUS] = branch[:, T_BUS] - 1
        gen[:, RAMP_10] = gencost[:, -8] * 30
        gen[:, RAMP_AGC] = gencost[:, -8] * 10
        gen[:, RAMP_30] = gencost[:, -8] * 60

        ng = shape(case['gen'])[0]  # number of schedule injections
        nl = shape(case['branch'])[0]  ## number of branches
        nb = shape(case['bus'])[0]  ## number of branches

        # Pass the information
        self.ng = ng
        self.nb = nb
        self.nl = nl
        self.bus = bus
        self.branch = branch
        self.gen = gen

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = r_[range(nl), range(nl)]  ## double set of row indices

        ## connection matrix
        Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], arange(ng))),
                    (nb, ng))

        u0 = [0] * ng  # The initial generation status
        for i in range(ng):
            u0[i] = int(gencost[i, 9] > 0)

        T = case["Load_profile"].shape[0]
        self.T = T

        nx = NG * T * ng + nb * T + nl * T
        lb = zeros((nx, 1))
        ub = zeros((nx, 1))
        vtypes = ["c"] * nx
        self.nx = nx

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

        # The bus angle
        for i in range(T):
            for j in range(nb):
                lb[NG * ng * T + i * nb + j] = -360
                ub[NG * ng * T + i * nb + j] = 360
                if bus[j, BUS_TYPE] == REF:
                    lb[NG * ng * T + i * nb + j] = 0
                    ub[NG * ng * T + i * nb + j] = 0

        # The power flow
        for i in range(T):
            for j in range(nl):
                lb[NG * ng * T + T * nb + i * nl + j] = -branch[j, RATE_A] * 10
                ub[NG * ng * T + T * nb + i * nl + j] = branch[j, RATE_A] * 10

        c = zeros((nx, 1))
        for i in range(T):
            for j in range(ng):
                # cost, the linear objective value
                c[ALPHA * ng * T + i * ng + j] = gencost[j, STARTUP]
                c[IG * ng * T + i * ng + j] = gencost[j, 6]
                c[PG * ng * T + i * ng + j] = gencost[j, 5]

        # 2) Constraint set
        # 2.1) Power balance equation, for each node
        Aeq = zeros((T * nb, nx))
        beq = zeros((T * nb, 1))
        for i in range(T):
            # For the unit
            Aeq[i * nb:(i + 1) * nb, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = Cg.todense()
            # For the transmission lines
            Aeq[i * nb:(i + 1) * nb, NG * ng * T + T * nb + i * nl: NG * ng * T + T * nb + (i + 1) * nl] = -(
                Cft.transpose()).todense()

            beq[i * nb:(i + 1) * nb, 0] = profile[i] * bus[:, PD]
        self.u_mean = beq  # Stochastic variables
        self.u_delta = beq * delta_r
        self.nu = T * nb
        self.Cg = Cg
        self.Cft = Cft

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
            Aeq_temp[i * nl:(i + 1) * nl, NG * ng * T + T * nb + i * nl:NG * ng * T + T * nb + (i + 1) * nl] = -eye(nl)
            Aeq_temp[i * nl:(i + 1) * nl, NG * ng * T + i * nb:NG * ng * T + (i + 1) * nb] = X.dot(Cft.todense())

        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate((beq, beq_temp), axis=0)
        # 2.4) Power range limitation
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

        # 2.5) Start up and shut down time limitation
        UP_LIMIT = [0] * ng
        DOWN_LIMIT = [0] * ng
        for i in range(ng):
            UP_LIMIT[i] = T - int(gencost[i, MIN_UP])
            DOWN_LIMIT[i] = T - int(gencost[i, MIN_UP])
        # 2.5.1) Up limit
        Aineq_temp = zeros((sum(UP_LIMIT), nx))
        bineq_temp = zeros((sum(UP_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gencost[i, MIN_UP]), T):
                for k in range(j - int(gencost[i, MIN_UP]), j):
                    Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), ALPHA * ng * T + k * ng + i] = 1
                Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), IG * ng * T + j * ng + i] = -1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.5.2) Down limit
        Aineq_temp = zeros((sum(DOWN_LIMIT), nx))
        bineq_temp = ones((sum(DOWN_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gencost[i, MIN_UP]), T):
                for k in range(j - int(gencost[i, MIN_UP]), j):
                    Aineq_temp[
                        sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), BETA * ng * T + k * ng + i] = 1
                Aineq_temp[sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), IG * ng * T + j * ng + i] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.6) Ramp constraints:
        # 2.6.1) Ramp up limitation
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
        # # 2.6.2) Ramp up limitation
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
        # 2.7)  Reserve constraints
        # 2.7.1) Rs<=Ig*RAMP_10
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, RAMP_10]
                Aineq_temp[i * ng + j, RS * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.7.2) ru<=Ig*RAMP_AGC
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, RAMP_AGC]
                Aineq_temp[i * ng + j, RU * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.7.3) rd<=Ig*RAMP_AGC
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, RAMP_AGC]
                Aineq_temp[i * ng + j, RD * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.8)  Up and down reserve for the forecasting errors
        # 2.8.1) Spinning reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RS * ng * T + i * ng + j] = -1
            bineq_temp[i] -= delta * profile[i] * sum(bus[:, PD])
        # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        # bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.8.2) Up regulation reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RU * ng * T + i * ng + j] = -1
            bineq_temp[i] -= delta_r * profile[i] * sum(bus[:, PD])
        # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        # bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.8.3) Down regulation reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RD * ng * T + i * ng + j] = -1
            bineq_temp[i] -= delta_r * profile[i] * sum(bus[:, PD])
        # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        # bineq = concatenate((bineq, bineq_temp), axis=0)

        model = {"c": c,
                 "lb": lb,
                 "ub": ub,
                 "A": Aineq,
                 "b": bineq,
                 "Aeq": Aeq,
                 "beq": beq,
                 "vtypes": vtypes}

        return model

    def problem_formulation_seond_stage(self):
        """
        Problem formulation for the second stage problem
        Found problem: The upper and lower boundary should be modified to the standard format
        :return:
        """

        ng = self.ng
        nb = self.nb
        nl = self.nl
        T = self.T
        bus = self.bus
        branch = self.branch
        gen = self.gen
        # Number of variables
        nx = ng * T + nb * T + nb * T + nl * T
        lb = zeros((nx, 1))
        ub = zeros((nx, 1))
        c = zeros((nx, 1))
        pg = 0
        pd = 1
        bigM = 10 ** 4

        for i in range(T):
            for j in range(ng):
                # real-time power dispatch
                lb[pg * ng * T + i * ng + j] = 0
                ub[pg * ng * T + i * ng + j] = gen[j, PMAX]
            for j in range(nb):
                # load shedding at different buses
                lb[pd * ng * T + i * nb + j] = 0
                ub[pd * ng * T + i * nb + j] = bigM
                c[pd * ng * T + i * nb + j] = bigM

        for i in range(T):
            for j in range(nb):
                # The bus angle
                lb[ng * T + nb * T + i * nb + j] = -360
                ub[ng * T + nb * T + i * nb + j] = 360
                if bus[j, BUS_TYPE] == REF:
                    lb[ng * T + nb * T + i * nb + j] = 0
                    ub[ng * T + nb * T + i * nb + j] = 0

        for i in range(T):
            for j in range(nl):
                # The power flow
                lb[ng * T + nb * T + nb * T + i * nl + j] = -branch[j, RATE_A]
                ub[ng * T + nb * T + nb * T + i * nl + j] = branch[j, RATE_A]
        # Construct the constraint set
        # 3.1) Power balance constraints
        NX = self.nx
        nu = self.nu
        Cg = self.Cg
        Cft = self.Cft

        E_temp = zeros((T * nb, NX))
        M_temp = zeros((T * nb, nu))
        G_temp = zeros((T * nb, nx))
        h_temp = zeros((T * nb, 1))
        for i in range(T):
            # For the unit
            G_temp[i * nb:(i + 1) * nb, pg * ng * T + i * ng:pg * ng * T + (i + 1) * ng] = Cg.todense()
            # For the load shedding
            G_temp[i * nb:(i + 1) * nb, pd * ng * T + i * nb:pd * ng * T + (i + 1) * nb] = eye(nb)
            # For the transmission lines
            G_temp[i * nb:(i + 1) * nb, ng * T + nb * T + nb * T + i * nl: ng * T + nb * T + nb * T + (i + 1) * nl] = -(
                Cft.transpose()).todense()
            M_temp[i * nb:(i + 1) * nb, i * nb:(i + 1) * nb] = -eye(nb)
        # Update G,M,E,h
        G = concatenate([G_temp, -G_temp])
        M = concatenate([M_temp, -M_temp])
        E = concatenate([E_temp, -E_temp])
        h = concatenate([h_temp, -h_temp])
        # 3.2 Line flow equation
        E_temp = zeros((T * nl, NX))
        M_temp = zeros((T * nl, nu))
        G_temp = zeros((T * nl, nx))
        h_temp = zeros((T * nl, 1))

        X = zeros((nl, nl))
        for i in range(nl):
            X[i, i] = 1 / branch[i, BR_X]

        for i in range(T):
            # For the unit
            G_temp[i * nl:(i + 1) * nl,
            ng * T + nb * T + nb * T + i * nl:ng * T + nb * T + nb * T + (i + 1) * nl] = -eye(nl)
            G_temp[i * nl:(i + 1) * nl, ng * T + nb * T + i * nb: ng * T + nb * T + (i + 1) * nb] = X.dot(Cft.todense())
        G = concatenate([G, G_temp, -G_temp])
        M = concatenate([M, M_temp, -M_temp])
        E = concatenate([E, E_temp, -E_temp])
        h = concatenate([h, h_temp, -h_temp])

        # 3.3) Power range limitation
        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                G_temp[i * ng + j, pg * ng * T + i * ng + j] = 1
                E_temp[i * ng + j, PG * ng * T + i * ng + j] = -1
                E_temp[i * ng + j, RD * ng * T + i * ng + j] = 1
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])

        E_temp = zeros((T * ng, NX))
        M_temp = zeros((T * ng, nu))
        G_temp = zeros((T * ng, nx))
        h_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                G_temp[i * ng + j, pg * ng * T + i * ng + j] = -1
                E_temp[i * ng + j, PG * ng * T + i * ng + j] = 1
                E_temp[i * ng + j, RU * ng * T + i * ng + j] = 1
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        # 3.4) Load shedding constraint
        E_temp = zeros((T * nb, NX))
        M_temp = zeros((T * nb, nu))
        G_temp = zeros((T * nb, nx))
        h_temp = zeros((T * nb, 1))
        for i in range(T):
            for j in range(nb):
                G_temp[i * nb + j, ng * T + i * nb + j] = -1
                M_temp[i * nb + j, i * nb + j] = 1
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        # 3.5) Upper and lower boundary inforamtion
        E_temp = zeros((nx, NX))
        M_temp = zeros((nx, nu))
        G_temp = eye(nx)
        h_temp = lb
        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])

        E_temp = zeros((nx, NX))
        M_temp = zeros((nx, nu))
        G_temp = -eye(nx)
        h_temp = -ub

        G = concatenate([G, G_temp])
        M = concatenate([M, M_temp])
        E = concatenate([E, E_temp])
        h = concatenate([h, h_temp])
        d = c

        model = {"G": G,
                 "M": M,
                 "E": E,
                 "h": h,
                 "d": d}
        # Modify the lower boundary

        return model

    def bender_decomposition(self, model_first_stage, model_second_stage, n_scenario, xx):
        """
        Bender decomposition strategies
        :param model_first_stage:
        :param model_second_stage:
        :param n_scenario:
        :param xx:
        :return:
        """
        h = model_second_stage["h"]
        G = model_second_stage["G"]
        M = model_second_stage["M"]
        E = model_second_stage["E"]
        d = model_second_stage["d"]
        nu = self.nu
        u_mean = self.u_mean
        u_delta = self.u_delta
        NX = self.nx

        xx = array(xx).reshape((len(xx), 1))
        u_scenario = zeros((nu, n_scenario))
        for i in range(n_scenario):
            u_scenario[:, i] = (u_mean - u_delta + 2 * diag(random.rand(nu)).dot(u_delta)).reshape(nu)

        model_second_stage_dual = [0] * n_scenario

        n_processors = os.cpu_count()
        for i in range(n_scenario):
            model_second_stage_dual[i] = {"c": h - M.dot((u_scenario[:, i]).reshape(nu, 1)) - E.dot(xx),
                                          "Aeq": G.transpose(),
                                          "beq": d,
                                          "lb": zeros((h.shape[0], 1))}
        # sub_problem_dual(model_second_stage_dual[0])
        with Pool(n_processors) as p:
            result_second_stage_dual = list(p.map(sub_problem_dual, model_second_stage_dual))

        # Modify the first stage optimization problem
        model_first_stage["c"] = concatenate([model_first_stage["c"], ones((n_scenario, 1)) / n_scenario])
        model_first_stage["lb"] = concatenate([model_first_stage["lb"], ones((n_scenario, 1)) * (-10 ** 8)])
        model_first_stage["ub"] = concatenate([model_first_stage["ub"], ones((n_scenario, 1)) * (10 ** 8)])
        model_first_stage["A"] = concatenate(
            [model_first_stage["A"], zeros((model_first_stage["A"].shape[0], n_scenario))],
            axis=1)
        model_first_stage["Aeq"] = concatenate(
            [model_first_stage["Aeq"], zeros((model_first_stage["Aeq"].shape[0], n_scenario))],
            axis=1)
        model_first_stage["vtypes"] += ["c"] * n_scenario

        # Obtain cuts for the first stage optimization
        Acuts = zeros((n_scenario, NX + n_scenario))
        bcuts = zeros((n_scenario, 1))
        for i in range(n_scenario):
            pi_temp = array(result_second_stage_dual[i]["x"]).reshape((1, len(result_second_stage_dual[i]["x"])))
            Acuts[i, 0:NX] = -pi_temp.dot(E)
            Acuts[i, NX + i] = -1
            bcuts[i] = -pi_temp.dot(h - M.dot((u_scenario[:, i]).reshape(nu, 1)))
        model_first_stage["A"] = concatenate([model_first_stage["A"], Acuts])
        model_first_stage["b"] = concatenate([model_first_stage["b"], bcuts])

        (xx, obj, success) = milp(model_first_stage["c"], Aeq=model_first_stage["Aeq"], beq=model_first_stage["beq"],
                                  A=model_first_stage["A"],
                                  b=model_first_stage["b"], xmin=model_first_stage["lb"], xmax=model_first_stage["ub"],
                                  vtypes=model_first_stage["vtypes"], objsense="min")

        xx = array(xx[0:NX]).reshape((NX, 1))

        LB = obj
        UB = 0
        for i in range(n_scenario):
            UB += result_second_stage_dual[i]["objvalue"] / n_scenario
        UB += (model_first_stage["c"][0:NX].transpose()).dot(xx)[0][0]
        Gap = abs((UB - LB) / LB)

        print("The upper boundary is {0}".format(UB))
        print("The lower boundary is {0}".format(LB))
        print("The gap is {0}".format(Gap))

        k = 1
        kmax = 1000
        while Gap > 10 ** -3:
            # Update the second stage optimization problem
            for i in range(n_scenario):
                model_second_stage_dual[i] = {"c": h - M.dot((u_scenario[:, i]).reshape(nu, 1)) - E.dot(xx),
                                              "Aeq": G.transpose(),
                                              "beq": d,
                                              "lb": zeros((h.shape[0], 1))}
            with Pool(n_processors) as p:
                result_second_stage_dual = list(p.map(sub_problem_dual, model_second_stage_dual))

            # Add cuts
            Acuts = zeros((n_scenario, NX + n_scenario))
            bcuts = zeros((n_scenario, 1))
            for i in range(n_scenario):
                pi_temp = array(result_second_stage_dual[i]["x"]).reshape((1, len(result_second_stage_dual[i]["x"])))
                Acuts[i, 0:NX] = -pi_temp.dot(E)
                Acuts[i, NX + i] = -1
                bcuts[i] = -pi_temp.dot(h - M.dot((u_scenario[:, i]).reshape(nu, 1)))
            model_first_stage["A"] = concatenate([model_first_stage["A"], Acuts])
            model_first_stage["b"] = concatenate([model_first_stage["b"], bcuts])

            (xx, obj, success) = milp(model_first_stage["c"],
                                      Aeq=model_first_stage["Aeq"],
                                      beq=model_first_stage["beq"],
                                      A=model_first_stage["A"],
                                      b=model_first_stage["b"],
                                      xmin=model_first_stage["lb"],
                                      xmax=model_first_stage["ub"],
                                      vtypes=model_first_stage["vtypes"], objsense="min")

            xx = array(xx[0:NX]).reshape((NX, 1))

            LB = obj
            UB_temp = 0
            for i in range(n_scenario):
                UB_temp += result_second_stage_dual[i]["objvalue"] / n_scenario
            UB_temp += (model_first_stage["c"][0:NX].transpose()).dot(xx)[0][0]
            UB = min(UB, UB_temp)
            Gap = abs((UB - LB) / LB)
            print("The upper boundary is {0}".format(UB))
            print("The lower boundary is {0}".format(LB))
            print("The gap is {0}".format(Gap))

            k += 1
            if k > kmax:
                break

        return xx, obj

    def problem_solving(self, model):
        """

        :param model: Formulated mathematical models
        :return:
        """
        (xx, obj, success) = milp(model["c"], Aeq=model["Aeq"], beq=model["beq"],
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

        for i in range(T):
            for j in range(nb):
                theta[j, i] = sol[NG * ng * T + i * nb + j]

        for i in range(T):
            for j in range(nl):
                pf[j, i] = sol[NG * ng * T + T * nb + i * nl + j]

        solution = {"ALPHA": alpha,
                    "BETA": beta,
                    "IG": ig,
                    "PG": pg,
                    "RS": Rs,
                    "RU": ru,
                    "RD": rd,
                    "THETA": theta,
                    "PF": pf}

        return solution


def sub_problem_dual(model):
    """
    Solve each slave problems
    :param model:
    :return:
        """
    (x, objvalue, status) = lp(model["c"], Aeq=model["Aeq"], beq=model["beq"], xmin=model["lb"])

    sol = {"x": x,
           "objvalue": objvalue,
           "status": status}

    return sol


if __name__ == "__main__":
    from unit_commitment.test_cases.case24 import case24

    case_base = case24()
    profile = array(
        [1.75, 1.65, 1.58, 1.54, 1.55, 1.60, 1.73, 1.77, 1.86, 2.07, 2.29, 2.36, 2.42, 2.44, 2.49, 2.56, 2.56, 2.47,
         2.46, 2.37, 2.37, 2.33, 1.96, 1.96]) / 3
    case_base["Load_profile"] = profile

    two_stage_unit_commitment = TwoStageStochasticUnitCommitment()

    model_first_stage = two_stage_unit_commitment.problem_formulation_first_stage(case_base)
    model_second_stage = two_stage_unit_commitment.problem_formulation_seond_stage()
    (sol, obj) = two_stage_unit_commitment.problem_solving(model_first_stage)
    (sol, obj) = two_stage_unit_commitment.bender_decomposition(model_first_stage, model_second_stage, 100, sol)
    sol = two_stage_unit_commitment.result_check(sol)
