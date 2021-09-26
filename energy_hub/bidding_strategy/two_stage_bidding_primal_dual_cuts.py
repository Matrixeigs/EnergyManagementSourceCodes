"""
Primal and dual cuts for two stage bidding strategies of energy hubs
"""

from energy_hub.bidding_strategy.bidding_strategy import EnergyHubManagement  # import the energy hub management class
from numpy import zeros, ones, array, eye, hstack, vstack, inf, transpose
import numpy as np
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp
from solvers.benders_decomposition import BendersDecomposition


class TwoStageBidding():
    def __init__(self):
        self.name = "two_stage_bidding_strategy"

    def problem_formualtion(self, ELEC_DA=None, ELEC_RT=None, BIC=None, ESS=None, CCHP=None, HVAC=None, THERMAL=None,
                            CHIL=None, BOIL=None, T=None, N=None):
        """

        :param ELEC: Scenarios in the second stage scheduling
        :param BIC:
        :param ESS:
        :param CCHP:
        :param HVAC:
        :param THERMAL:
        :param CHIL:
        :param BOIL:
        :param T:
        :param N: The number of scenarios in the second stage operation.
        :return:
        """
        energy_hub_management = EnergyHubManagement()  # Initialize the solutions
        # The second stage decision are shown as follows, no decomposition
        model = energy_hub_management.problem_formulation(ELEC=ELEC_DA, CCHP=CCHP, THERMAL=THERMAL, BIC=BIC, ESS=ESS,
                                                          HVAC=HVAC, BOIL=BOIL, CHIL=CHIL, T=T)

        neq = model["Aeq"].shape[0]
        nx = model["Aeq"].shape[1]

        if model["A"] is None:
            nineq = 0
        else:
            nineq = model["A"].shape[0]

        self.nx = nx
        self.T = T
        self.N = N
        Aeq_second_stage = zeros((neq * N, nx * N))
        beq_second_stage = zeros((neq * N, 1))
        lb_second_stage = zeros((nx * N, 1))
        ub_second_stage = zeros((nx * N, 1))
        c_second_stage = zeros((nx * N, 1))
        elec = [0] * N  # using the list to store the data set
        model_second_stage = [0] * N
        for i in range(N):
            elec[i] = {"UG_MAX": ELEC_DA["UG_MAX"],
                       "UG_MIN": ELEC_DA["UG_MIN"],
                       "UG_PRICE": ELEC_RT["UG_PRICE"][:, i],
                       "AC_PD": ELEC_RT["AC_PD"][:, i],
                       "DC_PD": ELEC_RT["DC_PD"][:, i],
                       "PV_PG": ELEC_RT["PV_PG"][:, i], }
            model_second_stage[i] = energy_hub_management.problem_formulation(ELEC=elec[i], CCHP=CCHP, THERMAL=THERMAL,
                                                                              BIC=BIC, ESS=ESS,
                                                                              HVAC=HVAC, BOIL=BOIL, CHIL=CHIL, T=T)

            lb_second_stage[i * nx:(i + 1) * nx] = model_second_stage[i]["lb"]
            Aeq_second_stage[i * neq:(i + 1) * neq, i * nx:(i + 1) * nx] = model_second_stage[i]["Aeq"]
            beq_second_stage[i * neq:(i + 1) * neq] = model_second_stage[i]["beq"]
            lb_second_stage[i * nx:(i + 1) * nx] = model_second_stage[i]["lb"]
            ub_second_stage[i * nx:(i + 1) * nx] = model_second_stage[i]["ub"]
            c_second_stage[i * nx:(i + 1) * nx] = model_second_stage[i]["c"] / N

        lb_first_stage = zeros((T, 1))
        ub_first_stage = zeros((T, 1))
        c_first_stage = zeros((T, 1))
        Aeq_first_stage = zeros((neq * N, T))
        A_first_stage = zeros((2 * T * N, T))
        A_second_stage = zeros((2 * T * N, nx * N))
        b_second_stage = zeros((2 * T * N, 1))
        b_second_stage[0:T * N] = ELEC_DA["UG_MAX"]
        b_second_stage[T * N:2 * T * N] = -ELEC_DA["UG_MIN"]

        for i in range(N):
            A_first_stage[i * T:(i + 1) * T, :] = eye(T)  # The upper limit
            A_first_stage[N * T + i * T:N * T + (i + 1) * T, :] = -eye(T)  # The lower limit
            for j in range(T):
                A_second_stage[i * T + j, i * nx + j * model["nx"] + model["pug"]] = 1  # The upper limit
                A_second_stage[N * T + i * T + j, i * nx + j * model["nx"] + model["pug"]] = -1  # The lower limit

        for i in range(T):
            lb_first_stage[i] = ELEC_DA["UG_MIN"]
            ub_first_stage[i] = ELEC_DA["UG_MAX"]
            c_first_stage[i] = ELEC_DA["UG_PRICE"][i]

        for i in range(N):
            Aeq_first_stage[i * neq + model["ac_eq"][0]:i * neq + model["ac_eq"][1], 0:T] = eye(T)

        model["Aeq"] = hstack([Aeq_first_stage, Aeq_second_stage])
        model["beq"] = beq_second_stage
        if model["A"] is None:
            model["A"] = hstack([A_first_stage, A_second_stage])
            model["b"] = b_second_stage
        else:
            model["A"] = vstack([model["A"], hstack([A_first_stage, A_second_stage])])
            model["b"] = vstack([model["A"], b_second_stage])

        model["lb"] = vstack([lb_first_stage, lb_second_stage])
        model["ub"] = vstack([ub_first_stage, ub_second_stage])
        model["c"] = vstack([c_first_stage, c_second_stage])
        # Test model for the boundary information
        n = N - 1
        nslack = 2 * T
        nx_first_stage = T + nslack
        neq_extended = neq + nslack

        # The decision of the first stage optimization
        c_first_stage = vstack([ELEC_DA["UG_PRICE"], zeros((nslack, 1)), model_second_stage[0]["c"]])
        lb_first_stage = vstack([ones((T, 1)) * ELEC_DA["UG_MIN"], zeros((nslack, 1)), model_second_stage[0]["lb"]])
        ub_first_stage = vstack(
            [ones((T, 1)) * ELEC_DA["UG_MAX"], inf * ones((nslack, 1)), model_second_stage[0]["ub"]])

        nx_extended_first_stage = nx_first_stage + nx

        Aeq = zeros((neq_extended, nx_extended_first_stage))

        Aeq[model["ac_eq"][0]:model["ac_eq"][1], 0:T] = eye(T)
        Aeq[0:neq, nx_first_stage:nx_extended_first_stage] = model_second_stage[0]["Aeq"]
        Aeq[neq:neq + T, 0:T] = eye(T)
        Aeq[neq:neq + T, T:2 * T] = eye(T)
        Aeq[neq + T:neq + 2 * T, 0:T] = -eye(T)
        Aeq[neq + T:neq + 2 * T, 2 * T:3 * T] = eye(T)
        beq = vstack(
            [model_second_stage[0]["beq"], ones((T, 1)) * ELEC_DA["UG_MAX"], -ones((T, 1)) * ELEC_DA["UG_MIN"]])

        Aeq_first_stage = zeros((neq_extended * n, nx_extended_first_stage))
        Aeq_second_stage = zeros((neq_extended * n, nx * n))
        beq_second_stage = zeros((neq_extended * n, 1))
        for i in range(n):
            Aeq_first_stage[i * neq_extended + model["ac_eq"][0]:i * neq_extended + model["ac_eq"][1], 0:T] = eye(T)

            Aeq_first_stage[i * neq_extended + neq:i * neq_extended + neq + T, 0:T] = eye(T)

            Aeq_first_stage[i * neq_extended + neq:i * neq_extended + neq + T, T:2 * T] = eye(T)

            Aeq_first_stage[i * neq_extended + neq + T:i * neq_extended + neq + 2 * T, 0:T] = -eye(T)
            Aeq_first_stage[i * neq_extended + neq + T:i * neq_extended + neq + 2 * T, 2 * T:3 * T] = eye(T)

            Aeq_second_stage[i * neq_extended:i * neq_extended + neq, i * nx:(i + 1) * nx] = model_second_stage[i + 1][
                "Aeq"]

            for j in range(T):
                Aeq_second_stage[
                    i * neq_extended + neq + j, i * nx + j * model["nx"] + model["pug"]] = 1  # The upper limit
                Aeq_second_stage[
                    i * neq_extended + neq + T + j, i * nx + j * model["nx"] + model["pug"]] = -1  # The lower limit

            beq_second_stage[i * neq_extended:i * neq_extended + neq] = model_second_stage[i + 1]["beq"]
            beq_second_stage[i * neq_extended + neq:i * neq_extended + neq + T] = ELEC_DA["UG_MAX"]
            beq_second_stage[i * neq_extended + neq + T:i * neq_extended + neq + 2 * T] = -ELEC_DA["UG_MIN"]

        model_test = {}
        model_test["Aeq"] = hstack([Aeq_first_stage, Aeq_second_stage])
        Aeq_temp = zeros((neq_extended, model_test["Aeq"].shape[1]))
        Aeq_temp[:, 0:nx_extended_first_stage] = Aeq
        model_test["Aeq"] = vstack([model_test["Aeq"], Aeq_temp])
        model_test["beq"] = vstack([beq_second_stage, beq])
        model_test["lb"] = vstack([lb_first_stage, lb_second_stage[nx::]])
        model_test["ub"] = vstack([ub_first_stage, ub_second_stage[nx::]])
        model_test["c"] = vstack([c_first_stage, c_second_stage[nx::]])
        #
        # # Reformulating the second stage decision making
        model_test_second_stage = {}

        neq_extended_second_stage = neq_extended + nx  # The extended second stage decision
        nx_extended_second_stage = nx + nx

        Aeq_first_stage = zeros((neq_extended_second_stage * n, nx_extended_first_stage))
        Aeq_second_stage = zeros((neq_extended_second_stage * n, nx_extended_second_stage * n))
        beq_second_stage = zeros((neq_extended_second_stage * n, 1))
        c_second_stage = zeros((nx_extended_second_stage * n, 1))
        f0 = zeros((n, 1))

        for i in range(n):
            Aeq_first_stage[i * neq_extended_second_stage:i * neq_extended_second_stage + neq_extended,
            0:nx_extended_first_stage] = model_test["Aeq"][i * neq_extended:(i + 1) * neq_extended,
                                         0:nx_extended_first_stage]

            Aeq_second_stage[i * neq_extended_second_stage:i * neq_extended_second_stage + neq_extended,
            i * nx_extended_second_stage:i * nx_extended_second_stage + nx] = model_test["Aeq"][
                                                                              i * neq_extended:(i + 1) * neq_extended,
                                                                              nx_extended_first_stage + i * nx:nx_extended_first_stage + (
                                                                                      i + 1) * nx]
            beq_second_stage[i * neq_extended_second_stage:i * neq_extended_second_stage + neq_extended] = \
                model_test["beq"][i * neq_extended:(i + 1) * neq_extended] - \
                Aeq_second_stage[i * neq_extended_second_stage:i * neq_extended_second_stage + neq_extended,
                i * nx_extended_second_stage:i * nx_extended_second_stage + nx].dot(
                    model_test["lb"][nx_extended_first_stage + i * nx:nx_extended_first_stage + (i + 1) * nx])

            Aeq_second_stage[i * neq_extended_second_stage + neq_extended:(i + 1) * neq_extended_second_stage,
            i * nx_extended_second_stage:i * nx_extended_second_stage + nx] = eye(nx)

            Aeq_second_stage[i * neq_extended_second_stage + neq_extended:(i + 1) * neq_extended_second_stage,
            i * nx_extended_second_stage + nx:(i + 1) * nx_extended_second_stage] = eye(nx)

            beq_second_stage[i * neq_extended_second_stage + neq_extended:(i + 1) * neq_extended_second_stage] = \
                model_test["ub"][nx_extended_first_stage + i * nx:nx_extended_first_stage + (i + 1) * nx] - \
                model_test["lb"][nx_extended_first_stage + i * nx:nx_extended_first_stage + (i + 1) * nx]

            c_second_stage[i * nx_extended_second_stage: i * nx_extended_second_stage + nx] = model_test["c"][
                                                                                              nx_extended_first_stage + i * nx:nx_extended_first_stage + (
                                                                                                      i + 1) * nx]

            f0[i] = transpose(
                model_test["c"][nx_extended_first_stage + i * nx:nx_extended_first_stage + (i + 1) * nx]).dot(
                model_test["lb"][nx_extended_first_stage + i * nx:nx_extended_first_stage + (i + 1) * nx])

        model_test_second_stage["Aeq"] = hstack([Aeq_first_stage, Aeq_second_stage])

        Aeq_temp = zeros((neq_extended, model_test_second_stage["Aeq"].shape[1]))
        Aeq_temp[:, 0:nx_extended_first_stage] = Aeq
        model_test_second_stage["Aeq"] = vstack([model_test_second_stage["Aeq"], Aeq_temp])
        model_test_second_stage["beq"] = vstack([beq_second_stage, beq])

        model_test_second_stage["lb"] = vstack([lb_first_stage, zeros((nx_extended_second_stage * n, 1))])
        model_test_second_stage["ub"] = vstack([ub_first_stage, inf * ones((nx_extended_second_stage * n, 1))])
        model_test_second_stage["c"] = vstack([c_first_stage, c_second_stage])
        model_test_second_stage["c0"] = f0
        # # Formulate the benders decomposition
        # # Reformulate the second stage optimization problems to the standard format
        # # Using the following transfering y = x-lb
        # # 1）Reformulate the first stage problem
        # # The coupling contraints between the first stage and second stage decision
        # # Two parts, the AC power balance equations
        hs = [0] * n
        Ts = [0] * n
        Ws = [0] * n
        ps = ones((n, 1))
        qs = [0] * n
        for i in range(n):
            # 1) The AC power balance equation
            hs[i] = model_test_second_stage["beq"][i * neq_extended_second_stage:(i + 1) * neq_extended_second_stage]

            Ts[i] = model_test_second_stage["Aeq"][i * neq_extended_second_stage:(i + 1) * neq_extended_second_stage,
                    0:nx_extended_first_stage]

            Ws[i] = model_test_second_stage["Aeq"][i * neq_extended_second_stage:(i + 1) * neq_extended_second_stage,
                    nx_extended_first_stage + i * nx_extended_second_stage:nx_extended_first_stage + (
                                i + 1) * nx_extended_second_stage]

            qs[i] = model_test_second_stage["c"][
                    nx_extended_first_stage + i * nx_extended_second_stage:nx_extended_first_stage + (
                                i + 1) * nx_extended_second_stage]

        model_decomposition = {"c": c_first_stage,
                               "lb": lb_first_stage,
                               "ub": ub_first_stage,
                               "A": None,
                               "b": None,
                               "Aeq": Aeq,
                               "beq": beq,
                               "ps": ps,
                               "qs": qs,
                               "Ts": Ts,
                               "hs": hs,
                               "Ws": Ws,
                               }

        return model, model_test_second_stage,model_decomposition
    def problem_solving(self, model):
        """
        Problem solving for the two-stage bidding strategy
        :param model:
        :return:
        """
        from energy_hub.bidding_strategy.data_format import PUG, PCHP, PAC2DC, PDC2AC, PIAC, EESS, PESS_CH, PESS_DC, \
            PPV, PCS, QCHP, QGAS, EHSS, QHS_DC, QHS_CH, QAC, QTD, QCE, QIAC, ECSS, QCS_DC, QCS_CH, QCD, VCHP, VGAS, NX

        (x, objvalue, status) = lp(model["c"], A=model["A"], b=model["b"], Aeq=model["Aeq"], beq=model["beq"],
                                   xmin=model["lb"], xmax=model["ub"])

        # Try to solve the linear programing problem
        T = self.T
        N = self.N
        nx = self.nx
        nx_da = T
        # decouple the solution
        pug = zeros((T, N))
        pchp = zeros((T, N))
        pac2dc = zeros((T, N))
        pdc2ac = zeros((T, N))
        piac = zeros((T, N))
        eess = zeros((T, N))
        pess_ch = zeros((T, N))
        pess_dc = zeros((T, N))
        ppv = zeros((T, N))
        qchp = zeros((T, N))
        qgas = zeros((T, N))
        etss = zeros((T, N))
        qes_dc = zeros((T, N))
        qes_ch = zeros((T, N))
        qac = zeros((T, N))
        qtd = zeros((T, N))
        qce = zeros((T, N))
        qiac = zeros((T, N))
        ecss = zeros((T, N))
        qcs_dc = zeros((T, N))
        qcs_ch = zeros((T, N))
        qcd = zeros((T, N))
        vchp = zeros((T, N))
        vgas = zeros((T, N))
        for j in range(N):
            for i in range(T):
                pug[i, j] = x[nx_da + j * nx + i * NX + PUG]
                pchp[i, j] = x[nx_da + j * nx + i * NX + PCHP]
                pac2dc[i, j] = x[nx_da + j * nx + i * NX + PAC2DC]
                pdc2ac[i, j] = x[nx_da + j * nx + i * NX + PDC2AC]
                piac[i, j] = x[nx_da + j * nx + i * NX + PIAC]
                eess[i, j] = x[nx_da + j * nx + i * NX + EESS]
                pess_ch[i, j] = x[nx_da + j * nx + i * NX + PESS_CH]
                pess_dc[i, j] = x[nx_da + j * nx + i * NX + PESS_DC]
                ppv[i, j] = x[nx_da + j * nx + i * NX + PPV]
                qchp[i, j] = x[nx_da + j * nx + i * NX + QCHP]
                qgas[i, j] = x[nx_da + j * nx + i * NX + QGAS]
                etss[i, j] = x[nx_da + j * nx + i * NX + EHSS]
                qes_dc[i, j] = x[nx_da + j * nx + i * NX + QHS_DC]
                qes_ch[i, j] = x[nx_da + j * nx + i * NX + QHS_CH]
                qac[i, j] = x[nx_da + j * nx + i * NX + QAC]
                qtd[i, j] = x[nx_da + j * nx + i * NX + QTD]
                qce[i, j] = x[nx_da + j * nx + i * NX + QCE]
                qiac[i, j] = x[nx_da + j * nx + i * NX + QIAC]
                ecss[i, j] = x[nx_da + j * nx + i * NX + ECSS]
                qcs_dc[i, j] = x[nx_da + j * nx + i * NX + QCS_DC]
                qcs_ch[i, j] = x[nx_da + j * nx + i * NX + QCS_CH]
                qcd[i, j] = x[nx_da + j * nx + i * NX + QCD]
                vchp[i, j] = x[nx_da + j * nx + i * NX + VCHP]
                vgas[i, j] = x[nx_da + j * nx + i * NX + VGAS]

        sol = {"obj": objvalue,
               "PUG_DA": x[0:T],
               "PUG": pug,
               "PCHP": pchp,
               "PAC2DC": pac2dc,
               "PDC2AC": pdc2ac,
               "PIAC": piac,
               "EESS": eess,
               "PESS_CH": pess_ch,
               "PESS_DC": pess_dc,
               "PPV": ppv,
               "QCHP": qchp,
               "QGAS": qgas,
               "ETSS": etss,
               "QES_CH": qes_ch,
               "QES_DC": qes_dc,
               "QAC": qac,
               "QTD": qtd,
               "QCE": qce,
               "QIAC": qiac,
               "ECSS": ecss,
               "QCS_CH": qcs_ch,
               "QCS_DC": qcs_dc,
               "QCD": qcd,
               "VCHP": vchp,
               "VGAS": vgas,
               }
        return sol

    def solution_check(self, sol):
        # Check the relaxations
        T = self.T
        N = self.N
        bic_relaxation = zeros((N, T))
        ess_relaxation = zeros((N, T))
        tes_relaxation = zeros((N, T))
        ces_relaxation = zeros((N, T))
        for i in range(N):
            bic_relaxation[:, i] = np.multiply(sol["PAC2DC"][i, :], sol["PDC2AC"][i, :])
            ess_relaxation[:, i] = np.multiply(sol["PESS_DC"][i, :], sol["PESS_CH"][i, :])
            tes_relaxation[:, i] = np.multiply(sol["QES_CH"][i, :], sol["QES_DC"][i, :])
            ces_relaxation[:, i] = np.multiply(sol["QCS_CH"][i, :], sol["QCS_DC"][i, :])

        sol_check = {"bic": bic_relaxation,
                     "ess": ess_relaxation,
                     "tes": tes_relaxation,
                     "ces": ces_relaxation}

        return sol_check


if __name__ == "__main__":
    # A test system
    # 1) System level configuration
    T = 24
    Delta_t = 1
    delat_t = 1
    T_second_stage = int(T / delat_t)
    N_sample = 2
    forecasting_errors_ac = 0.03
    forecasting_errors_dc = 0.03
    forecasting_errors_pv = 0.05
    forecasting_errors_prices = 0.03

    # For the HVAC system
    # 2) Thermal system configuration
    QHVAC_max = 100
    eff_HVAC = 4
    c_air = 1.85
    r_t = 1.3
    ambinent_temprature = array(
        [27, 27, 26, 26, 26, 26, 26, 25, 27, 28, 30, 31, 32, 32, 32, 32, 32, 32, 31, 30, 29, 28, 28, 27])
    temprature_in_min = 20
    temprature_in_max = 24

    CD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])
    HD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])

    # 3) Electricity system configuration
    PUG_MAX = 200
    PV_CAP = 50
    AC_PD_cap = 50
    DC_PD_cap = 50
    HD_cap = 100
    CD_cap = 100

    PESS_CH_MAX = 100
    PESS_DC_MAX = 100
    EFF_DC = 0.9
    EFF_CH = 0.9
    E0 = 50
    Emax = 100
    Emin = 20

    BIC_CAP = 100
    eff_BIC = 0.95

    electricity_price = array(
        [6.01, 73.91, 71.31, 69.24, 68.94, 70.56, 75.16, 73.19, 79.70, 85.76, 86.90, 88.60, 90.62, 91.26, 93.70, 90.94,
         91.26, 80.39, 76.25, 76.80, 81.22, 83.75, 76.16, 72.69])

    AC_PD = array([323.0284, 308.2374, 318.1886, 307.9809, 331.2170, 368.6539, 702.0040, 577.7045, 1180.4547, 1227.6240,
                   1282.9344, 1311.9738, 1268.9502, 1321.7436, 1323.9218, 1327.1464, 1386.9117, 1321.6387, 1132.0476,
                   1109.2701, 882.5698, 832.4520, 349.3568, 299.9920])
    DC_PD = array([287.7698, 287.7698, 287.7698, 287.7698, 299.9920, 349.3582, 774.4047, 664.0625, 1132.6996, 1107.7366,
                   1069.6837, 1068.9819, 1027.3295, 1096.3820, 1109.4778, 1110.7039, 1160.1270, 1078.7839, 852.2514,
                   791.5814, 575.4085, 551.1441, 349.3568, 299.992])
    PV_PG = array(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])

    ELEC_PRICE = electricity_price / 300
    ELEC_PRICE = ELEC_PRICE.reshape(T, 1)
    Eess_cost = 0.01

    PV_PG = PV_PG * PV_CAP
    # Modify the first stage profiles
    AC_PD = (AC_PD / max(AC_PD)) * AC_PD_cap
    DC_PD = (DC_PD / max(DC_PD)) * DC_PD_cap
    HD = (HD / max(HD)) * HD_cap
    CD = (CD / max(CD)) * CD_cap

    # Generate the second stage profiles using spline of scipy
    AC_PD_second_stage = zeros((T_second_stage, N_sample))
    DC_PD_second_stage = zeros((T_second_stage, N_sample))
    PV_second_stage = zeros((T_second_stage, N_sample))
    ELEC_PRICE_second_stage = zeros((T_second_stage, N_sample))

    for i in range(N_sample):
        AC_PD_second_stage[:, i] = ones((1, T_second_stage)) + np.random.normal(0, forecasting_errors_ac,
                                                                                T_second_stage)
        DC_PD_second_stage[:, i] = ones((1, T_second_stage)) + np.random.normal(0, forecasting_errors_dc,
                                                                                T_second_stage)
        PV_second_stage[:, i] = ones((1, T_second_stage)) + np.random.normal(0, forecasting_errors_pv, T_second_stage)

        ELEC_PRICE_second_stage[:, i] = ones((1, T_second_stage)) + np.random.normal(0, forecasting_errors_prices,
                                                                                     T_second_stage)

    for i in range(N_sample):
        AC_PD_second_stage[:, i] = np.multiply(AC_PD, AC_PD_second_stage[:, i])
        DC_PD_second_stage[:, i] = np.multiply(DC_PD, DC_PD_second_stage[:, i])
        PV_second_stage[:, i] = np.multiply(PV_PG, PV_second_stage[:, i])
        ELEC_PRICE_second_stage[:, i] = np.multiply(transpose(ELEC_PRICE), ELEC_PRICE_second_stage[:, i])

        # Chech the boundary information
        for j in range(T_second_stage):
            if AC_PD_second_stage[j, i] < 0:
                AC_PD_second_stage[j, i] = 0
            if DC_PD_second_stage[j, i] < 0:
                DC_PD_second_stage[j, i] = 0
            if PV_second_stage[j, i] < 0:
                PV_second_stage[j, i] = 0
            if ELEC_PRICE_second_stage[j, i] < 0:
                ELEC_PRICE_second_stage[j, i] = 0

    # CCHP system
    Gas_price = 0.1892
    Gmax = 200
    eff_CHP_e = 0.3
    eff_CHP_h = 0.4
    # Boiler information
    Boil_max = 100
    eff_boil = 0.9
    # Chiller information
    Chiller_max = 100
    eff_chiller = 1.2

    CCHP = {"MAX": Gmax,
            "EFF_E": eff_CHP_e,
            "EFF_C": eff_CHP_h,
            "EFF_H": eff_CHP_h,
            "COST": Gas_price}

    HVAC = {"CAP": QHVAC_max,
            "EFF": eff_HVAC,
            "C_AIR": c_air,
            "R_T": r_t,
            "TEMPERATURE": ambinent_temprature,
            "TEMP_MIN": temprature_in_min,
            "TEMP_MAX": temprature_in_max}

    THERMAL = {"HD": HD,
               "CD": CD, }

    ELEC = {"UG_MAX": PUG_MAX,
            "UG_MIN": -PUG_MAX,
            "UG_PRICE": ELEC_PRICE,
            "AC_PD": AC_PD,
            "DC_PD": DC_PD,
            "PV_PG": PV_PG
            }

    # The second stage scenarios
    ELEC_second_stage = {"AC_PD": AC_PD_second_stage,
                         "DC_PD": DC_PD_second_stage,
                         "PV_PG": PV_second_stage,
                         "UG_PRICE": ELEC_PRICE_second_stage, }

    BIC = {"CAP": BIC_CAP,
           "EFF": eff_BIC,
           }

    BESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "PC_MAX": PESS_CH_MAX,
            "PD_MAX": PESS_DC_MAX,
            "EFF_CH": EFF_CH,
            "EFF_DC": EFF_DC,
            "COST": Eess_cost,
            }

    TESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "TC_MAX": PESS_CH_MAX,
            "TD_MAX": PESS_DC_MAX,
            "EFF_CH": EFF_CH,
            "EFF_DC": EFF_DC,
            "EFF_SD": 0.98,  # The self discharging
            "COST": Eess_cost,
            }

    CESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "TC_MAX": PESS_CH_MAX,
            "TD_MAX": PESS_DC_MAX,
            "EFF_CH": EFF_CH,
            "EFF_DC": EFF_DC,
            "EFF_SD": 0.98,  # The self discharging
            "COST": Eess_cost,
            "PMAX": PESS_CH_MAX * 10,
            "ICE": 3.5,
            }

    ESS = {"BESS": BESS,
           "TESS": TESS,
           "CESS": CESS}

    BOIL = {"CAP": Boil_max,
            "EFF": eff_boil}

    CHIL = {"CAP": Chiller_max,
            "EFF": eff_chiller}

    two_stage_bidding = TwoStageBidding()

    (model, model_compact,model_decomposed) = two_stage_bidding.problem_formualtion(ELEC_DA=ELEC, ELEC_RT=ELEC_second_stage, CCHP=CCHP,
                                                                   THERMAL=THERMAL,
                                                                   BIC=BIC, ESS=ESS,
                                                                   HVAC=HVAC, BOIL=BOIL, CHIL=CHIL,
                                                                   T=T, N=N_sample)

    #
    # sol_test = lp(c=model["c"], Aeq=model["Aeq"], beq=model["beq"], A=model["A"], b=model["b"], xmin=model["lb"],
    #               xmax=model["ub"])  # The solver test has been passed
    #
    # sol_compact = lp(c=model_compact["c"], Aeq=model_compact["Aeq"], beq=model_compact["beq"], xmin=model_compact["lb"],
    #                  xmax=model_compact["ub"])

    bender_decomposition = BendersDecomposition()

    sol_decomposed = bender_decomposition.main(c=model_decomposed["c"], A=model_decomposed["A"],
                                               b=model_decomposed["b"], Aeq=model_decomposed["Aeq"],
                                               beq=model_decomposed["beq"], lb=model_decomposed["lb"],
                                               ub=model_decomposed["ub"], ps=model_decomposed["ps"],
                                               qs=model_decomposed["qs"],
                                               Ws=model_decomposed["Ws"], Ts=model_decomposed["Ts"],
                                               hs=model_decomposed["hs"])

    sol = two_stage_bidding.problem_solving(model)  # This problem is feasible and optimal
    # obj = sol_decomposed["objvalue"][0] + model_decomposed["qs0"]
    sol_check = two_stage_bidding.solution_check(sol)

    print(sol)
