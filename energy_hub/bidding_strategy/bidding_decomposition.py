"""
Decomposition based bidding strategy
"""


# sys.path.append('/home/matrix/PycharmProjects/Optimization/energy_hub')

from numpy import zeros, ones, array, eye, hstack, vstack, inf, transpose
import numpy as np
from solvers.benders_decomposition import BendersDecomposition
from energy_hub.bidding_strategy.bidding_standard_models import EnergyHubManagement
# from energy_hub.bidding_strategy.bidding_strategy import EnergyHubManagement  # import the energy hub management class

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
        nx = T  # The demision of first stage decision making
        Cs = vstack([eye(nx), -eye(nx)])
        Es = eye(nx)

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

        del elec
        # Formulate the standard second stage problem
        hs = [0] * N
        Ts = [0] * N
        Ws = [0] * N
        ps = ones((N, 1)) / N
        qs = [0] * N
        qs0 = zeros((N, 1))

        for i in range(N):
            # Update the decision making problem
            ny = model_second_stage[i]["nx"] + model_second_stage[i]["nh"] + model_second_stage[i]["nd"] + \
                 model_second_stage[i]["nx"]  # Including the auxiliary variables
            # Update the objective functions
            qs[i] = zeros((ny, 1))
            qs[i][0:model_second_stage[i]["nx"]] = model_second_stage[i]["q"]
            # Update the constraint sets
            neq = model_second_stage[i]["nh"] + model_second_stage[i]["ng"] + model_second_stage[i]["nd"] + \
                  model_second_stage[i]["nf"] + model_second_stage[i]["nx"]
            hs[i] = zeros((neq, 1))
            Ts[i] = zeros((neq, nx))
            Ws[i] = zeros((neq, ny))
            # 1) Cs*x + Ds*y + slack(nh) = hs - Ds*lbs
            if model_second_stage[i]["nh"] != 0:
                hs[i][0:model_second_stage[i]["nh"]] = model_second_stage[i]["h"] - model_second_stage[i]["D"].dot(
                    model_second_stage[i]["lb"])
                Ts[i][0:model_second_stage[i]["nh"], :] = Cs
                Ws[i][0:model_second_stage[i]["nh"], 0:model_second_stage[i]["nx"]] = model_second_stage[i]["D"]
                Ws[i][0:model_second_stage[i]["nh"],
                model_second_stage[i]["nx"]:model_second_stage[i]["nx"] + model_second_stage[i]["nh"]] = \
                    eye(model_second_stage[i]["nh"])
            # 2) Es*x + Fs*y = gs - Fs*lbs
            if model_second_stage[i]["ng"] != 0:
                hs[i][model_second_stage[i]["nh"]:model_second_stage[i]["nh"] + model_second_stage[i]["ng"]] = \
                    model_second_stage[i]["g"] - model_second_stage[i]["F"].dot(model_second_stage[i]["lb"])
                Ts[i][model_second_stage[i]["nh"]:model_second_stage[i]["nh"] + model_second_stage[i]["ng"], :] = Es
                Ws[i][model_second_stage[i]["nh"]:model_second_stage[i]["nh"] + model_second_stage[i]["ng"],
                0:model_second_stage[i]["nx"]] = model_second_stage[i]["F"]
            # 3) Bs *y + slack(nd) = ds - Bs * lbs
            if model_second_stage[i]["nd"] != 0:
                hs[i][model_second_stage[i]["nh"] + model_second_stage[i]["ng"]:model_second_stage[i]["nh"] +
                                                                                model_second_stage[i]["ng"] +
                                                                                model_second_stage[i]["nd"]] = \
                    model_second_stage[i]["d"] - model_second_stage[i]["B"].dot(model_second_stage[i]["lb"])

                Ws[i][model_second_stage[i]["nh"] + model_second_stage[i]["ng"]:model_second_stage[i]["nh"] +
                                                                                model_second_stage[i]["ng"] +
                                                                                model_second_stage[i]["nd"],
                0:model_second_stage[i]["nx"]] = model_second_stage[i]["B"]

                Ws[i][model_second_stage[i]["nh"] + model_second_stage[i]["ng"]:model_second_stage[i]["nh"] +
                                                                                model_second_stage[i]["ng"] +
                                                                                model_second_stage[i]["nd"],
                model_second_stage[i]["nx"] + model_second_stage[i]["nh"]:model_second_stage[i]["nx"] +
                                                                          model_second_stage[i]["nh"] +
                                                                          model_second_stage[i]["nd"]] = eye(
                    model_second_stage[i]["nd"])
            # 4) Beq * y = fs
            hs[i][model_second_stage[i]["nh"] + model_second_stage[i]["ng"] + model_second_stage[i]["nd"]:
                  model_second_stage[i]["nh"] +
                  model_second_stage[i]["ng"] +
                  model_second_stage[i]["nd"] + model_second_stage[i]["nf"]] = \
                model_second_stage[i]["f"] - model_second_stage[i]["Beq"].dot(model_second_stage[i]["lb"])

            Ws[i][model_second_stage[i]["nh"] + model_second_stage[i]["ng"] + model_second_stage[i]["nd"]:
                  model_second_stage[i]["nh"] +
                  model_second_stage[i]["ng"] +
                  model_second_stage[i]["nd"] + model_second_stage[i]["nf"],
            0:model_second_stage[i]["nx"]] = model_second_stage[i]["Beq"]
            # 5) y + slack(nx) = ubs-lbs
            hs[i][model_second_stage[i]["nh"] + model_second_stage[i]["ng"] + model_second_stage[i]["nd"] +
                  model_second_stage[i]["nf"]:
                  model_second_stage[i]["nh"] +
                  model_second_stage[i]["ng"] +
                  model_second_stage[i]["nd"] + model_second_stage[i]["nf"] + model_second_stage[i]["nx"]] = \
                model_second_stage[i]["ub"] - model_second_stage[i]["lb"]

            Ws[i][model_second_stage[i]["nh"] + model_second_stage[i]["ng"] + model_second_stage[i]["nd"] +
                  model_second_stage[i]["nf"]:
                  model_second_stage[i]["nh"] +
                  model_second_stage[i]["ng"] +
                  model_second_stage[i]["nd"] + model_second_stage[i]["nf"] + model_second_stage[i]["nx"],
            0:model_second_stage[i]["nx"]] = eye(model_second_stage[i]["nx"])

            Ws[i][model_second_stage[i]["nh"] + model_second_stage[i]["ng"] + model_second_stage[i]["nd"] +
                  model_second_stage[i]["nf"]:
                  model_second_stage[i]["nh"] +
                  model_second_stage[i]["ng"] +
                  model_second_stage[i]["nd"] + model_second_stage[i]["nf"] + model_second_stage[i]["nx"],
            model_second_stage[i]["nx"] + model_second_stage[i]["nh"] + model_second_stage[i]["nd"]:] = eye(
                model_second_stage[i]["nx"])

            qs0[i] = transpose(model_second_stage[i]["q"]).dot(model_second_stage[i]["lb"])

        model_decomposed = {"c": ELEC_DA["UG_PRICE"],
                            "lb": ELEC_DA["UG_MIN"] * ones((T, 1)),
                            "ub": ELEC_DA["UG_MAX"] * ones((T, 1)),
                            "A": None,
                            "b": None,
                            "Aeq": None,
                            "beq": None,
                            "ps": ps,
                            "qs": qs,
                            "Ts": Ts,
                            "hs": hs,
                            "Ws": Ws,
                            "qs0": qs0
                            }

        return model, model_decomposed, model_second_stage


if __name__ == "__main__":
    # A test system
    # 1) System level configuration
    T = 24
    Delta_t = 1
    delat_t = 1
    T_second_stage = int(T / delat_t)
    N_sample = 10
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
        # Check the boundary information
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

    (model, model_decomposed, model_base) = two_stage_bidding.problem_formualtion(ELEC_DA=ELEC,
                                                                                  ELEC_RT=ELEC_second_stage,
                                                                                  CCHP=CCHP, THERMAL=THERMAL,
                                                                                  BIC=BIC, ESS=ESS,
                                                                                  HVAC=HVAC, BOIL=BOIL, CHIL=CHIL,
                                                                                  T=T, N=N_sample)
    bender_decomposition = BendersDecomposition()
    energy_hub_management = EnergyHubManagement()
    sol_decomposed = bender_decomposition.main(c=model_decomposed["c"], A=model_decomposed["A"],
                                               b=model_decomposed["b"], Aeq=model_decomposed["Aeq"],
                                               beq=model_decomposed["beq"], lb=model_decomposed["lb"],
                                               ub=model_decomposed["ub"], ps=model_decomposed["ps"],
                                               qs=model_decomposed["qs"],
                                               Ws=model_decomposed["Ws"], Ts=model_decomposed["Ts"],
                                               hs=model_decomposed["hs"])
    sol = [0] * N_sample
    sol_check = [0] * N_sample
    # Convert the solution back
    for i in range(N_sample):
        model_base[i]["x"] = model_base[i]["lb"] + array(
            sol_decomposed["x_second_stage"][i][0:model_base[i]["nx"]]).reshape(model_base[i]["nx"],
                                                                                1)  # Recover the second stage decision
        model_base[i]["T"] = T

    for i in range(N_sample):
        sol[i] = energy_hub_management.solution_update(model_base[i])
        sol_check[i] = energy_hub_management.solution_check(sol[i])

    print(sol_decomposed["objvalue"] + transpose(model_decomposed["ps"]).dot(model_decomposed["qs0"]))
