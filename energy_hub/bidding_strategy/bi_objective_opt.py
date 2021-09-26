"""
CVaR and excepted operational cost balancing for energy hubs
"""
from numpy import zeros, ones, array
import numpy as np
from gurobipy import *
from scipy import stats


def main(N_scenario_first_stage=100, N_scenario_second_stage=1000, alpha=0.99, Lam=0.1, mode=0, cvar=0, ec=0):
    # 1) System level configuration
    T = 24
    weight_first_stage = ones((N_scenario_first_stage, 1)) / N_scenario_first_stage
    Weight = stats.norm.pdf(stats.norm.isf(1 - alpha)) / (1 - alpha)
    forecasting_errors_ac = 0.03
    forecasting_errors_dc = 0.03
    forecasting_errors_pv = 0.10
    forecasting_errors_prices = 0.03
    forecasting_errors_temperature = 0.10
    bigM = 10 ** 2
    # For the HVAC system
    # 2) Thermal system configuration
    QHVAC_max = 200
    eff_HVAC = 4
    c_air = 0.185
    r_t = 0.13
    ambinent_temprature = array(
        [27, 27, 26, 26, 26, 26, 26, 25, 27, 28, 30, 31, 32, 32, 32, 32, 32, 32, 31, 30, 29, 28, 28, 27])
    temprature_in_min = 22
    temprature_in_max = 23

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

    electricity_price_DA = array(
        [6.01, 75.91, 73.31, 71.24, 70.94, 69.56, 74.16, 72.19, 80.70, 86.76, 85.90, 87.60, 91.62, 90.26, 95.70, 87.94,
         91.26, 82.39, 75.25, 76.80, 81.22, 83.75, 76.16, 72.69])

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
    ELEC_PRICE_DA = electricity_price_DA / 300

    Eess_cost = 0.01
    PV_cost = 0.012

    PV_PG = PV_PG * PV_CAP
    # Modify the first stage profiles
    AC_PD = (AC_PD / max(AC_PD)) * AC_PD_cap
    DC_PD = (DC_PD / max(DC_PD)) * DC_PD_cap
    HD = (HD / max(HD)) * HD_cap
    CD = (CD / max(CD)) * CD_cap

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

    # load the scenarios
    f = open("ac_pd_second_stage.txt", "r+")
    AC_PD_second_stage = np.loadtxt(f, delimiter=',')
    f.close()
    f = open("dc_pd_second_stage.txt", "r+")
    DC_PD_second_stage = np.loadtxt(f, delimiter=',')
    f.close()
    f = open("pv_second_stage.txt", "r+")
    PV_second_stage = np.loadtxt(f, delimiter=',')
    f.close()
    f = open("price_first_stage.txt", "r+")
    Price_DA = np.loadtxt(f, delimiter=',')
    f.close()
    f = open("temperature_second_stage.txt", "r+")
    Temperature_second_stage = np.loadtxt(f, delimiter=',')
    f.close()
    f = open("weight_second_stage.txt", "r+")
    weight_second_stage = np.loadtxt(f, delimiter=',')
    f.close()

    # Generate the order bidding curve, the constrain will be added from the highest order to the  lowest
    Order = zeros((T, N_scenario_first_stage))
    for i in range(T):
        Order[i, :] = np.argsort(Price_DA[i, :])

    N_scenario_second_stage = len(weight_second_stage)

    model = Model("EnergyHub")
    PDA = {}  # Day-ahead bidding strategy
    pRT = {}  # Real-time prices
    pRT_positive = {}  # Real-time prices
    pRT_negative = {}  # Real-time prices
    pCHP = {}  # Real-time output of CHP units
    pAC2DC = {}  # Real-time power transfered from AC to DC
    pDC2AC = {}  # Real-time power transfered from DC to AC
    eESS = {}  # Real-time energy status
    pESS_DC = {}  # ESS discharging rate
    pESS_CH = {}  # ESS charging rate
    pIAC = {}  # HVAC consumption
    pPV = {}  # PV consumption
    pCS = {}
    ## Group 2: Heating ##
    qCHP = {}
    qGAS = {}
    eHSS = {}
    qHS_DC = {}
    qHS_CH = {}
    qAC = {}
    qTD = {}
    ## Group 3: Cooling ##
    qCE = {}
    qIAC = {}
    eCSS = {}
    qCS_DC = {}  # The output is cooling
    qCS_CH = {}  # The input is electricity
    qCD = {}
    ## Group 4: Gas ##
    vCHP = {}
    vGAS = {}
    # Group 5: Temperature
    temprature_in = {}
    # Group 7: conditional variables
    nu_index = {}
    epison = model.addVar(name="EPISON")
    # Define the conditional variables
    for i in range(N_scenario_second_stage):
        nu_index[i] = model.addVar(lb=0, name="NU_INDEX{0}".format(i))

    # Define the day-ahead scheduling plan
    for i in range(T):
        for j in range(N_scenario_first_stage):
            PDA[i, j] = model.addVar(lb=-PUG_MAX, ub=PUG_MAX, name="PDA{0}".format(i * N_scenario_first_stage + j))

    # Define the real-time scheduling plan
    for i in range(T):  # Dispatch at each time slot
        for j in range(N_scenario_second_stage):  # Scheduling plan under second stage plan
            for k in range(N_scenario_first_stage):  # Dispatch at each time slot
                pRT[i, j, k] = model.addVar(lb=-PUG_MAX, ub=PUG_MAX,
                                            name="pRT{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pRT_positive[i, j, k] = model.addVar(lb=0, ub=PUG_MAX,
                                                     name="pRT_positive{0}".format(
                                                         i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pRT_negative[i, j, k] = model.addVar(lb=0, ub=PUG_MAX,
                                                     name="pRT_negative{0}".format(
                                                         i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pCHP[i, j, k] = model.addVar(lb=0, ub=CCHP["MAX"] * CCHP["EFF_E"],
                                             name="pCHP{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pAC2DC[i, j, k] = model.addVar(lb=0, ub=BIC["CAP"],
                                               name="pAC2DC{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pDC2AC[i, j, k] = model.addVar(lb=0, ub=BIC["CAP"],
                                               name="pDC2AC{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                eESS[i, j, k] = model.addVar(lb=ESS["BESS"]["E_MIN"], ub=ESS["BESS"]["E_MAX"],
                                             name="eESS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pESS_DC[i, j, k] = model.addVar(lb=0, ub=ESS["BESS"]["PD_MAX"],
                                                name="pESS_DC{0}".format(
                                                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pESS_CH[i, j, k] = model.addVar(lb=0, ub=ESS["BESS"]["PC_MAX"],
                                                name="pESS_CH{0}".format(
                                                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pIAC[i, j, k] = model.addVar(lb=0, ub=HVAC["CAP"],
                                             name="pIAC{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pPV[i, j, k] = model.addVar(lb=0, ub=PV_second_stage[i, j],
                                            name="pPV{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pCS[i, j, k] = model.addVar(lb=0, ub=ESS["CESS"]["PMAX"],
                                            name="pCS{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                ## Group 2: Heating ##
                qCHP[i, j, k] = model.addVar(lb=0, ub=CCHP["MAX"] * CCHP["EFF_H"],
                                             name="qCHP{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qGAS[i, j, k] = model.addVar(lb=0, ub=CCHP["MAX"] * CCHP["EFF_H"],
                                             name="qGAS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                eHSS[i, j, k] = model.addVar(lb=0, ub=ESS["TESS"]["E_MAX"],
                                             name="eHSS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qHS_DC[i, j, k] = model.addVar(lb=0, ub=ESS["TESS"]["TD_MAX"],
                                               name="qHS_DC{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qHS_CH[i, j, k] = model.addVar(lb=0, ub=ESS["TESS"]["TC_MAX"],
                                               name="qHS_CH{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qAC[i, j, k] = model.addVar(lb=0, ub=CHIL["CAP"],
                                            name="qAC{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qTD[i, j, k] = model.addVar(lb=0, ub=QHVAC_max,
                                            name="qTD{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                ## Group 3: Cooling ##
                qCE[i, j, k] = model.addVar(lb=0, ub=CHIL["CAP"],
                                            name="qCE{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qIAC[i, j, k] = model.addVar(lb=0, ub=HVAC["CAP"] * HVAC["EFF"],
                                             name="qIAC{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                eCSS[i, j, k] = model.addVar(lb=0, ub=ESS["CESS"]["E_MAX"],
                                             name="eCSS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qCS_DC[i, j, k] = model.addVar(lb=0, ub=ESS["CESS"]["TD_MAX"],
                                               name="qCS_DC{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))  # The output is cooling
                qCS_CH[i, j, k] = model.addVar(lb=0, ub=ESS["CESS"]["TC_MAX"],
                                               name="qCS_CH{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))  # The input is electricity
                qCD[i, j, k] = model.addVar(lb=0, ub=QHVAC_max,
                                            name="qCD{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                ## Group 4: Gas ##
                vCHP[i, j, k] = model.addVar(lb=0, ub=CCHP["MAX"],
                                             name="vCHP{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                vGAS[i, j, k] = model.addVar(lb=0, ub=BOIL["CAP"],
                                             name="vGAS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))

                temprature_in[i, j, k] = model.addVar(lb=temprature_in_min / 10, ub=temprature_in_max * 10,
                                                      name="temprature_in{0}".format(
                                                          i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))

                if i == T - 1:
                    eESS[i, j, k] = model.addVar(lb=ESS["BESS"]["E0"], ub=ESS["BESS"]["E0"],
                                                 name="eESS{0}".format(
                                                     i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                    eHSS[i, j, k] = model.addVar(lb=ESS["TESS"]["E0"], ub=ESS["TESS"]["E0"],
                                                 name="eHSS{0}".format(
                                                     i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                    eCSS[i, j, k] = model.addVar(lb=ESS["CESS"]["E0"], ub=ESS["CESS"]["E0"],
                                                 name="eCSS{0}".format(
                                                     i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))

    ## Formulate the constraints set
    # 1） Energy storage systems
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                if i != 0:
                    # Battery energy constraint
                    model.addConstr(
                        eESS[i, j, k] == eESS[i - 1, j, k] + pESS_CH[i, j, k] * ESS["BESS"]["EFF_CH"] - pESS_DC[
                            i, j, k] / ESS["BESS"]["EFF_DC"])
                    # Heat energy storage constraint
                    model.addConstr(
                        eHSS[i, j, k] == ESS["TESS"]["EFF_SD"] * eHSS[i - 1, j, k] + qHS_CH[i, j, k] * ESS["TESS"][
                            "EFF_CH"] - qHS_DC[i, j, k] / ESS["TESS"]["EFF_DC"])
                    # Cooling energy storage constraint
                    model.addConstr(
                        eCSS[i, j, k] == ESS["CESS"]["EFF_SD"] * eCSS[i - 1, j, k] + qCS_CH[i, j, k] * ESS["CESS"][
                            "EFF_CH"] - qCS_DC[i, j, k] / ESS["CESS"]["EFF_DC"])
                else:
                    model.addConstr(
                        eESS[i, j, k] == ESS["BESS"]["E0"] + pESS_CH[i, j, k] * ESS["BESS"]["EFF_CH"] - pESS_DC[
                            i, j, k] / ESS["BESS"]["EFF_DC"])
                    model.addConstr(
                        eHSS[i, j, k] == ESS["TESS"]["EFF_SD"] * ESS["TESS"]["E0"] + qHS_CH[i, j, k] * ESS["TESS"][
                            "EFF_CH"] - qHS_DC[i, j, k] / ESS["TESS"]["EFF_DC"])
                    # Cooling energy storage constraint
                    model.addConstr(
                        eCSS[i, j, k] == ESS["CESS"]["EFF_SD"] * ESS["CESS"]["E0"] + qCS_CH[i, j, k] * ESS["CESS"][
                            "EFF_CH"] - qCS_DC[i, j, k] / ESS["CESS"]["EFF_DC"])
    # 2） Energy conversion relationship
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                model.addConstr(pCHP[i, j, k] == vCHP[i, j, k] * CCHP["EFF_E"])
                model.addConstr(qCHP[i, j, k] == vCHP[i, j, k] * CCHP["EFF_H"])
                model.addConstr(qGAS[i, j, k] == vGAS[i, j, k] * BOIL["EFF"])
                model.addConstr(qCE[i, j, k] == qAC[i, j, k] * CHIL["EFF"])
                model.addConstr(qIAC[i, j, k] == pIAC[i, j, k] * HVAC["EFF"])
                model.addConstr(qCS_CH[i, j, k] == pCS[i, j, k] * ESS["CESS"]["ICE"])
    # 3) Energy balance equations
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                # AC bus power balance equation
                model.addConstr(
                    PDA[i, k] + pRT[i, j, k] + pCHP[i, j, k] - pAC2DC[i, j, k] + BIC["EFF"] * pDC2AC[i, j, k] ==
                    AC_PD_second_stage[i, j])

                # DC bus power balance equation
                model.addConstr(
                    BIC["EFF"] * pAC2DC[i, j, k] - pDC2AC[i, j, k] - pIAC[i, j, k] - pESS_CH[i, j, k] + pESS_DC[
                        i, j, k] + pPV[i, j, k] - pCS[i, j, k] ==
                    DC_PD_second_stage[i, j])

                # Heat energy balance
                model.addConstr(
                    qCHP[i, j, k] + qGAS[i, j, k] + qHS_DC[i, j, k] - qHS_CH[i, j, k] - qAC[i, j, k] - qTD[i, j, k] ==
                    HD[i])

                # Cooling energy balance
                model.addConstr(
                    qIAC[i, j, k] + qCE[i, j, k] + qCS_DC[i, j, k] - qCD[i, j, k] == CD[i])
    # 4) Constraints for the day-ahead and real-time energy trading
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                model.addConstr(pRT[i, j, k] + PDA[i, k] <= PUG_MAX)
                model.addConstr(pRT[i, j, k] + PDA[i, k] >= -PUG_MAX)

    # 5) Constraints for the bidding curves
    for i in range(T):
        Index = Order[i, :].tolist()
        for j in range(N_scenario_first_stage - 1):
            model.addConstr(PDA[i, Index[j]] >= PDA[i, Index[j + 1]])
    # 6) For the real-time dispatch
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                model.addConstr(pRT_positive[i, j, k] >= pRT[i, j, k])
                model.addConstr(pRT_negative[i, j, k] >= -pRT[i, j, k])
    # 7) For the indoor temperature control
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                if i != 0:
                    model.addConstr(
                        qTD[i, j, k] - qCD[i, j, k] == (temprature_in[i, j, k] - temprature_in[i - 1, j, k]) * c_air - (
                                Temperature_second_stage[i, j] - temprature_in[i, j, k]) / r_t)
                else:
                    model.addConstr(
                        qTD[i, j, k] - qCD[i, j, k] == (temprature_in[i, j, k] - ambinent_temprature[0]) * c_air - (
                                Temperature_second_stage[i, j] - temprature_in[i, j, k]) / r_t)
    # 8) Indoor temperature relaxation
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                model.addConstr(temprature_in[i, j, k] >= temprature_in_min)
                model.addConstr(temprature_in[i, j, k] <= temprature_in_max)

    ## Formulate the objective functions
    # The first stage objective value
    obj_DA = 0
    for i in range(T):
        for j in range(N_scenario_first_stage):
            obj_DA += PDA[i, j] * Price_DA[i, j] * weight_first_stage[j]

    obj_RT = 0
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                obj_RT += pRT[i, j, k] * ELEC_PRICE[i] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += pRT_positive[i, j, k] * ELEC_PRICE[i] * forecasting_errors_prices * Weight * \
                          weight_first_stage[k] * weight_second_stage[j]
                obj_RT += pRT_negative[i, j, k] * ELEC_PRICE[i] * forecasting_errors_prices * Weight * \
                          weight_first_stage[k] * weight_second_stage[j]
                obj_RT += pESS_DC[i, j, k] * ESS["BESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += pESS_CH[i, j, k] * ESS["BESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += qHS_DC[i, j, k] * ESS["TESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += qHS_CH[i, j, k] * ESS["TESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += qCS_DC[i, j, k] * ESS["CESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += qCS_CH[i, j, k] * ESS["CESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += vGAS[i, j, k] * CCHP["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += vCHP[i, j, k] * CCHP["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += pPV[i, j, k] * PV_cost * weight_first_stage[k] * weight_second_stage[j]

    for j in range(N_scenario_second_stage):
        expr = 0
        for k in range(N_scenario_first_stage):
            for i in range(T):
                expr += pRT[i, j, k] * ELEC_PRICE[i] * weight_first_stage[k]
                expr += pRT_positive[i, j, k] * ELEC_PRICE[i] * forecasting_errors_prices * Weight * weight_first_stage[
                    k]
                expr += pRT_negative[i, j, k] * ELEC_PRICE[i] * forecasting_errors_prices * Weight * weight_first_stage[
                    k]
                expr += pESS_DC[i, j, k] * ESS["BESS"]["COST"] * weight_first_stage[k]
                expr += pESS_CH[i, j, k] * ESS["BESS"]["COST"] * weight_first_stage[k]
                expr += qHS_DC[i, j, k] * ESS["TESS"]["COST"] * weight_first_stage[k]
                expr += qHS_CH[i, j, k] * ESS["TESS"]["COST"] * weight_first_stage[k]
                expr += qCS_DC[i, j, k] * ESS["CESS"]["COST"] * weight_first_stage[k]
                expr += qCS_CH[i, j, k] * ESS["CESS"]["COST"] * weight_first_stage[k]
                expr += vGAS[i, j, k] * CCHP["COST"] * weight_first_stage[k]
                expr += vCHP[i, j, k] * CCHP["COST"] * weight_first_stage[k]
                expr += pPV[i, j, k] * PV_cost * weight_first_stage[k]

        expr += obj_DA

        model.addConstr(lhs=expr - epison - nu_index[j], sense=GRB.LESS_EQUAL, rhs=0)

    obj_CVaR = epison
    for i in range(N_scenario_second_stage):
        obj_CVaR += nu_index[i] * weight_second_stage[i] / (1 - alpha)

    if mode == 1:
        model.addConstr(obj_DA + obj_RT == ec)
    elif mode == 2:
        model.addConstr(obj_CVaR == cvar)
    elif mode == 3:
        model.addConstr(obj_CVaR <= cvar)

    obj = (1 - Lam) * (obj_DA + obj_RT) + Lam * obj_CVaR

    model.setObjective(obj)

    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    model.Params.DisplayInterval = 1
    model.Params.LogFile = ""
    model.Params.MIPGap = 10 ** -3
    model.Params.timeLimit = 3 * 10 ** 2

    model.optimize()

    obj = obj.getValue()

    # Obtain the solutions
    # 1） Day ahead trading plan
    pDA = zeros((T, N_scenario_first_stage))
    for i in range(T):
        for j in range(N_scenario_first_stage):
            pDA[i, j] = model.getVarByName("PDA{0}".format(i * N_scenario_first_stage + j)).X

    PRT = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PRT_positive = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PRT_negative = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PCHP = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PAC2DC = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PDC2AC = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    EESS = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PESS_DC = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PESS_CH = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PIAC = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PPV = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    PCS = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    ## Group 2: Heating ##
    QCHP = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    QGAS = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    EHSS = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    QHS_DC = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    QHS_CH = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    QAC = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    QTD = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    ## Group 3: Cooling ##
    QCE = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    QIAC = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    ECSS = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    QCS_DC = zeros((T, N_scenario_second_stage, N_scenario_first_stage))  # The output is cooling
    QCS_CH = zeros((T, N_scenario_second_stage, N_scenario_first_stage))  # The input is electricity
    QCD = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    ## Group 4: Gas ##
    VCHP = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    VGAS = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    Temprature_in = zeros((T, N_scenario_second_stage, N_scenario_first_stage))
    for i in range(T):  # Dispatch at each time slot
        for j in range(N_scenario_second_stage):  # Scheduling plan under second stage plan
            for k in range(N_scenario_first_stage):  # Dispatch at each time slot
                PRT[i, j, k] = model.getVarByName("pRT{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PRT_positive[i, j, k] = model.getVarByName("pRT_positive{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PRT_negative[i, j, k] = model.getVarByName("pRT_negative{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PCHP[i, j, k] = model.getVarByName("pCHP{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PAC2DC[i, j, k] = model.getVarByName("pAC2DC{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PDC2AC[i, j, k] = model.getVarByName("pDC2AC{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                EESS[i, j, k] = model.getVarByName("eESS{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PESS_DC[i, j, k] = model.getVarByName("pESS_DC{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PESS_CH[i, j, k] = model.getVarByName("pESS_CH{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PIAC[i, j, k] = model.getVarByName("pIAC{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PPV[i, j, k] = model.getVarByName("pPV{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                PCS[i, j, k] = model.getVarByName("pCS{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                ## Group 2: Heating ##
                QCHP[i, j, k] = model.getVarByName("qCHP{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                QGAS[i, j, k] = model.getVarByName("qGAS{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                EHSS[i, j, k] = model.getVarByName("eHSS{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                QHS_DC[i, j, k] = model.getVarByName("qHS_DC{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                QHS_CH[i, j, k] = model.getVarByName("qHS_CH{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                QAC[i, j, k] = model.getVarByName("qAC{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                QTD[i, j, k] = model.getVarByName("qTD{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                ## Group 3: Cooling ##
                QCE[i, j, k] = model.getVarByName("qCE{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                QIAC[i, j, k] = model.getVarByName("qIAC{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                ECSS[i, j, k] = model.getVarByName("eCSS{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                QCS_DC[i, j, k] = model.getVarByName("qCS_DC{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                # The output is cooling
                QCS_CH[i, j, k] = model.getVarByName("qCS_CH{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                # The input is electricity
                QCD[i, j, k] = model.getVarByName("qCD{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                ## Group 4: Gas ##
                VCHP[i, j, k] = model.getVarByName("vCHP{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                VGAS[i, j, k] = model.getVarByName("vGAS{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X
                Temprature_in[i, j, k] = model.getVarByName("temprature_in{0}".format(
                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k)).X

    # Solution check
    for i in range(T):  # Dispatch at each time slot
        for j in range(N_scenario_second_stage):  # Scheduling plan under second stage plan
            for k in range(N_scenario_first_stage):  # Dispatch at each time slot
                if PAC2DC[i, j, k] * PDC2AC[i, j, k] != 0:
                    print(PAC2DC[i, j, k] * PDC2AC[i, j, k])
                if PESS_DC[i, j, k] * PESS_CH[i, j, k] != 0:
                    print(PESS_DC[i, j, k] * PESS_CH[i, j, k])
                if QHS_DC[i, j, k] * QHS_CH[i, j, k] != 0:
                    print(QHS_DC[i, j, k] * QHS_CH[i, j, k])
                if QCS_DC[i, j, k] * QCS_CH[i, j, k] != 0:
                    print(QCS_DC[i, j, k] * QCS_CH[i, j, k])

    EPISON = model.getVarByName("EPISON").X
    Nu_index = zeros(N_scenario_second_stage)

    for i in range(N_scenario_second_stage):
        Nu_index[i] = model.getVarByName("NU_INDEX{0}".format(i)).X

    obj_CVaR = EPISON
    for i in range(N_scenario_second_stage):
        obj_CVaR += Nu_index[i] * weight_second_stage[i] / (1 - alpha)

    if (1 - Lam) > 0:
        obj = (obj - Lam * obj_CVaR) / (1 - Lam)
    else:
        obj_DA = 0
        for i in range(T):
            for j in range(N_scenario_first_stage):
                obj_DA += pDA[i, j] * Price_DA[i, j] * weight_first_stage[j]

        obj_RT = 0
        for k in range(N_scenario_first_stage):
            for j in range(N_scenario_second_stage):
                for i in range(T):
                    obj_RT += PRT[i, j, k] * ELEC_PRICE[i] * weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += PRT_positive[i, j, k] * ELEC_PRICE[i] * forecasting_errors_prices * Weight * \
                              weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += PRT_negative[i, j, k] * ELEC_PRICE[i] * forecasting_errors_prices * Weight * \
                              weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += PESS_DC[i, j, k] * ESS["BESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += PESS_CH[i, j, k] * ESS["BESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += QHS_DC[i, j, k] * ESS["TESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += QHS_CH[i, j, k] * ESS["TESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += QCS_DC[i, j, k] * ESS["CESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += QCS_CH[i, j, k] * ESS["CESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += VGAS[i, j, k] * CCHP["COST"] * weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += VCHP[i, j, k] * CCHP["COST"] * weight_first_stage[k] * weight_second_stage[j]
                    obj_RT += PPV[i, j, k] * PV_cost * weight_first_stage[k] * weight_second_stage[j]

    return obj[0], obj_CVaR


if __name__ == "__main__":
    """
        Using Lexicographic construction of the pay-off table to solve the bi-objective optimization problem
    """
    # 1) To calculate Lex tablet
    (obj, obj_cVaR) = main(10, 20, Lam=0)  # The operational cost
    L11 = obj
    (obj, obj_cVaR) = main(10, 20, Lam=0.999)  # The operational cost
    L22 = obj_cVaR
    (obj, obj_cVaR) = main(10, 20, Lam=0, mode=1, ec=L11)  # The operational cost
    L12 = obj_cVaR
    (obj, obj_cVaR) = main(10, 20, Lam=0, mode=2, cvar=L22)  # The operational cost
    L21 = obj

    P = 20
    ep = [0] * P
    for i in range(P):
        ep[i] = L22 + i * (L12 - L22) / P

    obj = zeros(P + 2)
    obj_cVaR = zeros(P + 2)
    for i in range(1, P + 1):
        (obj[i], obj_cVaR[i]) = main(10, 20, Lam=0, mode=2, cvar=ep[i - 1])

    obj[P + 1] = L11
    obj_cVaR[P + 1] = L12
    obj[0] = L21
    obj_cVaR[0] = L22

    f = open("obj.txt", "w+")
    np.savetxt(f, obj, '%.18g', delimiter=',')
    f.close()

    f = open("obj_cvar.txt", "w+")
    np.savetxt(f, obj_cVaR, '%.18g', delimiter=',')
    f.close()

    print(obj)
