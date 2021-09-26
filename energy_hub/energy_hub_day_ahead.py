"""
Two stage stochastic optimization problem for the hybrid AC/DC microgrid embedded energy hub
@author: Tianyang Zhao
@mail: zhaoty@ntu.edu.sg
@date:27 Jan 2018
"""
from numpy import array, arange, zeros
from matplotlib import pyplot
from scipy import interpolate
from random import random
from gurobipy import *


def main(N, delta, weight_factor):
    """
    Jointed optimization for the electrical and thermal optimisation
    :param N: number of scenario
    :param delta: forecasting errors
    :param weight_factor: weight factor between the first stage decision making and second stage decision makin
    :return:
    """
    # Parameters settings for the
    PHVDC_max = 10
    eff_HVDC = 0.9
    Pess_ch_max = 10
    Pess_dc_max = 10
    eff_dc = 0.9
    eff_ch = 0.9
    E0 = 10
    Emax = 20
    Emin = 2
    BIC_cap = 10
    # For the HVAC system
    Gmax = 20
    eff_BIC = 0.95
    eff_CHP_e = 0.4
    eff_CHP_h = 0.35

    PV_cap = 20
    AC_PD_cap = 10
    DC_PD_cap = 10
    HD_cap = 5
    CD_cap = 5

    Delta_first_stage = 1
    Delta_second_stage = 0.25
    T_first_stage = 24
    T_second_stage = int(T_first_stage / Delta_second_stage)
    # AC electrical demand
    AC_PD = array([323.0284, 308.2374, 318.1886, 307.9809, 331.2170, 368.6539, 702.0040, 577.7045, 1180.4547, 1227.6240,
                   1282.9344, 1311.9738, 1268.9502, 1321.7436, 1323.9218, 1327.1464, 1386.9117, 1321.6387, 1132.0476,
                   1109.2701, 882.5698, 832.4520, 349.3568, 299.9920])

    # DC electrical demand
    DC_PD = array([287.7698, 287.7698, 287.7698, 287.7698, 299.9920, 349.3582, 774.4047, 664.0625, 1132.6996, 1107.7366,
                   1069.6837, 1068.9819, 1027.3295, 1096.3820, 1109.4778, 1110.7039, 1160.1270, 1078.7839, 852.2514,
                   791.5814, 575.4085, 551.1441, 349.3568, 299.992])

    # Heating demand
    HD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])

    # Cooling demand
    CD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])

    # PV load profile
    PV_PG = array(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])

    # Price profile
    Electric_price = array(
        [76.01, 73.91, 71.31, 69.24, 68.94, 70.56, 75.16, 73.19, 79.70, 85.76, 86.90, 88.60, 90.62, 91.26, 93.70, 90.94,
         91.26, 80.39, 76.25, 76.80, 81.22, 83.75, 76.16, 72.69])  # Electrical prices
    Electric_price = Electric_price / 1000
    Gas_price = 0.1892  # Gas prices
    Eess_cost = 0.01  # Battery degradation cost

    PV_PG = PV_PG * PV_cap
    # Modify the first stage profiles
    AC_PD = (AC_PD / max(AC_PD)) * AC_PD_cap
    DC_PD = (DC_PD / max(DC_PD)) * DC_PD_cap
    HD = (HD / max(HD)) * HD_cap
    CD = (CD / max(CD)) * CD_cap

    # Formulation of the two-stage optimization problem
    # 1) First stage optimization problems
    model = Model("EnergyHub")
    Pug = {}
    G = {}
    PAC2DC = {}
    PDC2AC = {}
    PHVAC = {}
    Eess = {}
    Pess_dc = {}
    Pess_ch = {}
    for i in range(T_first_stage):
        Pug[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Pug{0}".format(i))
        G[i] = model.addVar(lb=0, ub=Gmax, vtype=GRB.CONTINUOUS, name="G{0}".format(i))
        PAC2DC[i] = model.addVar(lb=0, ub=BIC_cap, vtype=GRB.CONTINUOUS, name="A2D{0}".format(i))
        PDC2AC[i] = model.addVar(lb=0, ub=BIC_cap, vtype=GRB.CONTINUOUS, name="D2A{0}".format(i))
        PHVAC[i] = model.addVar(lb=0, ub=PHVDC_max, vtype=GRB.CONTINUOUS, name="PHVAC{0}".format(i))
        Eess[i] = model.addVar(lb=Emin, ub=Emax, vtype=GRB.CONTINUOUS, name="Eess{0}".format(i))
        Pess_dc[i] = model.addVar(lb=0, ub=Pess_dc_max, vtype=GRB.CONTINUOUS, name="Pess_dc{0}".format(i))
        Pess_ch[i] = model.addVar(lb=0, ub=Pess_ch_max, vtype=GRB.CONTINUOUS, name="Pess_ch{0}".format(i))

    obj = 0
    for i in range(T_first_stage):
        obj = obj + G[i] * Gas_price + Electric_price[i] * Pug[i] + Eess_cost * (Pess_ch[i] + Pess_dc[i])

    for i in range(T_first_stage):
        model.addConstr(G[i] * eff_CHP_h == HD[i])
        model.addConstr(PHVAC[i] * eff_HVDC == CD[i])
        model.addConstr(Pug[i] + G[i] * eff_CHP_e + eff_BIC * PDC2AC[i] == AC_PD[i] + PAC2DC[i])
        model.addConstr(Pess_dc[i] - Pess_ch[i] + eff_BIC * PAC2DC[i] + PV_PG[i] == DC_PD[i] + PDC2AC[i])
        if i == 0:
            model.addConstr(Eess[i] - E0 == Pess_ch[i] * eff_ch - Pess_dc[i] / eff_dc)
        else:
            model.addConstr(Eess[i] - Eess[i - 1] == Pess_ch[i] * eff_ch - Pess_dc[i] / eff_dc)

    # 2) second stage optimisation

    # set the objective function
    model.setObjective(obj)

    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.Params.DisplayInterval = 1
    model.Params.LogFile = ""
    model.optimize()

    pug = []
    g = []
    pAC2DC = []
    pDC2AC = []
    pHVAC = []
    eess = []
    pess_dc = []
    pess_ch = []

    for i in range(T_first_stage):
        pug.append(model.getVarByName("Pug{0}".format(i)).X)
        g.append(model.getVarByName("G{0}".format(i)).X)
        pAC2DC.append(model.getVarByName("A2D{0}".format(i)).X)
        pDC2AC.append(model.getVarByName("D2A{0}".format(i)).X)
        pHVAC.append(model.getVarByName("PHVAC{0}".format(i)).X)
        eess.append(model.getVarByName("Eess{0}".format(i)).X)
        pess_dc.append(model.getVarByName("Pess_dc{0}".format(i)).X)
        pess_ch.append(model.getVarByName("Pess_ch{0}".format(i)).X)

    obj = obj.getValue()

    result = {"PUG": pug,
              "G": g,
              "PAC2DC": pAC2DC,
              "PDC2AC": pDC2AC,
              "PHVAC": pHVAC,
              "Eess": eess,
              "Pess_dc": pess_dc,
              "Pess_ch": pess_ch,
              "obj": obj
              }
    return result  # Formulated mixed integer linear programming problem


if __name__ == "__main__":
    result = main(50, 0.03, 0)
    print(result)
