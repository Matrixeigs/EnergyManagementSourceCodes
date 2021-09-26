"""
Two stage stochastic optimization problem for the hybrid AC/DC microgrid embedded energy hub
@author: Tianyang Zhao
@mail: zhaoty@ntu.edu.sg
@date:27 Jan 2018
Note:
This function can also be used as a test function to evaluate the value of information
"""
from numpy import array, arange, zeros, inf
from matplotlib import pyplot
from scipy import interpolate
from random import random
from gurobipy import *


def problem_formulation(N, delta, weight_factor):
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
    ws = 0.5
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

    # Generate the second stage profiles using spline of scipy
    Time_first_stage = arange(0, T_first_stage, Delta_first_stage)
    Time_second_stage = arange(0, T_first_stage, Delta_second_stage)

    AC_PD_tck = interpolate.splrep(Time_first_stage, AC_PD, s=0)
    DC_PD_tck = interpolate.splrep(Time_first_stage, DC_PD, s=0)
    HD_tck = interpolate.splrep(Time_first_stage, HD, s=0)
    CD_tck = interpolate.splrep(Time_first_stage, CD, s=0)
    PV_PG_tck = interpolate.splrep(Time_first_stage, PV_PG, s=0)

    AC_PD_second_stage = interpolate.splev(Time_second_stage, AC_PD_tck, der=0)
    DC_PD_second_stage = interpolate.splev(Time_second_stage, DC_PD_tck, der=0)
    HD_second_stage = interpolate.splev(Time_second_stage, HD_tck, der=0)
    CD_second_stage = interpolate.splev(Time_second_stage, CD_tck, der=0)
    PV_PG_second_stage = interpolate.splev(Time_second_stage, PV_PG_tck, der=0)
    for i in range(T_second_stage):
        if AC_PD_second_stage[i] < 0:
            AC_PD_second_stage[i] = 0
        if DC_PD_second_stage[i] < 0:
            DC_PD_second_stage[i] = 0
        if HD_second_stage[i] < 0:
            HD_second_stage[i] = 0
        if CD_second_stage[i] < 0:
            CD_second_stage[i] = 0
        if PV_PG_second_stage[i] < 0:
            PV_PG_second_stage[i] = 0
    # Check the result
    # pyplot.plot(Time_first_stage, AC_PD, 'x', Time_second_stage, AC_PD_second_stage, 'b')
    # pyplot.plot(Time_first_stage, DC_PD, 'x', Time_second_stage, DC_PD_second_stage, 'b')
    # pyplot.plot(Time_first_stage, HD, 'x', Time_second_stage, HD_second_stage, 'b')
    # pyplot.plot(Time_first_stage, CD, 'x', Time_second_stage, CD_second_stage, 'b')
    # pyplot.plot(Time_first_stage, PV_PG, 'x', Time_second_stage, PV_PG_second_stage, 'b')
    # pyplot.show()

    # Generate profiles for each scenarion in the second stage
    AC_PD_scenario = zeros(shape=(N, T_second_stage))
    DC_PD_scenario = zeros(shape=(N, T_second_stage))
    HD_scenario = zeros(shape=(N, T_second_stage))
    CD_scenario = zeros(shape=(N, T_second_stage))
    PV_PG_scenario = zeros(shape=(N, T_second_stage))

    for i in range(N):
        for j in range(T_second_stage):
            AC_PD_scenario[i, j] = AC_PD_second_stage[j] * (1 - delta + 2 * delta * random())
            DC_PD_scenario[i, j] = DC_PD_second_stage[j] * (1 - delta + 2 * delta * random())
            HD_scenario[i, j] = HD_second_stage[j] * (1 - delta + 2 * delta * random())
            CD_scenario[i, j] = CD_second_stage[j] * (1 - delta + 2 * delta * random())
            PV_PG_scenario[i, j] = PV_PG_second_stage[j] * (1 - delta + 2 * delta * random())
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
    # 2) second stage optimisation
    # This serves as the test system for the test system
    pug = {}
    g = {}
    pAC2DC = {}
    pDC2AC = {}
    pHVAC = {}
    eess = {}
    pess_dc = {}
    pess_ch = {}
    ph_positive_derivation = {}
    ph_negative_derivation = {}
    pc_positive_derivation = {}
    pc_negative_derivation = {}

    for j in range(N):
        for i in range(T_second_stage):
            pug[i + j * T_second_stage] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                       name="pug{0}".format(i + j * T_second_stage))
            g[i + j * T_second_stage] = model.addVar(lb=0, ub=Gmax, vtype=GRB.CONTINUOUS,
                                                     name="g{0}".format(i + j * T_second_stage))
            pAC2DC[i + j * T_second_stage] = model.addVar(lb=0, ub=BIC_cap, vtype=GRB.CONTINUOUS,
                                                          name="a2d{0}".format(i + j * T_second_stage))
            pDC2AC[i + j * T_second_stage] = model.addVar(lb=0, ub=BIC_cap, vtype=GRB.CONTINUOUS,
                                                          name="d2a{0}".format(i + j * T_second_stage))
            pHVAC[i + j * T_second_stage] = model.addVar(lb=0, ub=PHVDC_max, vtype=GRB.CONTINUOUS,
                                                         name="pHVAC{0}".format(i + j * T_second_stage))
            eess[i + j * T_second_stage] = model.addVar(lb=Emin, ub=Emax, vtype=GRB.CONTINUOUS,
                                                        name="eess{0}".format(i + j * T_second_stage))
            pess_dc[i + j * T_second_stage] = model.addVar(lb=0, ub=Pess_dc_max, vtype=GRB.CONTINUOUS,
                                                           name="pess_dc{0}".format(i + j * T_second_stage))
            pess_ch[i + j * T_second_stage] = model.addVar(lb=0, ub=Pess_ch_max, vtype=GRB.CONTINUOUS,
                                                           name="pess_ch{0}".format(i + j * T_second_stage))
            # Auxiliary variables
            ph_positive_derivation[i + j * T_second_stage] = model.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS,
                                                                          name="ph_positive_derivation{0}".format(
                                                                              i + j * T_second_stage))
            ph_negative_derivation[i + j * T_second_stage] = model.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS,
                                                                          name="ph_negative_derivation{0}".format(
                                                                              i + j * T_second_stage))
            pc_positive_derivation[i + j * T_second_stage] = model.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS,
                                                                          name="pc_positive_derivation{0}".format(
                                                                              i + j * T_second_stage))
            pc_negative_derivation[i + j * T_second_stage] = model.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS,
                                                                          name="pc_negative_derivation{0}".format(
                                                                              i + j * T_second_stage))

    obj_first_stage = 0
    for i in range(T_first_stage):
        obj_first_stage += (G[i] * Gas_price + Electric_price[i] * Pug[i] + Eess_cost * (
                Pess_ch[i] + Pess_dc[i])) * Delta_first_stage

    obj_second_stage = 0
    for j in range(N):
        for i in range(T_second_stage):
            obj_second_stage += (g[i + j * T_second_stage] * Gas_price + Electric_price[int(i * Delta_second_stage)] * \
                                 pug[i + j * T_second_stage] + \
                                 Eess_cost * (pess_ch[i + j * T_second_stage] + pess_dc[i + j * T_second_stage]) + \
                                 ph_negative_derivation[i + j * T_second_stage] * weight_factor + \
                                 ph_positive_derivation[i + j * T_second_stage] * weight_factor + \
                                 pc_negative_derivation[i + j * T_second_stage] * weight_factor + \
                                 pc_positive_derivation[i + j * T_second_stage] * weight_factor) * Delta_second_stage

    for i in range(T_first_stage):
        model.addConstr(G[i] * eff_CHP_h == HD[i])
        model.addConstr(PHVAC[i] * eff_HVDC == CD[i])
        model.addConstr(Pug[i] + G[i] * eff_CHP_e + eff_BIC * PDC2AC[i] == AC_PD[i] + PAC2DC[i])
        model.addConstr(Pess_dc[i] - Pess_ch[i] + eff_BIC * PAC2DC[i] + PV_PG[i] == DC_PD[i] + PDC2AC[i])
        if i == 0:
            model.addConstr(Eess[i] - E0 == Pess_ch[i] * eff_ch - Pess_dc[i] / eff_dc)
        else:
            model.addConstr(Eess[i] - Eess[i - 1] == Pess_ch[i] * eff_ch - Pess_dc[i] / eff_dc)

    for j in range(N):
        for i in range(T_second_stage):
            # model.addConstr(g[i + j * T_second_stage] * eff_CHP_h == HD_scenario[j, i])
            # model.addConstr(pHVAC[i + j * T_second_stage] * eff_HVDC == CD_scenario[j, i])
            model.addConstr(pug[i + j * T_second_stage] + g[i + j * T_second_stage] * eff_CHP_e + eff_BIC * pDC2AC[
                i + j * T_second_stage] == AC_PD_scenario[j, i] + pAC2DC[i + j * T_second_stage])
            model.addConstr(pess_dc[i + j * T_second_stage] - pess_ch[i + j * T_second_stage] + eff_BIC * pAC2DC[
                i + j * T_second_stage] + PV_PG_scenario[j, i] == DC_PD_scenario[j, i] + pDC2AC[
                                i + j * T_second_stage])
            if i == 0:
                model.addConstr(
                    eess[i + j * T_second_stage] - E0 == pess_ch[i + j * T_second_stage] * eff_ch * Delta_second_stage -
                    pess_dc[
                        i + j * T_second_stage] * Delta_second_stage / eff_dc)
            else:
                model.addConstr(eess[i + j * T_second_stage] - eess[i + j * T_second_stage - 1] == pess_ch[
                    i + j * T_second_stage] * Delta_second_stage * eff_ch - pess_dc[
                                    i + j * T_second_stage] * Delta_second_stage / eff_dc)
            # The correlationship between the first and second stage optimization problem
    for j in range(N):
        for i in range(T_first_stage):
            model.addConstr((g[4 * i + j * T_second_stage] + g[4 * i + 1 + j * T_second_stage] + g[
                4 * i + 2 + j * T_second_stage] + g[4 * i + 3 + j * T_second_stage]) * eff_CHP_h * Delta_second_stage ==
                            HD[i])
            model.addConstr((pHVAC[4 * i + j * T_second_stage] + pHVAC[4 * i + 1 + j * T_second_stage] + pHVAC[
                4 * i + 2 + j * T_second_stage] + pHVAC[
                                 4 * i + 3 + j * T_second_stage]) * eff_HVDC * Delta_second_stage == CD[i])

    # Coupling constraints
    for j in range(N):
        for i in range(T_first_stage):
            model.addConstr(eess[int(i / Delta_second_stage) + j * T_second_stage] == Eess[i])

    for j in range(N):
        for i in range(T_second_stage):
            model.addConstr(
                ph_positive_derivation[i + j * T_second_stage] >= g[i] * eff_CHP_h - HD[int(i * Delta_second_stage)])
            model.addConstr(
                ph_negative_derivation[i + j * T_second_stage] >= HD[int(i * Delta_second_stage)] - g[i] * eff_CHP_h)
            model.addConstr(
                pc_positive_derivation[i + j * T_second_stage] >= pHVAC[i] * eff_HVDC - CD[int(i * Delta_second_stage)])
            model.addConstr(
                pc_negative_derivation[i + j * T_second_stage] >= CD[int(i * Delta_second_stage)] - pHVAC[i] * eff_HVDC)

    # set the objective function
    obj = ws * obj_first_stage + (1 - ws) * obj_second_stage / N
    model.setObjective(obj)

    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    model.Params.DisplayInterval = 1
    model.Params.LogFile = ""
    model.optimize()

    Pug_x = zeros((T_first_stage, 1))
    G_x = zeros((T_first_stage, 1))
    PAC2DC_x = zeros((T_first_stage, 1))
    PDC2AC_x = zeros((T_first_stage, 1))
    PHVAC_x = zeros((T_first_stage, 1))
    Eess_x = zeros((T_first_stage, 1))
    Pess_dc_x = zeros((T_first_stage, 1))
    Pess_ch_x = zeros((T_first_stage, 1))

    pug_x = zeros((T_second_stage, N))
    g_x = zeros((T_second_stage, N))
    pAC2DC_x = zeros((T_second_stage, N))
    pDC2AC_x = zeros((T_second_stage, N))
    pHVAC_x = zeros((T_second_stage, N))
    eess_x = zeros((T_second_stage, N))
    pess_dc_x = zeros((T_second_stage, N))
    pess_ch_x = zeros((T_second_stage, N))

    ph_positive_derivation_x = zeros((T_second_stage, N))
    ph_negative_derivation_x = zeros((T_second_stage, N))
    pc_positive_derivation_x = zeros((T_second_stage, N))
    pc_negative_derivation_x = zeros((T_second_stage, N))

    for i in range(T_first_stage):
        Pug_x[i] = model.getVarByName("Pug{0}".format(i)).X
        G_x[i] = model.getVarByName("G{0}".format(i)).X
        PAC2DC_x[i] = model.getVarByName("A2D{0}".format(i)).X
        PDC2AC_x[i] = model.getVarByName("D2A{0}".format(i)).X
        PHVAC_x[i] = model.getVarByName("PHVAC{0}".format(i)).X
        Eess_x[i] = model.getVarByName("Eess{0}".format(i)).X
        Pess_dc_x[i] = model.getVarByName("Pess_dc{0}".format(i)).X
        Pess_ch_x[i] = model.getVarByName("Pess_ch{0}".format(i)).X

    for j in range(N):
        for i in range(T_second_stage):
            pug_x[i, j] = model.getVarByName("pug{0}".format(i + j * T_second_stage)).X
            g_x[i, j] = model.getVarByName("g{0}".format(i + j * T_second_stage)).X
            pAC2DC_x[i, j] = model.getVarByName("a2d{0}".format(i + j * T_second_stage)).X
            pDC2AC_x[i, j] = model.getVarByName("d2a{0}".format(i + j * T_second_stage)).X
            pHVAC_x[i, j] = model.getVarByName("pHVAC{0}".format(i + j * T_second_stage)).X
            eess_x[i, j] = model.getVarByName("eess{0}".format(i + j * T_second_stage)).X
            pess_dc_x[i, j] = model.getVarByName("pess_dc{0}".format(i + j * T_second_stage)).X
            pess_ch_x[i, j] = model.getVarByName("pess_ch{0}".format(i + j * T_second_stage)).X

            ph_positive_derivation_x[i, j] = model.getVarByName(
                "ph_positive_derivation{0}".format(i + j * T_second_stage)).X
            ph_negative_derivation_x[i, j] = model.getVarByName(
                "ph_negative_derivation{0}".format(i + j * T_second_stage)).X
            pc_positive_derivation_x[i, j] = model.getVarByName(
                "pc_positive_derivation{0}".format(i + j * T_second_stage)).X
            pc_negative_derivation_x[i, j] = model.getVarByName(
                "pc_negative_derivation{0}".format(i + j * T_second_stage)).X

    obj = obj.getValue()

    result = {"PUG": Pug_x,
              "G": G_x,
              "PAC2DC": PAC2DC_x,
              "PDC2AC": PDC2AC_x,
              "PHVAC": PHVAC_x,
              "Eess": Eess_x,
              "Pess_dc": Pess_dc_x,
              "Pess_ch": Pess_ch_x,
              "obj": obj
              }

    # from energy_hub.problem_formualtion import ProblemFormulation
    # from solvers.mixed_integer_solvers_gurobi import mixed_integer_linear_programming as milp

    # test_model = ProblemFormulation()
    # test_model = test_model.first_stage_problem(PHVDC_max, eff_HVDC, Pess_ch_max, Pess_dc_max, eff_dc, eff_ch, E0, Emax,
    #                                             Emin, BIC_cap, Gmax, eff_BIC, eff_CHP_e, eff_CHP_h, AC_PD, DC_PD, HD,
    #                                             CD, PV_PG, Gas_price, Electric_price, Eess_cost, Delta_first_stage, T_first_stage)
    # c = test_model["c"]
    # A = test_model["A"]
    # b = test_model["b"]
    # Aeq = test_model["Aeq"]
    # beq = test_model["beq"]
    # lb = test_model["lb"]
    # ub = test_model["ub"]
    # vtypes = ["c"] * len(lb)
    # (solution, obj, success) = milp(c, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lb, xmax=ub,vtypes=vtypes)
    # gap = obj/result["obj"]
    return result  # Formulated mixed integer linear programming problem


if __name__ == "__main__":
    model = problem_formulation(50, 0.05, 0.01)
    print(model)
