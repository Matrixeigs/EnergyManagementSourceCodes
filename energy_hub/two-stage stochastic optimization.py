"""
Two-stage stochastic optimization with recovery for the energy hub system
"""
from numpy import array, arange, zeros, inf
from matplotlib import pyplot
from scipy import interpolate
from random import random
from energy_hub.problem_formualtion import ProblemFormulation
from solvers.mixed_integer_solvers_gurobi import mixed_integer_linear_programming as milp


def problem_formulation(N, delta, weight_factor):
    """
    Jointed optimization for the electrical and thermal optimisation
    :param N: number of scenario
    :param delta: forecasting errors
    :param weight_factor: weight factor between the first stage decision making and second stage decision making
    :return:
    """
    ## Parameter announcement group:
    # 1) Parameter settings for the HVAC system
    PHVAC_max = 10
    eff_HVAC = 0.9
    # 2)Parameter settings for the ESS system
    E0 = 10
    Emax = 20
    Emin = 2
    Pess_ch_max = 10
    Pess_dc_max = 10
    eff_dc = 0.9
    eff_ch = 0.9
    # 3)Parameter settings for the energy hub system
    BIC_cap = 10
    eff_BIC = 0.95
    # 4)Parameter settings for CHP system
    Gmax = 20
    eff_CHP_e = 0.4
    eff_CHP_h = 0.35
    # 5)Parameters settings for PV and load
    PV_cap = 20
    AC_PD_cap = 10
    DC_PD_cap = 10
    HD_cap = 5
    CD_cap = 5
    # 6)Time settings for the two-stage optimization
    Delta_first_stage = 1
    Delta_second_stage = 0.25
    T_first_stage = 24
    T_second_stage = int(T_first_stage / Delta_second_stage)
    ws = 0.5
    # 7) Load profile generation
    # 7.1) AC electrical demand
    AC_PD = array([323.0284, 308.2374, 318.1886, 307.9809, 331.2170, 368.6539, 702.0040, 577.7045, 1180.4547, 1227.6240,
                   1282.9344, 1311.9738, 1268.9502, 1321.7436, 1323.9218, 1327.1464, 1386.9117, 1321.6387, 1132.0476,
                   1109.2701, 882.5698, 832.4520, 349.3568, 299.9920])

    # 7.2) DC electrical demand
    DC_PD = array([287.7698, 287.7698, 287.7698, 287.7698, 299.9920, 349.3582, 774.4047, 664.0625, 1132.6996, 1107.7366,
                   1069.6837, 1068.9819, 1027.3295, 1096.3820, 1109.4778, 1110.7039, 1160.1270, 1078.7839, 852.2514,
                   791.5814, 575.4085, 551.1441, 349.3568, 299.992])

    # 7.3) Heating demand
    HD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])

    # 7.4) Cooling demand
    CD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])

    # 7.5) PV load profile
    PV_PG = array(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])
    PV_PG = PV_PG * PV_cap
    # Modify the first stage profiles
    AC_PD = (AC_PD / max(AC_PD)) * AC_PD_cap
    DC_PD = (DC_PD / max(DC_PD)) * DC_PD_cap
    HD = (HD / max(HD)) * HD_cap
    CD = (CD / max(CD)) * CD_cap

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
    # 8) Price profile
    Electric_price = array(
        [76.01, 73.91, 71.31, 69.24, 68.94, 70.56, 75.16, 73.19, 79.70, 85.76, 86.90, 88.60, 90.62, 91.26, 93.70, 90.94,
         91.26, 80.39, 76.25, 76.80, 81.22, 83.75, 76.16, 72.69])  # Electrical prices
    Electric_price = Electric_price / 1000
    Gas_price = 0.1892  # Gas prices
    Eess_cost = 0.01  # Battery degradation cost

    # Step 2:test the formulated problems
    # 2.1) The first stage optimization problem
    problem_formulation_model = ProblemFormulation()
    first_stage_model = problem_formulation_model.first_stage_problem(PHVAC_max, eff_HVAC, Pess_ch_max, Pess_dc_max,
                                                                      eff_dc, eff_ch, E0, Emax,
                                                                      Emin, BIC_cap, Gmax, eff_BIC, eff_CHP_e,
                                                                      eff_CHP_h, AC_PD, DC_PD, HD,
                                                                      CD, PV_PG, Gas_price, Electric_price, Eess_cost,
                                                                      Delta_first_stage, T_first_stage)
    c = first_stage_model["c"]
    A = first_stage_model["A"]
    b = first_stage_model["b"]
    Aeq = first_stage_model["Aeq"]
    beq = first_stage_model["beq"]
    lb = first_stage_model["lb"]
    ub = first_stage_model["ub"]
    vtypes = ["c"] * len(lb)
    (solution, obj, success) = milp(c, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lb, xmax=ub, vtypes=vtypes)
    # 2.2) The second stage optimization problem
    (second_stage_mode, second_stage_model) = problem_formulation_model.second_stage_problem(PHVAC_max, eff_HVAC,
                                                                                            Pess_ch_max, Pess_dc_max,
                                                                                            eff_dc, eff_ch, E0, Emax,
                                                                                            Emin, BIC_cap, Gmax,
                                                                                            eff_BIC, eff_CHP_e,
                                                                                            eff_CHP_h,
                                                                                            AC_PD_second_stage,
                                                                                            DC_PD_second_stage, HD,
                                                                                            CD, PV_PG_second_stage,
                                                                                            Gas_price,
                                                                                            Electric_price, Eess_cost,
                                                                                            Delta_second_stage,
                                                                                            T_second_stage,
                                                                                            weight_factor)
    # c = second_stage_mode["c"]
    # A = second_stage_mode["A"]
    # b = second_stage_mode["b"]
    # Aeq = second_stage_mode["Aeq"]
    # beq = second_stage_mode["beq"]
    # lb = second_stage_mode["lb"]
    # ub = second_stage_mode["ub"]
    # vtypes = ["c"] * len(lb)
    c = second_stage_model["c"]
    A = second_stage_model["A"]
    b = second_stage_model["b"]
    Aeq = second_stage_model["Aeq"]
    beq = second_stage_model["beq"]
    lb = second_stage_model["lb"]
    ub = second_stage_model["ub"]
    vtypes = ["c"] * len(lb)
    (solution, obj, success) = milp(c, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lb, xmax=ub, vtypes=vtypes)

    # Step 3: formulate the coupling matrix
    # 3.1) Fixed recourse equation formulation
    # 3.1.1) Energy storage system systems
    # Remove the constraints to the
    T_s = zeros((T_first_stage, len(first_stage_model)))
    # Step 4: Using the benders iteration

    # Step 5: Obtained solution check and analysis

    # Step 6: return result
    return obj  # Formulated mixed integer linear programming problem


if __name__ == "__main__":
    model = problem_formulation(100, 0.05, 0.01)
    print(model)
