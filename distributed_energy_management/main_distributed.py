# The distributed energy management system

from numpy import zeros,vstack,asarray
from gurobipy import *

def problem_formulation(*args):
    from distributed_energy_management.configuration import configuration_time_line
    from distributed_energy_management.modelling.power_flow.idx_ed_recovery_format import PG, RG, PUG, RUG, PBIC_AC2DC, PBIC_DC2AC, \
        PESS_C, PESS_DC, RESS, EESS, PMG, PPV, PWP, PL_AC, PL_UAC, PL_DC, PL_UDC, NX

    model = args[0]  # If multiple models are inputed, more local ems models will be formulated
    ## The infeasible optimal problem formulation
    T = args[1]
    nx = NX * T
    lb = zeros(NX)
    ub = zeros(NX)
    ## Update lower boundary
    lb[PG] = model["DG"]["PMIN"]
    lb[RG] = model["DG"]["PMIN"]

    lb[PUG] = model["UG"]["PMIN"]
    lb[RUG] = model["UG"]["PMIN"]

    lb[PBIC_AC2DC] = 0
    lb[PBIC_DC2AC] = 0

    lb[PESS_C] = 0
    lb[PESS_DC] = 0
    lb[RESS] = 0
    lb[EESS] = model["ESS"]["SOC_MIN"] * model["ESS"]["CAP"]

    lb[PMG] = 0  # The line flow limitation, the predefined status is, the transmission line is off-line

    ## Update lower boundary
    ub[PG] = model["DG"]["PMAX"]
    ub[RG] = model["DG"]["PMAX"]

    ub[PUG] = model["UG"]["PMAX"]
    ub[RUG] = model["UG"]["PMAX"]

    ub[PBIC_AC2DC] = model["BIC"]["CAP"]
    ub[PBIC_DC2AC] = model["BIC"]["CAP"]

    ub[PESS_C] = model["ESS"]["PMAX_CH"]
    ub[PESS_DC] = model["ESS"]["PMAX_DIS"]
    ub[RESS] = model["ESS"]["PMAX_DIS"] + model["ESS"]["PMAX_CH"]
    ub[EESS] = model["ESS"]["SOC_MAX"] * model["ESS"]["CAP"]

    ub[PMG] = 0  # The line flow limitation, the predefined status is, the transmission line is off-line
    ## Constraints set
    # 1) Power balance equation
    Aeq = zeros((T, nx))
    beq = [ ]
    for i in range(T):
        Aeq[i][i * NX + PG] = 1
        Aeq[i][i * NX + PUG] = 1
        Aeq[i][i * NX + PBIC_AC2DC] = -1
        Aeq[i][i * NX + PBIC_DC2AC] = model["BIC"]["EFF_DC2AC"]
        beq.append(model["Load_ac"]["PD"] + model["Load_uac"]["PD"])# Add the
    # 2) DC power balance equation
    Aeq_temp = zeros((T, nx))
    for i in range(T):
        Aeq_temp[i][i * NX + PBIC_AC2DC] = model["BIC"]["EFF_AC2DC"]
        Aeq_temp[i][i * NX + PBIC_DC2AC] = -1
        Aeq_temp[i][i * NX + PESS_C] = -1
        Aeq_temp[i][i * NX + PESS_DC] = 1
        Aeq_temp[i][i * NX + PMG] = -1
        beq.append(model["Load_dc"]["PD"] + model["Load_udc"]["PD"] - model["PV"]["PG"] - model["WP"]["PG"])
    Aeq = vstack([Aeq, Aeq_temp])

    # 3) Energy storage system
    Aeq_temp = zeros((T, nx))
    for i in range(T):
        if i == 0:
            Aeq_temp[i][i * NX + EESS] = 1
            Aeq_temp[i][i * NX + PESS_C] = -model["ESS"]["EFF_CH"] * configuration_time_line.default_time[
                "Time_step_ed"] / 3600
            Aeq_temp[i][i * NX + PESS_DC] = 1 / model["ESS"]["EFF_DIS"] * configuration_time_line.default_time[
                "Time_step_ed"] / 3600
            beq.append(model["ESS"]["SOC"] * model["ESS"]["CAP"])
        else:
            Aeq_temp[i][(i - 1) * NX + EESS] = -1
            Aeq_temp[i][i * NX + EESS] = 1
            Aeq_temp[i][i * NX + PESS_C] = -model["ESS"]["EFF_CH"] * configuration_time_line.default_time[
                "Time_step_ed"] / 3600
            Aeq_temp[i][i * NX + PESS_DC] = 1 / model["ESS"]["EFF_DIS"] * configuration_time_line.default_time[
                "Time_step_ed"] / 3600
            beq.append(0)
    Aeq = vstack([Aeq, Aeq_temp])
    # Inequality constraints
    # 1) PG + RG <= PGMAX
    Aineq = zeros((T, nx))
    bineq = []
    for i in range(T):
        Aineq[i][i * NX + PG] = 1
        Aineq[i][i * NX + RG] = 1
        bineq.append(model["DG"]["PMAX"])
    # 2) PG - RG >= PGMIN
    Aineq_temp = zeros((T, nx))
    for i in range(T):
        Aineq_temp[i][i * NX + PG] = -1
        Aineq_temp[i][i * NX + RG] = 1
        bineq.append(-model["DG"]["PMIN"])
    Aineq = vstack([Aineq, Aineq_temp])
    # 3) PUG + RUG <= PUGMAX
    Aineq_temp = zeros((T, nx))
    for i in range(T):
        Aineq_temp[i][i * NX + PUG] = 1
        Aineq_temp[i][i * NX + RUG] = 1
        bineq.append(model["UG"]["PMAX"])
    Aineq = vstack([Aineq, Aineq_temp])
    # 4) PUG - RUG >= PUGMIN
    Aineq_temp = zeros((T, nx))
    for i in range(T):
        Aineq_temp[i][i * NX + PUG] = -1
        Aineq_temp[i][i * NX + RUG] = 1
        bineq.append(-model["UG"]["PMIN"])
    Aineq = vstack([Aineq, Aineq_temp])
    # 5) PESS_DC - PESS_C + RESS <= PESS_DC_MAX
    Aineq_temp = zeros((T, nx))
    for i in range(T):
        Aineq_temp[i][i * NX + PESS_DC] = 1
        Aineq_temp[i][i * NX + PESS_C] = -1
        Aineq_temp[i][i * NX + RESS] = 1
        bineq.append(model["ESS"]["PMAX_DIS"])
    Aineq = vstack([Aineq, Aineq_temp])
    # 6) PESS_DC - PESS_C - RESS >= -PESS_C_MAX
    Aineq_temp = zeros((T, nx))
    for i in range(T):
        Aineq_temp[i][i * NX + PESS_DC] = -1
        Aineq_temp[i][i * NX + PESS_C] = 1
        Aineq_temp[i][i * NX + RESS] = 1
        bineq.append(model["ESS"]["PMAX_CH"])
    Aineq = vstack([Aineq, Aineq_temp])
    # 7) EESS - RESS*delta >= EESSMIN
    Aineq_temp = zeros((T, nx))
    for i in range(T):
        Aineq_temp[i][i * NX + EESS] = -1
        Aineq_temp[i][i * NX + RESS] = configuration_time_line.default_time["Time_step_ed"] / 3600
        bineq.append(-model["ESS"]["SOC_MIN"] * model["ESS"]["CAP"])
    Aineq = vstack([Aineq, Aineq_temp])
    # 8) EESS + RESS*delta <= EESSMAX
    Aineq_temp = zeros((T, nx))
    for i in range(T):
        Aineq_temp[i][i * NX + EESS] = 1
        Aineq_temp[i][i * NX + RESS] = configuration_time_line.default_time["Time_step_ed"] / 3600
        bineq.append(model["ESS"]["SOC_MAX"] * model["ESS"]["CAP"])
    Aineq = vstack([Aineq, Aineq_temp])
    # 9) RG + RUG + RESS >= sum(Load)*beta + sum(PV)*beta_pv + sum(WP)*beta_wp
    # No reserve requirement
    c = zeros(NX)
    if model["DG"]["COST_MODEL"] == 2:
        c[PG] = model["DG"]["COST"][1]
    else:
        c[PG] = model["DG"]["COST"][0]
    c[PUG] = model["UG"]["COST"][0]
    c[PESS_C] = model["ESS"]["COST_CH"][0]
    c[PESS_DC] = model["ESS"]["COST_DIS"][0]

    lb_list = [ ]
    ub_list = [ ]
    c_list = [ ]
    for i in range(NX):
        lb_list.append(lb[i])
        ub_list.append(ub[i])
        c_list.append(c[i])

    C = c_list * T
    LB = lb_list * T
    UB = ub_list * T
    # Generate the quadratic parameters
    Q = zeros((nx, nx))
    for i in range(T):
        if model["DG"]["COST_MODEL"] == 2:
            Q[i * NX + PG][i * NX + PG] = model["DG"]["COST"][1]
    mathematical_model = {"Q": Q,
                          "c": C,
                          "Aeq": Aeq,
                          "beq": beq,
                          "A": Aineq,
                          "b": bineq,
                          "lb": LB,
                          "ub": UB}

    return mathematical_model
# The master problem formulation
def global_problem_formulation(*args):
    model = args[0] # The transmission line parameters
    T = args[1] # The


# The solving procedure
def gurobi_solver(c, Aeq=None, beq=None, A=None, b=None, xmin=None, xmax=None, opt=None):
    from numpy import Inf, ones

    nx = c.shape[0]  # number of decision variables

    if A.any() != None:
        nineq = A.shape[0]  # number of equality constraints
    else:
        nineq = 0

    if Aeq.any() != None:
        neq = Aeq.shape[0]  # number of inequality constraints
    else:
        neq = 0
    # Fulfilling the missing informations
    if beq is None or len(beq) == 0: beq = -Inf * ones(neq)
    if b is None or len(b) == 0: b = Inf * ones(nineq)
    if xmin is None or len(xmin) == 0: xmin = -Inf * ones(nx)
    if xmax is None or len(xmax) == 0: xmax = Inf * ones(nx)

    # modelling based on the high level gurobi api
    try:
        gurobi_model = Model("MIP")
        # Declear the variables
        x = {}
        for i in range(nx):
            x[i] = gurobi_model.addVar(lb=xmin[i], ub=xmax[i], vtype=GRB.CONTINUOUS, name='"x{0}"'.format(i))
        # Constraints set
        # Equal constraints
        if neq != 0:
            for i in range(neq):
                expr = 0
                for j in range(nx):
                    expr += x[j] * Aeq[i, j]
                gurobi_model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

        # Inequal constraints
        if nineq != 0:
            for i in range(nineq):
                expr = 0
                for j in range(nx):
                    expr += x[j] * A[i, j]
                gurobi_model.addConstr(lhs=expr, sense=GRB.LESS_EQUAL, rhs=b[i])
        # Set the objective function
        obj = 0
        for i in range(nx):
            obj += x[i] * c[i]

        gurobi_model.setObjective(obj)

        gurobi_model.Params.OutputFlag = 0
        gurobi_model.Params.LogToConsole = 0
        gurobi_model.Params.DisplayInterval = 1
        gurobi_model.optimize()
        xx = [ ]
        for v in gurobi_model.getVars():
            # print('%s %g' % (v.varName, v.x))
            xx.append(v.x)

        obj = obj.getValue()
        # print('Obj: %g' % obj.getValue())

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
        xx = 0
        obj = 0

    except AttributeError:
        print('Encountered an attribute error')
        xx = 0
        obj = 0

    return xx, obj


if __name__ == "__main__":
    from distributed_energy_management.modelling import generators, loads, energy_storage_systems, convertors, transmission_lines  # Import modellings

    local_models = {"DG": generators.Generator_AC.copy(),
                    "UG": generators.Generator_AC.copy(),
                    "Load_ac": loads.Load_AC.copy(),
                    "Load_uac": loads.Load_AC.copy(),
                    "BIC": convertors.BIC.copy(),
                    "ESS": energy_storage_systems.BESS.copy(),
                    "PV": generators.Generator_RES.copy(),
                    "WP": generators.Generator_RES.copy(),
                    "Load_dc": loads.Load_DC.copy(),
                    "Load_udc": loads.Load_DC.copy(),
                    "PMG": 0,
                    "V_DC": 0}

    model = problem_formulation(local_models, 24)
    # Solve the problem by using gurobi
    c = asarray(model["c"])
    Aeq = model["Aeq"]
    beq = model["beq"]
    A = model["A"]
    b = model["b"]
    lb = asarray(model["lb"])
    ub = asarray(model["ub"])

    (xx,obj) = gurobi_solver(c, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lb, xmax=ub)
    obj_real = 0
    for i in range(c.shape[0]):
        obj_real += c[i]*xx[i]

    print(obj_real-obj)






