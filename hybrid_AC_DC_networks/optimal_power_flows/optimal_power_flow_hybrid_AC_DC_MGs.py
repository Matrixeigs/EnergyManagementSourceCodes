"""
Optimal power flow for hybrid AC/DC micro-grids
Two versions of optimal power flow models are proposed.
1) Single period
2) Multiple periods
@author: Tianyang Zhao
@email: zhaoty@ntu.edu.sg
"""

from numpy import power, array, zeros, ones, vstack, shape, concatenate

# import test cases
from distribution_system_optimization.test_cases import case33
from pypower import case30

from gurobipy import *

M = 1e5


class MultipleMicrogridsDirect_CurrentNetworks():
    """
    Dynamic optimal power flow modelling for micro-grid power parks
    The power parks include
    """

    def __init__(self):
        self.name = "Test_MGs_DC_networks"

    def run(self, case_MGs=None, case_DC_network=None, case_AC_networks=None, T=1):
        # 1) Optimal power flow modelling for MGs
        # 2) Optimal power flow modelling for DC networks
        # 3) Connnection matrix between MGs and DC networks
        # 3.1) Update the decision variables
        # 3.2) Update the constraint set
        # 3.3) Update the objective functions
        # 4) Results check
        # 4.1) Bi-directional power flows on ESSs
        # 4.2) Bi-directional power flows on BICs
        # 4.3) Relaxation of DC power flows
        # 4.4) Stochastic simulation

        model_MGs = MultipleMicrogridsDirect_CurrentNetworks.optimal_power_flow_microgrid(self, case_MGs, T)

        # nx = len(model_MGs["lx"])
        # vtypes = ["c"] * nx
        # sol = milp(c=model_MGs["c"], Aeq=model_MGs["Aeq"], beq=model_MGs["beq"], A=model_MGs["A"], b=model_MGs["b"],
        #            xmin=model_MGs["lx"], xmax=model_MGs["ux"], vtypes=vtypes)

        model_DC = MultipleMicrogridsDirect_CurrentNetworks.optimal_power_flow_direct_current_networks(self,
                                                                                                       case_DC_network,
                                                                                                       caseMGs)
        # Formulate the dynamic optimal power optimal power flow problem
        neq = shape(model_DC["Aeq"])[0]
        NX = model_DC["nx"]
        nx = NX * T
        Q = zeros((nx, 1))
        c = zeros((nx, 1))
        c0 = zeros((nx, 1))
        Aeq = zeros((neq * T, nx))
        beq = zeros((neq * T, 1))
        lx = zeros((nx, 1))
        ux = zeros((nx, 1))

        for i in range(T):
            lx[i * NX:(i + 1) * NX] = model_DC["lx"]
            ux[i * NX:(i + 1) * NX] = model_DC["ux"]
            beq[i * neq:(i + 1) * neq] = model_DC["beq"]
            Q[i * NX:(i + 1) * NX] = model_DC["Q"]
            c[i * NX:(i + 1) * NX] = model_DC["c"]
            c0[i * NX:(i + 1) * NX] = model_DC["c0"]
            Aeq[i * neq:(i + 1) * neq, i * NX:(i + 1) * NX] = model_DC["Aeq"]

        # model = Model("OPF")
        # x = {}
        #
        # for i in range(nx):
        #     x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)
        #
        # for i in range(neq * T):
        #     expr = 0
        #     for j in range(nx):
        #         # if Aeq_agg[i, j] != 0:
        #         expr += x[j] * Aeq[i, j]
        #     model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])
        # # Add conic constraints
        # for i in range(T):
        #     for j in range(model_DC["nl"]):
        #         model.addConstr(
        #             x[i * NX + j] * x[i * NX + j] <= x[
        #                 i * NX + j + model_DC["nl"]] * x[
        #                 i * NX + model_DC["f"][j] + 2 * model_DC["nl"]])
        #
        # obj = 0
        # for i in range(nx):
        #     obj += Q[i, 0] * x[i] * x[i] + c[i, 0] * x[i] + c0[i, 0]
        #
        # model.setObjective(obj)
        # model.Params.OutputFlag = 0
        # model.Params.LogToConsole = 0
        # model.Params.DisplayInterval = 1
        # model.optimize()
        #
        # xx = []
        # for v in model.getVars():
        #     xx.append(v.x)
        #
        # obj = obj.getValue()
        # primal_residual = zeros(model_DC["nl"] * T)
        #
        # for i in range(T):
        #     for j in range(model_DC["nl"]):
        #         primal_residual[i * model_DC["nl"] + j] = xx[i * NX + j] * xx[i * NX + j] - xx[
        #             i * NX + j + model_DC["nl"]] * xx[
        #                                                       i * NX + int(model_DC["f"][j]) + 2 * model_DC["nl"]]

        # Formulate the centralized optimization problem
        nx_agg = nx + model_MGs["nx"]
        neq_agg = neq * T + model_MGs["neq"]
        nineq_agg = model_MGs["nineq"]

        lx_agg = vstack([model_MGs["lx"], lx])
        ux_agg = vstack([model_MGs["ux"], ux])

        Q_agg = vstack([zeros((model_MGs["nx"], 1)), Q])
        c_agg = vstack([model_MGs["c"], c])
        c0_agg = vstack([zeros((model_MGs["nx"], 1)), c0])

        Aeq_agg = zeros((neq_agg, nx_agg))
        Aeq_agg[0:model_MGs["neq"], 0:model_MGs["nx"]] = model_MGs["Aeq"]
        Aeq_agg[model_MGs["neq"]:neq_agg, model_MGs["nx"]:nx_agg] = Aeq
        beq_agg = vstack([model_MGs["beq"], beq])

        A_agg = zeros((nineq_agg, nx_agg))
        A_agg[0:model_MGs["nineq"], 0:model_MGs["nx"]] = model_MGs["A"]
        b_agg = model_MGs["b"]
        # The additional constraints for the interconnection
        nmg = len(case_MGs)
        Aeq_coupling = zeros((T * nmg, nx_agg))

        for i in range(nmg):
            for j in range(T):
                Aeq_coupling[
                    i * T + j, i * T * model_MGs["NX"] + j * model_MGs["NX"] + model_MGs["PMG"]] = 1 / case_DC_network[
                    "baseMVA"]  # The index in
                Aeq_coupling[
                    i * T + j, model_MGs["nx"] + j * model_DC["nx"] + 2 * model_DC["nl"] + model_DC["nb"] + model_DC[
                        "ng"] + i] = -1
        Aeq_agg = vstack([Aeq_agg, Aeq_coupling])
        beq_agg = vstack([beq_agg, zeros((T * nmg, 1))])
        neq_agg = len(beq_agg)
        # Formulate the optimization problem
        model = Model("OPF")
        x = {}

        for i in range(nx_agg):
            x[i] = model.addVar(lb=lx_agg[i], ub=ux_agg[i], vtype=GRB.CONTINUOUS)

        for i in range(neq_agg):
            expr = 0
            for j in range(nx_agg):
                # if Aeq_agg[i, j] != 0:
                expr += x[j] * Aeq_agg[i, j]
            model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq_agg[i])

        for i in range(nineq_agg):
            expr = 0
            for j in range(nx_agg):
                # if A_agg[i, j] != 0:
                expr += x[j] * A_agg[i, j]
            model.addConstr(lhs=expr, sense=GRB.LESS_EQUAL, rhs=b_agg[i])

        # Add conic constraints
        for i in range(T):
            for j in range(model_DC["nl"]):
                model.addConstr(
                    x[model_MGs["nx"] + i * NX + j] * x[model_MGs["nx"] + i * NX + j] <= x[
                        model_MGs["nx"] + i * NX + j + model_DC["nl"]] * x[
                        model_MGs["nx"] + i * NX + model_DC["f"][j] + 2 * model_DC["nl"]])

        obj = 0
        for i in range(nx):
            obj += Q_agg[i, 0] * x[i] * x[i] + c_agg[i, 0] * x[i] + c0_agg[i, 0]

        model.setObjective(obj)
        model.Params.OutputFlag = 0
        model.Params.LogToConsole = 0
        model.Params.DisplayInterval = 1
        model.optimize()

        xx = []
        for v in model.getVars():
            xx.append(v.x)

        obj = obj.getValue()
        primal_residual = zeros(model_DC["nl"] * T)

        for i in range(T):
            for j in range(model_DC["nl"]):
                primal_residual[i * model_DC["nl"] + j] = xx[model_MGs["nx"] + i * NX + j] * xx[
                    model_MGs["nx"] + i * NX + j] - xx[model_MGs["nx"] + i * NX + j + model_DC["nl"]] * xx[
                                                              model_MGs["nx"] + i * NX + int(model_DC["f"][j]) + 2 *
                                                              model_DC["nl"]]

        sol = {"x": 0}
        return sol

    def optimal_power_flow_microgrid(self, caseMGs, T):
        from distribution_system_optimization.data_format.idx_MGs_RO import PG, QG, BETA_PG, PUG, QUG, \
            BETA_UG, PBIC_AC2DC, PBIC_DC2AC, QBIC, PESS_C, PESS_DC, BETA_ESS, EESS, PMG, NX
        NMG = len(caseMGs)  # Number of hybrid AC/DC micro-grirds
        nx = NMG * T * NX
        # Boundary information
        lx = zeros((nx, 1))
        ux = zeros((nx, 1))
        for i in range(NMG):
            for j in range(T):
                # The lower boundary
                lx[i * T * NX + j * NX + PG] = caseMGs[i]["DG"]["PMIN"]
                lx[i * T * NX + j * NX + QG] = caseMGs[i]["DG"]["QMIN"]
                lx[i * T * NX + j * NX + BETA_PG] = 0
                lx[i * T * NX + j * NX + PUG] = caseMGs[i]["UG"]["PMIN"]
                lx[i * T * NX + j * NX + QUG] = caseMGs[i]["UG"]["QMIN"]
                lx[i * T * NX + j * NX + BETA_UG] = 0
                lx[i * T * NX + j * NX + PBIC_AC2DC] = 0
                lx[i * T * NX + j * NX + PBIC_DC2AC] = 0
                lx[i * T * NX + j * NX + QBIC] = -caseMGs[i]["BIC"]["SMAX"]
                lx[i * T * NX + j * NX + PESS_C] = 0
                lx[i * T * NX + j * NX + PESS_DC] = 0
                lx[i * T * NX + j * NX + BETA_ESS] = 0
                lx[i * T * NX + j * NX + EESS] = caseMGs[i]["ESS"]["SOC_MIN"] * caseMGs[i]["ESS"]["CAP"]
                lx[i * T * NX + j * NX + PMG] = -M
                # The upper boundary
                ux[i * T * NX + j * NX + PG] = caseMGs[i]["DG"]["PMAX"]
                ux[i * T * NX + j * NX + QG] = caseMGs[i]["DG"]["QMAX"]
                ux[i * T * NX + j * NX + BETA_PG] = 1
                ux[i * T * NX + j * NX + PUG] = caseMGs[i]["UG"]["PMAX"]
                ux[i * T * NX + j * NX + QUG] = caseMGs[i]["UG"]["QMAX"]
                ux[i * T * NX + j * NX + BETA_UG] = 1
                ux[i * T * NX + j * NX + PBIC_AC2DC] = caseMGs[i]["BIC"]["SMAX"]
                ux[i * T * NX + j * NX + PBIC_DC2AC] = caseMGs[i]["BIC"]["SMAX"]
                ux[i * T * NX + j * NX + QBIC] = caseMGs[i]["BIC"]["SMAX"]
                ux[i * T * NX + j * NX + PESS_C] = caseMGs[i]["ESS"]["PMAX_CH"]
                ux[i * T * NX + j * NX + PESS_DC] = caseMGs[i]["ESS"]["PMAX_DIS"]
                ux[i * T * NX + j * NX + BETA_ESS] = 1
                ux[i * T * NX + j * NX + EESS] = caseMGs[i]["ESS"]["SOC_MAX"] * caseMGs[i]["ESS"]["CAP"]
                ux[i * T * NX + j * NX + PMG] = M

        # The participating factors
        Aeq_beta = zeros((T * NMG, nx))
        beq_beta = ones((T * NMG, 1))
        for i in range(NMG):
            for j in range(T):
                Aeq_beta[i * T + j, i * T * NX + j * NX + BETA_ESS] = 1
                Aeq_beta[i * T + j, i * T * NX + j * NX + BETA_PG] = 1
                Aeq_beta[i * T + j, i * T * NX + j * NX + BETA_UG] = 1
        # AC bus power balance equation
        Aeq_power_balance_equation_AC = zeros((T * NMG, nx))
        beq_power_balance_equation_AC = zeros((T * NMG, 1))
        for i in range(NMG):
            for j in range(T):
                Aeq_power_balance_equation_AC[i * T + j, i * T * NX + j * NX + PG] = 1
                Aeq_power_balance_equation_AC[i * T + j, i * T * NX + j * NX + PUG] = 1
                Aeq_power_balance_equation_AC[i * T + j, i * T * NX + j * NX + PBIC_DC2AC] = caseMGs[i]["BIC"][
                    "EFF_DC2AC"]
                Aeq_power_balance_equation_AC[i * T + j, i * T * NX + j * NX + PBIC_AC2DC] = -1
                beq_power_balance_equation_AC[i * T + j] = caseMGs[i]["LOAD_AC"]["P"][j]
        # DC bus power balance equation
        Aeq_power_balance_equation_DC = zeros((T * NMG, nx))
        beq_power_balance_equation_DC = zeros((T * NMG, 1))
        for i in range(NMG):
            for j in range(T):
                Aeq_power_balance_equation_DC[i * T + j, i * T * NX + j * NX + PESS_DC] = 1
                Aeq_power_balance_equation_DC[i * T + j, i * T * NX + j * NX + PESS_C] = -1
                Aeq_power_balance_equation_DC[i * T + j, i * T * NX + j * NX + PBIC_DC2AC] = -1
                Aeq_power_balance_equation_DC[i * T + j, i * T * NX + j * NX + PBIC_AC2DC] = caseMGs[i]["BIC"][
                    "EFF_AC2DC"]
                Aeq_power_balance_equation_DC[i * T + j, i * T * NX + j * NX + PMG] = -1
                beq_power_balance_equation_DC[i * T + j] = caseMGs[i]["LOAD_DC"]["P"][j] - caseMGs[i]["PV"]["P"][j]
        # Energy storage system
        Aeq_energy_storage_system = zeros((T * NMG, nx))
        beq_energy_storage_system = zeros((T * NMG, 1))
        for i in range(NMG):
            for j in range(T - 1):
                Aeq_energy_storage_system[i * T + j, i * T * NX + (j + 1) * NX + EESS] = 1
                Aeq_energy_storage_system[i * T + j, i * T * NX + j * NX + EESS] = 1
                Aeq_energy_storage_system[i * T + j, i * T * NX + (j + 1) * NX + PESS_C] = caseMGs[i]["ESS"]["EFF_CH"]
                Aeq_energy_storage_system[i * T + j, i * T * NX + (j + 1) * NX + PESS_DC] = -caseMGs[i]["ESS"][
                    "EFF_DIS"]
        for i in range(NMG):
            Aeq_energy_storage_system[(i + 1 * T) - 1, i * T * NX + EESS] = 1
            Aeq_energy_storage_system[(i + 1 * T) - 1, i * T * NX + PESS_C] = -caseMGs[i]["ESS"]["EFF_CH"]
            Aeq_energy_storage_system[(i + 1 * T) - 1, i * T * NX + PESS_DC] = 1 / caseMGs[i]["ESS"][
                "EFF_DIS"]
            beq_energy_storage_system[(i + 1 * T) - 1] = caseMGs[i]["ESS"]["E0"]
        Aeq = vstack(
            [Aeq_power_balance_equation_AC, Aeq_power_balance_equation_DC, Aeq_energy_storage_system, Aeq_beta])
        beq = vstack(
            [beq_power_balance_equation_AC, beq_power_balance_equation_DC, beq_energy_storage_system, beq_beta])
        neq = len(beq)

        ## Inequality constraints
        # The ramp up and down constraint of diesel generators
        A_ramp_up = zeros((T * NMG, nx))
        b_ramp_up = zeros((T * NMG, 1))
        A_ramp_down = zeros((T * NMG, nx))
        b_ramp_down = zeros((T * NMG, 1))
        for i in range(NMG):
            for j in range(T - 1):
                A_ramp_up[i * T + j, i * T * NX + (j + 1) * NX + PG] = 1
                A_ramp_up[i * T + j, i * T * NX + j * NX + PG] = -1
                b_ramp_up[i * T + j] = caseMGs[i]["DG"]["RU"]

                A_ramp_down[i * T + j, i * T * NX + (j + 1) * NX + PG] = -1
                A_ramp_down[i * T + j, i * T * NX + j * NX + PG] = 1
                b_ramp_down[(i - 1) * T + j] = caseMGs[i]["DG"]["RD"]

        # Additional constraints on beta and set-points
        A_re_DG_up = zeros((T * NMG, nx))
        A_re_DG_down = zeros((T * NMG, nx))
        b_re_DG_up = zeros((T * NMG, 1))
        b_re_DG_down = zeros((T * NMG, 1))

        A_re_ESS_up = zeros((T * NMG, nx))
        A_re_ESS_down = zeros((T * NMG, nx))
        b_re_ESS_up = zeros((T * NMG, 1))
        b_re_ESS_down = zeros((T * NMG, 1))
        for i in range(NMG):
            for j in range(T):
                A_re_DG_up[i * T + j, i * T * NX + j * NX + PG] = -1
                A_re_DG_up[i * T + j, i * T * NX + j * NX + BETA_PG] = caseMGs[i]["PV"]["DELTA"][j] + \
                                                                       caseMGs[i]["LOAD_AC"]["DELTA"][j] + \
                                                                       caseMGs[i]["LOAD_DC"]["DELTA"][j]
                b_re_DG_up[i * T + j] = -caseMGs[i]["DG"]["PMIN"]

                A_re_DG_down[i * T + j, i * T * NX + j * NX + PG] = 1
                A_re_DG_down[i * T + j, i * T * NX + j * NX + BETA_PG] = caseMGs[i]["PV"]["DELTA"][j] + \
                                                                         caseMGs[i]["LOAD_AC"]["DELTA"][j] + \
                                                                         caseMGs[i]["LOAD_DC"]["DELTA"][j]
                b_re_DG_down[i * T + j] = caseMGs[i]["DG"]["PMAX"]

                A_re_ESS_up[i * T + j, i * T * NX + j * NX + EESS] = 1
                A_re_ESS_up[i * T + j, i * T * NX + j * NX + BETA_ESS] = caseMGs[i]["PV"]["DELTA"][j] + \
                                                                         caseMGs[i]["LOAD_AC"]["DELTA"][j] + \
                                                                         caseMGs[i]["LOAD_DC"]["DELTA"][j]
                b_re_ESS_up[i * T + j] = caseMGs[i]["ESS"]["SOC_MAX"] * caseMGs[i]["ESS"]["CAP"]

                A_re_ESS_down[i * T + j, i * T * NX + j * NX + EESS] = -1
                A_re_ESS_down[i * T + j, i * T * NX + j * NX + BETA_ESS] = caseMGs[i]["PV"]["DELTA"][j] + \
                                                                           caseMGs[i]["LOAD_AC"]["DELTA"][j] + \
                                                                           caseMGs[i]["LOAD_DC"]["DELTA"][j]
                b_re_ESS_down[i * T + j] = -caseMGs[i]["ESS"]["SOC_MIN"] * caseMGs[i]["ESS"]["CAP"]

        A = vstack([A_ramp_up, A_ramp_down, A_re_DG_up, A_re_DG_down, A_re_ESS_up, A_re_ESS_down])
        b = vstack([b_ramp_up, b_ramp_down, b_re_DG_up, b_re_DG_down, b_re_ESS_up, b_re_ESS_down])
        c = zeros((nx, 1))
        for i in range(NMG):
            for j in range(T):
                c[i * T * NX + j * NX + PUG] = caseMGs[i]["UG"]["C"][i]
                c[i * T * NX + j * NX + PG] = caseMGs[i]["DG"]["C"]
                c[i * T * NX + j * NX + PESS_DC] = caseMGs[i]["ESS"]["COST_DIS"]
                c[i * T * NX + j * NX + PESS_C] = caseMGs[i]["ESS"]["COST_DIS"]

        model = {"lx": lx,
                 "ux": ux,
                 "Aeq": Aeq,
                 "beq": beq,
                 "A": A,
                 "b": b,
                 "c": c,
                 "nx": nx,
                 "neq": len(beq),
                 "nineq": len(b),
                 "NX": NX,
                 "PMG": PMG}
        return model

    def optimal_power_flow_direct_current_networks(self, case, caseMGs):
        """
            :param case: The network information
            :param case: The microgrids information
            :return: formulated DC networks models
            """
        from pypower.idx_brch import F_BUS, T_BUS, BR_X, RATE_A
        from pypower.idx_bus import PD, VMAX, VMIN
        from pypower.idx_gen import GEN_BUS, PMAX, PMIN
        from pypower.ext2int import ext2int
        from scipy.sparse import csr_matrix as sparse
        from scipy.sparse import hstack, vstack, diags

        case = ext2int(case)
        baseMVA, bus, gen, branch, gencost = case["baseMVA"], case["bus"], case["gen"], case["branch"], case["gencost"]

        nb = shape(case['bus'])[0]  ## number of buses
        nl = shape(case['branch'])[0]  ## number of branches
        ng = shape(case['gen'])[0]  ## number of dispatchable injections
        nmg = len(caseMGs)
        mg_l = zeros(nmg)
        mg_u = zeros(nmg)
        for i in range(nmg):
            mg_l[i] = -2 * caseMGs[i]["BIC"]["SMAX"]
            mg_u[i] = 2 * caseMGs[i]["BIC"]["SMAX"]

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses

        index_MG = zeros(nmg)
        for i in range(nmg):
            index_MG[i] = caseMGs[i]["AREA_DC"] - 1
        i = range(nl)  ## double set of row indices
        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
        Cmg = sparse((ones(nmg), (index_MG, range(nmg))), (nb, nmg))
        # Modify the branch resistance
        Branch_R = branch[:, BR_X]
        for i in range(nl):
            if Branch_R[i] <= 0:
                Branch_R[i] = max(Branch_R)

        Cf = Cf.T
        Ct = Ct.T
        # Obtain the boundary information
        Slmax = branch[:, RATE_A]
        Pij_l = -Slmax
        Iij_l = zeros(nl)
        Vm_l = power(bus[:, VMIN], 2)
        Pg_l = gen[:, PMIN] / baseMVA

        Pij_u = Slmax
        Iij_u = Slmax
        # Vm_u = [max(turn_to_power(bus[:, VMAX], 2))] * nb
        Vm_u = power(bus[:, VMAX], 2)
        Pg_u = gen[:, PMAX] / baseMVA
        # Pg_l = -Pg_u

        # KCL equation
        Aeq_p = hstack([Ct - Cf, -diags(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg, Cmg])
        beq_p = bus[:, PD] / baseMVA
        beq_p = zeros(nb)
        Pg_l = zeros(ng)
        Pg_u = zeros(ng)

        # KVL equation
        Aeq_KVL = hstack(
            [-2 * diags(Branch_R), diags(power(Branch_R, 2)), Cf.T - Ct.T, zeros((nl, ng)), zeros((nl, nmg))])
        beq_KVL = zeros(nl)

        Aeq = vstack([Aeq_p, Aeq_KVL])
        Aeq = Aeq.todense()
        neq = shape(Aeq)[0]

        beq = zeros((neq, 1))
        beq[:, 0] = concatenate([beq_p, beq_KVL])

        nx = 2 * nl + nb + ng + nmg

        lx = zeros((nx, 1))
        ux = zeros((nx, 1))
        lx[:, 0] = concatenate([Pij_l, Iij_l, Vm_l, Pg_l, mg_l])
        ux[:, 0] = concatenate([Pij_u, Iij_u, Vm_u, Pg_u, mg_u])

        Q = zeros((nx, 1))
        c = zeros((nx, 1))
        c0 = zeros((nx, 1))
        for i in range(ng):
            Q[i + 2 * nl + nb, 0] = gencost[i, 4] * baseMVA * baseMVA
            c[i + 2 * nl + nb, 0] = gencost[i, 5] * baseMVA
            c0[i + 2 * nl + nb, 0] = gencost[i, 6]

        # for i in range(nl):
        #     c[nl + i, 0] = Branch_R[i]

        model = {"Q": Q,
                 "c": c,
                 "c0": c0,
                 "Aeq": Aeq,
                 "beq": beq,
                 "lx": lx,
                 "ux": ux,
                 "nx": nx,
                 "nb": nb,
                 "nl": nl,
                 "ng": ng,
                 "f": f,
                 "MGs": index_MG}

        return model


# def optimal_power_flow_solving(self):
#
# def optimal_power_flow_solving_result_check(self):


if __name__ == '__main__':
    T = 24
    NMG = 3
    # The test system for micro-grid systems
    # 1) Load profile within each MG
    # 1.1) Forecasting information
    Load_profile_MGs = array([
        [221, 219.7, 224.9, 221, 227.5, 240.5, 260, 315.9, 330.2, 338, 347.1, 336.7, 336.7, 331.5, 340.6, 344.5, 331.5,
         328.9, 325, 331.5, 338, 322.4, 296, 239.2],
        [114.15, 109.78, 107.6, 107.16, 110.07, 118.52, 132.79, 142.25, 144.29, 145.16, 145.6, 145.16, 144.44, 144,
         142.83, 141.38, 140.36, 139.19, 138.17, 137.59, 140.07, 138.32, 128.71, 119.68],
        [52, 50, 50, 51, 56, 63, 70, 75, 76, 80, 78, 74, 72, 72, 76, 80, 85, 88, 90, 87, 78, 71, 65, 56]
    ])
    # 1.2) Forecasting errors
    Delta_MGs = array([0.1, 0.1, 0.1]) * 2
    # 1.3) Information interval
    Load_profile_interval_MGs_min = zeros((NMG, T))
    Load_profile_interval_MGs_max = zeros((NMG, T))
    for i in range(NMG):
        Load_profile_interval_MGs_min[i, :] = Load_profile_MGs[i, :] * (1 - Delta_MGs[i])
        Load_profile_interval_MGs_max[i, :] = Load_profile_MGs[i, :] * (1 + Delta_MGs[i])
    # 2) PV information
    PV_profile_MGs_base = array(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])
    PV_cap = array([100, 50, 50])
    Delta_PV = [0.1, 0.1, 0.1] * 3
    PV_profile_MGs = zeros((NMG, T))
    PV_profile_MGs_min = zeros((NMG, T))
    PV_profile_MGs_max = zeros((NMG, T))
    for i in range(NMG):
        for j in range(T):
            PV_profile_MGs[i, j] = PV_profile_MGs_base[j] * PV_cap[i]
            PV_profile_MGs_min[i, j] = PV_profile_MGs_base[j] * PV_cap[i] * (1 - Delta_PV[i])
            PV_profile_MGs_max[i, j] = PV_profile_MGs_base[j] * PV_cap[i] * (1 + Delta_PV[i])

    # 3) Market prices information
    Price_Wholesale = array(
        [0.0484, 0.0446, 0.0437, 0.0445, 0.0518, 0.069, 0.0824, 0.0801, 0.088, 0.09, 0.0926, 0.0887, 0.0906, 0.0905,
         0.086, 0.0791, 0.0746, 0.0655, 0.0624, 0.0658, 0.0727, 0.0609, 0.0525, 0.0449])
    # 4) Schedulable resources information
    Pdg_max = array([200, 100, 100])
    Pdg_min = array([40, 20, 20])
    Qdg_max = array([200, 100, 100])
    Qdg_min = array([-200, -100, -100])
    Ru_dg = array([80, 40, 40])
    Rd_dg = array([80, 40, 40])
    Pbic_max = array([200, 100, 100])
    Pug_max = array([1000, 0, 0])
    Pug_min = array([0, 0, 0])
    Pmg_min = array([-50, -50, -50])
    Pmg_max = array([50, 50, 50])
    Pc_max = array([25, 25, 25])
    Capacity_ESS = array([50, 50, 50])
    Soc_max = array([1, 1, 1])
    Soc_min = array([0.1, 0.1, 0.1])
    eff_c = array([0.95, 0.95, 0.95])
    eff_dc = array([0.95, 0.95, 0.95])
    E0 = Capacity_ESS * 0.5
    eff_bic = array([0.95, 0.95, 0.95])
    C_dg = array([0.04335, 0.04554, 0.05154])
    Q_dg = array([0.01, 0.01, 0.01])
    C_ess = array([0.01, 0.01, 0.01])

    # 5) Generate information models for each MG
    # 5.1) MG queue for a cluster of MGs. Using list to store thees information.
    MG = []
    for i in range(NMG):
        DG_temp = {"PMAX": Pdg_max[i],
                   "PMIN": Pdg_min[i],
                   "QMIN": -Pdg_max[i],
                   "QMAX": Pdg_max[i],
                   "RU": Ru_dg[i],
                   "RD": Rd_dg[i],
                   "C": C_dg[i],
                   "Q": Q_dg[i]}
        UG_temp = {"PMAX": Pug_max[i],
                   "PMIN": Pug_min[i],
                   "QMIN": 0,
                   "QMAX": Pug_max[i],
                   "C": Price_Wholesale}
        BIC_temp = {"SMAX": Pbic_max[i],
                    "EFF_AC2DC": eff_bic[i],
                    "EFF_DC2AC": eff_bic[i], }
        ESS_temp = {"E0": E0[i],
                    "CAP": Capacity_ESS[i],
                    "SOC_MAX": 1,
                    "SOC_MIN": 0.1,
                    "PMAX_DIS": Pc_max[i],
                    "PMAX_CH": Pc_max[i],
                    "EFF_DIS": eff_dc[i],
                    "EFF_CH": eff_c[i],
                    "COST_DIS": C_ess[i],
                    "COST_CH": C_ess[i], }
        Load_ac_temp = {"P": Load_profile_MGs[i, :] / 2,
                        "DELTA": (Load_profile_interval_MGs_max[i, :] - Load_profile_interval_MGs_min[i, :]) / 2}
        Load_dc_temp = {"P": Load_profile_MGs[i, :] / 2,
                        "DELTA": (Load_profile_interval_MGs_max[i, :] - Load_profile_interval_MGs_min[i, :]) / 2}
        PV_temp = {"P": PV_profile_MGs[i, :],
                   "DELTA": PV_profile_MGs_max[i, :] - PV_profile_MGs_min[i, :]}
        MG_temp = {"DG": DG_temp,
                   "UG": UG_temp,
                   "BIC": BIC_temp,
                   "ESS": ESS_temp,
                   "LOAD_AC": Load_ac_temp,
                   "LOAD_DC": Load_dc_temp,
                   "PV": PV_temp,
                   "AREA_AC": i + 1,
                   "AREA_DC": i + 1}
        MG.append(MG_temp)
        del DG_temp, UG_temp, BIC_temp, ESS_temp, Load_ac_temp, Load_dc_temp, PV_temp, MG_temp

    # The test MG system
    caseMGs = MG
    # The test DC system
    caseDC = case30.case30()
    # The test AC system
    caseAC = case33.case33()

    mmDC = MultipleMicrogridsDirect_CurrentNetworks()
    sol = mmDC.run(case_MGs=caseMGs, case_DC_network=caseDC, case_AC_networks=caseAC, T=T)
