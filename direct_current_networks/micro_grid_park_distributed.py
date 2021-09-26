"""
Distributed unit commitment for micro-gird power park
"""

from numpy import zeros, shape, ones, diag, concatenate, eye, array
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack
from numpy import flatnonzero as find
from gurobipy import *

from distribution_system_optimization.test_cases import case33
from micro_grids.test_cases.cases_unit_commitment import micro_grid

from pypower.idx_brch import F_BUS, T_BUS, BR_X, BR_STATUS, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, PD, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, PMAX, PMIN
from pypower.ext2int import ext2int

from unit_commitment.distributed_unit_commitment.idx_unit_commitment import ICH, IG, IUG, IBIC_AC2DC, \
    PBIC_AC2DC, PG, PESS_DC, PMG, PBIC_DC2AC, PUG, PESS_CH, RUG, RESS, RG, NX, EESS

from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp
from copy import deepcopy

from matplotlib import pyplot


class UnitCommitmentPowerPark():
    """
    Unit commitment for power park
    """

    def __init__(self):
        self.name = "Power park unit commitment"

    def problem_formulation(self, case, micro_grids, profile):
        """

        :param cases: Distribution network models
        :param micro_grids: Micro-grid models
        :param profile: Load-profile within the DC networks
        :return: Formulated centralized optimization problem
        """
        T = len(profile)
        self.T = T
        # Formulate the DC network reconfiguration
        case["branch"][:, BR_STATUS] = ones(case["branch"].shape[0])
        mpc = ext2int(case)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections
        nmg = len(micro_grids)

        self.nmg = nmg
        self.nl = nl
        self.nb = nb
        self.ng = ng

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        m = zeros(nmg)  ## list of integration index
        Pmg_l = zeros(nmg)  ## list of lower boundary
        Pmg_u = zeros(nmg)  ## list of upper boundary
        for i in range(nmg):
            m[i] = micro_grids[i]["BUS"]
            Pmg_l[i] = micro_grids[i]["MG"]["PMIN"]
            Pmg_u[i] = micro_grids[i]["MG"]["PMAX"]

        # Connection matrix
        i = range(nl)  ## double set of row indices
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
        Cmg = sparse((ones(nmg), (m, range(nmg))), (nb, nmg))

        Branch_R = branch[:, BR_X]
        Cf = Cf.T
        Ct = Ct.T
        # Obtain the boundary information
        Slmax = branch[:, RATE_A] / baseMVA

        Pij_l = -Slmax
        Iij_l = zeros(nl)
        Vm_l = bus[:, VMIN] ** 2
        Pg_l = gen[:, PMIN] / baseMVA
        Alpha_l = zeros(nl)
        Beta_f_l = zeros(nl)
        Beta_t_l = zeros(nl)

        Pij_u = Slmax
        Iij_u = Slmax
        Vm_u = bus[:, VMAX] ** 2
        Pg_u = 2 * gen[:, PMAX] / baseMVA
        Alpha_u = ones(nl)
        Beta_f_u = ones(nl)
        Beta_t_u = ones(nl)
        bigM = max(Vm_u)
        # For the spanning tree constraints
        Root_node = find(bus[:, BUS_TYPE] == REF)
        Root_line = find(branch[:, F_BUS] == Root_node)

        Span_f = zeros((nb, nl))
        Span_t = zeros((nb, nl))
        for i in range(nb):
            Span_f[i, find(branch[:, F_BUS] == i)] = 1
            Span_t[i, find(branch[:, T_BUS] == i)] = 1

        Alpha_l[Root_line] = 1
        Alpha_u[Root_line] = 1
        Beta_f_l[Root_line] = 0
        Beta_f_l[Root_line] = 0
        PIJ = 0
        IIJ = 1
        VM = 2
        # force the original network configuration
        # Alpha_l[0:nb-1] = 1
        # Alpha_u[0:nb-1] = 1

        nx = (2 * nl + nb + ng + nmg) * T + 3 * nl  ## Dimension of decision variables
        NX = 2 * nl + nb + ng + nmg
        # 1) Lower, upper and types of variables
        lx = zeros(nx)
        ux = zeros(nx)
        vtypes = ["c"] * nx
        c = zeros(nx)
        q = zeros(nx)

        lx[0:nl] = Alpha_l
        ux[0:nl] = Alpha_u
        lx[nl:2 * nl] = Beta_f_l
        ux[nl:2 * nl] = Beta_f_u
        vtypes[nl:2 * nl] = ["b"] * nl

        lx[2 * nl:3 * nl] = Beta_t_l
        ux[2 * nl:3 * nl] = Beta_t_u
        vtypes[2 * nl:3 * nl] = ["b"] * nl

        for i in range(T):
            # Upper boundary
            lx[3 * nl + i * NX + PIJ * nl:3 * nl + i * NX + IIJ * nl] = Pij_l
            lx[3 * nl + i * NX + IIJ * nl:3 * nl + i * NX + VM * nl] = Iij_l
            lx[3 * nl + i * NX + VM * nl:3 * nl + i * NX + VM * nl + nb] = Vm_l
            lx[3 * nl + i * NX + VM * nl + nb:3 * nl + i * NX + VM * nl + nb + ng] = Pg_l
            lx[3 * nl + i * NX + VM * nl + nb + ng:3 * nl + i * NX + VM * nl + nb + ng + nmg] = Pmg_l / baseMVA
            # Lower boundary
            ux[3 * nl + i * NX + PIJ * nl:3 * nl + i * NX + IIJ * nl] = Pij_u
            ux[3 * nl + i * NX + IIJ * nl:3 * nl + i * NX + VM * nl] = Iij_u
            ux[3 * nl + i * NX + VM * nl:3 * nl + i * NX + VM * nl + nb] = Vm_u
            ux[3 * nl + i * NX + VM * nl + nb:3 * nl + i * NX + VM * nl + nb + ng] = Pg_u
            ux[3 * nl + i * NX + VM * nl + nb + ng:3 * nl + i * NX + VM * nl + nb + ng + nmg] = Pmg_u / baseMVA
            # Cost
            c[3 * nl + i * NX + VM * nl + nb:3 * nl + i * NX + VM * nl + nb + ng] = gencost[:, 5] * baseMVA
            q[3 * nl + i * NX + VM * nl + nb:3 * nl + i * NX + VM * nl + nb + ng] = gencost[:, 4] * baseMVA * baseMVA

        # Formulate equal constraints
        ## 2) Equal constraints
        # 2.1) Alpha = Beta_f + Beta_t
        Aeq_f = zeros((nl, nx))
        beq_f = zeros(nl)
        Aeq_f[:, 0: nl] = -eye(nl)
        Aeq_f[:, nl:2 * nl] = eye(nl)
        Aeq_f[:, 2 * nl: 3 * nl] = eye(nl)
        # 2.2) sum(alpha)=nb-1
        Aeq_alpha = zeros((1, nx))
        beq_alpha = zeros(1)
        Aeq_alpha[0, 0:  nl] = ones(nl)
        beq_alpha[0] = nb - 1
        # 2.3) Span_f*Beta_f+Span_t*Beta_t = Spanning_tree
        Aeq_span = zeros((nb, nx))
        beq_span = ones(nb)
        beq_span[Root_node] = 0
        Aeq_span[:, nl:2 * nl] = Span_f
        Aeq_span[:, 2 * nl:3 * nl] = Span_t
        # 2.4) Power balance equation
        Aeq_p = hstack([Ct - Cf, -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg, Cmg])
        beq_p = bus[:, PD] / baseMVA
        Aeq_power_balance = zeros((nb * T, nx))
        beq_power_balance = zeros(nb * T)

        for i in range(T):
            Aeq_power_balance[i * nb:(i + 1) * nb, 3 * nl + i * NX: 3 * nl + (i + 1) * NX] = Aeq_p.toarray()
            beq_power_balance[i * nb:(i + 1) * nb] = beq_p * profile[i]

        Aeq = concatenate([Aeq_f, Aeq_alpha, Aeq_span, Aeq_power_balance])
        beq = concatenate([beq_f, beq_alpha, beq_span, beq_power_balance])

        ## 3) Inequality constraints
        # 3.1) Pij<=Alpha*Pij_max
        A_pij = zeros((nl * T, nx))
        b_pij = zeros(nl * T)
        for i in range(T):
            A_pij[i * nl:(i + 1) * nl, 3 * nl + i * NX + PIJ * nl:3 * nl + i * NX + (PIJ + 1) * nl] = eye(nl)
            A_pij[i * nl:(i + 1) * nl, 0: nl] = -diag(Pij_u)
        # 3.2) lij<=Alpha*lij_max
        A_lij = zeros((nl * T, nx))
        b_lij = zeros(nl * T)
        for i in range(T):
            A_lij[i * nl:(i + 1) * nl, 3 * nl + i * NX + IIJ * nl:3 * nl + i * NX + (IIJ + 1) * nl] = eye(nl)
            A_lij[i * nl:(i + 1) * nl, 0: nl] = -diag(Iij_u)
        # 3.3) KVL equation
        A_kvl = zeros((2 * nl * T, nx))
        b_kvl = zeros(2 * nl * T)
        for i in range(T):
            A_kvl[i * nl:(i + 1) * nl, 3 * nl + i * NX + PIJ * nl:3 * nl + i * NX + (PIJ + 1) * nl] = -2 * diag(
                Branch_R)
            A_kvl[i * nl:(i + 1) * nl, 3 * nl + i * NX + IIJ * nl:3 * nl + i * NX + (IIJ + 1) * nl] = diag(
                Branch_R ** 2)
            A_kvl[i * nl:(i + 1) * nl, 3 * nl + i * NX + VM * nl:3 * nl + i * NX + VM * nl + nb] = (
                    Cf.T - Ct.T).toarray()
            A_kvl[i * nl:(i + 1) * nl, 0:nl] = eye(nl) * bigM
            b_kvl[i * nl:(i + 1) * nl] = ones(nl) * bigM

            A_kvl[nl * T + i * nl:nl * T + (i + 1) * nl,
            3 * nl + i * NX + PIJ * nl:3 * nl + i * NX + (PIJ + 1) * nl] = 2 * diag(Branch_R)
            A_kvl[nl * T + i * nl:nl * T + (i + 1) * nl,
            3 * nl + i * NX + IIJ * nl:3 * nl + i * NX + (IIJ + 1) * nl] = -diag(Branch_R ** 2)
            A_kvl[nl * T + i * nl:nl * T + (i + 1) * nl, 3 * nl + i * NX + VM * nl:3 * nl + i * NX + VM * nl + nb] = -(
                    Cf.T - Ct.T).toarray()
            A_kvl[nl * T + i * nl:nl * T + (i + 1) * nl, 0:nl] = eye(nl) * bigM
            b_kvl[nl * T + i * nl:nl * T + (i + 1) * nl] = ones(nl) * bigM

        A = concatenate([A_pij, A_lij, A_kvl])
        b = concatenate([b_pij, b_lij, b_kvl])

        Ax2y = zeros((nmg * T, nx))  # The boundary information
        for i in range(T):
            for j in range(nmg):
                Ax2y[i * nmg + j, 3 * nl + i * NX + 2 * nl + nb + ng + j] = baseMVA

        ## For the microgrids
        model = {"c": c,
                 "lb": lx,
                 "ub": ux,
                 "vtypes": vtypes,
                 "A": A,
                 "b": b,
                 "Aeq": Aeq,
                 "beq": beq,
                 "Ax2y": Ax2y,
                 "f": f,
                 "NX": NX}

        #
        # model = Model("Network_reconfiguration")
        # # Define the decision variables
        # x = {}
        # nx = lx.shape[0]
        #
        # for i in range(nx):
        #     if vtypes[i] == "c":
        #         x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)
        #     elif vtypes[i] == "b":
        #         x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.BINARY)

        return model

    def micro_grid(self, micro_grid):
        """
        Unit commitment problem formulation of single micro_grid
        :param micro_grid:
        :return:
        """
        try:
            T = self.T
        except:
            T = 24
        nx = NX * T
        ## 1) boundary information and objective function
        lx = zeros(nx)
        ux = zeros(nx)
        c = zeros(nx)
        vtypes = ["c"] * nx
        for i in range(T):
            ## 1.1) lower boundary
            lx[i * NX + IG] = 0
            lx[i * NX + PG] = 0
            lx[i * NX + RG] = 0
            lx[i * NX + IUG] = 0
            lx[i * NX + PUG] = 0
            lx[i * NX + RUG] = 0
            lx[i * NX + IBIC_AC2DC] = 0
            lx[i * NX + PBIC_DC2AC] = 0
            lx[i * NX + PBIC_AC2DC] = 0
            lx[i * NX + ICH] = 0
            lx[i * NX + PESS_CH] = 0
            lx[i * NX + PESS_DC] = 0
            lx[i * NX + RESS] = 0
            lx[i * NX + EESS] = micro_grid["ESS"]["EMIN"]
            lx[i * NX + PMG] = micro_grid["MG"]["PMIN"]
            ## 1.2) upper boundary
            ux[i * NX + IG] = 1
            ux[i * NX + PG] = micro_grid["DG"]["PMAX"]
            ux[i * NX + RG] = micro_grid["DG"]["PMAX"]
            ux[i * NX + IUG] = 1
            ux[i * NX + PUG] = micro_grid["UG"]["PMAX"]
            ux[i * NX + RUG] = micro_grid["UG"]["PMAX"]
            ux[i * NX + IBIC_AC2DC] = 1
            ux[i * NX + PBIC_DC2AC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX + PBIC_AC2DC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX + ICH] = 1
            ux[i * NX + PESS_CH] = micro_grid["ESS"]["PCH_MAX"]
            ux[i * NX + PESS_DC] = micro_grid["ESS"]["PDC_MAX"]
            ux[i * NX + RESS] = micro_grid["ESS"]["PCH_MAX"] + micro_grid["ESS"]["PDC_MAX"]
            ux[i * NX + EESS] = micro_grid["ESS"]["EMAX"]
            ux[i * NX + PMG] = micro_grid["MG"]["PMAX"]
            ## 1.3) Objective functions
            c[i * NX + PG] = micro_grid["DG"]["COST_A"]
            c[i * NX + IG] = micro_grid["DG"]["COST_B"]
            c[i * NX + PUG] = micro_grid["UG"]["COST"][i]

            ## 1.4) Variable types
            vtypes[i * NX + IG] = "b"
            vtypes[i * NX + IUG] = "b"
            vtypes[i * NX + IBIC_AC2DC] = "b"
            vtypes[i * NX + ICH] = "b"
        # 2) Formulate the equal constraints
        # 2.1) Power balance equation
        # a) AC bus equation
        Aeq = zeros((T, nx))
        beq = zeros(T)
        for i in range(T):
            Aeq[i, i * NX + PG] = 1
            Aeq[i, i * NX + PUG] = 1
            Aeq[i, i * NX + PBIC_AC2DC] = -1
            Aeq[i, i * NX + PBIC_DC2AC] = micro_grid["BIC"]["EFF_DC2AC"]
            beq[i] = micro_grid["PD"]["AC"][i]
        # b) DC bus equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + PBIC_AC2DC] = micro_grid["BIC"]["EFF_AC2DC"]
            Aeq_temp[i, i * NX + PBIC_DC2AC] = -1
            Aeq_temp[i, i * NX + PESS_CH] = -1
            Aeq_temp[i, i * NX + PESS_DC] = 1
            Aeq_temp[i, i * NX + PMG] = -1
            beq_temp[i] = micro_grid["PD"]["DC"][i]
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # 2.2) Energy storage balance equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + EESS] = 1
            Aeq_temp[i, i * NX + PESS_CH] = -micro_grid["ESS"]["EFF_CH"]
            Aeq_temp[i, i * NX + PESS_DC] = 1 / micro_grid["ESS"]["EFF_DC"]
            if i == 0:
                beq_temp[i] = micro_grid["ESS"]["E0"]
            else:
                Aeq_temp[i, (i - 1) * NX + EESS] = -1
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # 3) Formualte inequal constraints
        # 3.1) Pg+Rg<=Ig*Pgmax
        A = zeros((T, nx))
        b = zeros(T)
        for i in range(T):
            A[i, i * NX + IG] = -micro_grid["DG"]["PMAX"]
            A[i, i * NX + PG] = 1
            A[i, i * NX + RG] = 1
        # 3.2) Pg-Rg>=Ig*Pgmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + IG] = micro_grid["DG"]["PMIN"]
            A_temp[i, i * NX + RG] = 1
            A_temp[i, i * NX + PG] = -1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.3) Pbic_ac2dc<=Ibic*Pbic_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + IBIC_AC2DC] = -micro_grid["BIC"]["PMAX"]
            A_temp[i, i * NX + PBIC_AC2DC] = 1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.4) Pbic_dc2sc<=(1-Ibic)*Pbic_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + IBIC_AC2DC] = micro_grid["BIC"]["PMAX"]
            A_temp[i, i * NX + PBIC_DC2AC] = 1
            b_temp[i] = micro_grid["BIC"]["PMAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.5) Pess_ch<=Ich*Pess_ch_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + ICH] = -micro_grid["ESS"]["PCH_MAX"]
            A_temp[i, i * NX + PESS_CH] = 1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.6) Pess_dc<=(1-Ich)*Pess_dc_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + ICH] = micro_grid["ESS"]["PDC_MAX"]
            A_temp[i, i * NX + PESS_DC] = 1
            b_temp[i] = micro_grid["ESS"]["PDC_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.7) Pess_dc-Pess_ch+Ress<=Pess_dc_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PESS_DC] = 1
            A_temp[i, i * NX + PESS_CH] = -1
            A_temp[i, i * NX + RESS] = 1
            b_temp[i] = micro_grid["ESS"]["PDC_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.8) Pess_ch-Pess_dc+Ress<=Pess_ch_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PESS_CH] = 1
            A_temp[i, i * NX + PESS_DC] = -1
            A_temp[i, i * NX + RESS] = 1
            b_temp[i] = micro_grid["ESS"]["PCH_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.9) Pug+Rug<=Iug*Pugmax
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + IUG] = -micro_grid["UG"]["PMAX"]
            A_temp[i, i * NX + PUG] = 1
            A_temp[i, i * NX + RUG] = 1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.10) Pug-Rug>=Iug*Pugmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + IUG] = micro_grid["DG"]["PMIN"]
            A_temp[i, i * NX + RUG] = 1
            A_temp[i, i * NX + PUG] = -1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])

        model_micro_grid = {"c": c,
                            "lb": lx,
                            "ub": ux,
                            "vtypes": vtypes,
                            "A": A,
                            "b": b,
                            "Aeq": Aeq,
                            "beq": beq,
                            "NX": NX}

        return model_micro_grid


if __name__ == "__main__":
    mpc = case33.case33()  # Default test case
    unit_commitment_power_park = UnitCommitmentPowerPark()

    # import the information models of micro-grids
    micro_grid_1 = deepcopy(micro_grid)
    micro_grid_1["PD"]["AC"] = micro_grid_1["PD"]["AC"] * micro_grid_1["PD"]["AC_MAX"]
    micro_grid_1["PD"]["DC"] = micro_grid_1["PD"]["DC"] * micro_grid_1["PD"]["DC_MAX"]
    micro_grid_1["BUS"] = 2
    # micro_grid_1["MG"]["PMIN"] = 0
    # micro_grid_1["MG"]["PMAX"] = 0

    micro_grid_2 = deepcopy(micro_grid)
    micro_grid_2["PD"]["AC"] = micro_grid_2["PD"]["AC"] * micro_grid_1["PD"]["AC_MAX"]
    micro_grid_2["PD"]["DC"] = micro_grid_2["PD"]["DC"] * micro_grid_2["PD"]["DC_MAX"]
    micro_grid_2["BUS"] = 4
    # micro_grid_2["MG"]["PMIN"] = 0
    # micro_grid_2["MG"]["PMAX"] = 0

    micro_grid_3 = deepcopy(micro_grid)
    micro_grid_3["PD"]["AC"] = micro_grid_3["PD"]["AC"] * micro_grid_3["PD"]["AC_MAX"]
    micro_grid_3["PD"]["DC"] = micro_grid_3["PD"]["DC"] * micro_grid_3["PD"]["DC_MAX"]
    micro_grid_3["BUS"] = 10
    # micro_grid_3["MG"]["PMIN"] = 0
    # micro_grid_3["MG"]["PMAX"] = 0

    case_micro_grids = [micro_grid_1, micro_grid_2, micro_grid_3]
    # Test the unit commitment problem within each micro-grid
    nmg = len(case_micro_grids)
    # result = [0] * nmg
    model_micro_grids = [0] * nmg
    load_profile = array(
        [0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95,
         0.81, 0.59, 0.35, 0.14, 0.17, 0.41])

    model_distribution_network = unit_commitment_power_park.problem_formulation(case=mpc, micro_grids=case_micro_grids,
                                                                                profile=load_profile)
    for i in range(nmg):
        model_micro_grids[i] = unit_commitment_power_park.micro_grid(case_micro_grids[i])

    nx = len(model_distribution_network["c"])
    neq_x = model_distribution_network["Aeq"].shape[0]
    nineq_x = model_distribution_network["A"].shape[0]

    nVariables_index = zeros(nmg + 1)
    neq_index = zeros(nmg + 1)
    nineq_index = zeros(nmg + 1)

    nVariables_index[0] = 0
    neq_index[0] = 0
    nineq_index[0] = 0

    ny = 0
    neq_y = 0
    nineq_y = 0

    for i in range(nmg):
        nVariables_index[i + 1] = nVariables_index[i] + len(model_micro_grids[i]["c"])
        neq_index[i + 1] = neq_index[i] + model_micro_grids[0]["Aeq"].shape[0]
        nineq_index[i + 1] = nineq_index[i] + model_micro_grids[0]["A"].shape[0]
        ny += len(model_micro_grids[i]["c"])
        neq_y += int(model_micro_grids[0]["Aeq"].shape[0])
        nineq_y += int(model_micro_grids[0]["A"].shape[0])

    # Extract information from information models
    lx = model_distribution_network["lb"]
    ux = model_distribution_network["ub"]
    cx = model_distribution_network["c"]
    vtypes_x = model_distribution_network["vtypes"]
    beq_x = model_distribution_network["beq"]
    b_x = model_distribution_network["b"]
    Aeq_x = model_distribution_network["Aeq"]
    A_x = model_distribution_network["A"]

    A_y = zeros((nineq_y, ny))
    Aeq_y = zeros((neq_y, ny))
    vtypes_y = []
    ly = zeros(0)
    uy = zeros(0)
    cy = zeros(0)
    beq_y = zeros(0)
    b_y = zeros(0)

    for i in range(nmg):
        ly = concatenate([ly, model_micro_grids[i]["lb"]])
        uy = concatenate([uy, model_micro_grids[i]["ub"]])
        cy = concatenate([cy, model_micro_grids[i]["c"]])
        vtypes_y += model_micro_grids[i]["vtypes"]
        beq_y = concatenate([beq_y, model_micro_grids[i]["beq"]])
        b_y = concatenate([b_y, model_micro_grids[i]["b"]])
        Aeq_y[int(neq_index[i]):int(neq_index[i + 1]), int(nVariables_index[i]):int(nVariables_index[i + 1])] = \
            model_micro_grids[i]["Aeq"]
        A_y[int(nineq_index[i]):int(nineq_index[i + 1]), int(nVariables_index[i]):int(nVariables_index[i + 1])] = \
            model_micro_grids[i]["A"]

    # Add coupling constraints
    T = unit_commitment_power_park.T
    Ay2x = zeros((nmg * T, ny))
    for i in range(T):
        for j in range(nmg):
            Ay2x[i * nmg + j, int(nVariables_index[j] - nVariables_index[0]) + i * NX + PMG] = -1

    #### ADMM solver ####
    step = 0.1
    lam = zeros(nmg * T)
    ru = 0.5
    x0 = zeros((nx, 1))
    y0 = zeros((ny, 1))

    # 1) Formulate the distribution network and microgrid cluster problem respectively
    model_dso = Model("Model_DSO")
    x = {}
    for i in range(nx):  # Decision making problem for the DSO
        if vtypes_x[i] == "c":
            x[i] = model_dso.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)
        elif vtypes_x[i] == "b":
            x[i] = model_dso.addVar(lb=lx[i], ub=ux[i], vtype=GRB.BINARY)

    for i in range(neq_x):
        expr = 0
        for j in range(nx):
            expr += x[j] * Aeq_x[i, j]
        model_dso.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq_x[i])

    for i in range(nineq_x):
        expr = 0
        for j in range(nx):
            expr += x[j] * A_x[i, j]
        model_dso.addConstr(lhs=expr, sense=GRB.LESS_EQUAL, rhs=b_x[i])

    nl = unit_commitment_power_park.nl
    f = model_distribution_network["f"]
    NX_dsn = model_distribution_network["NX"]
    for i in range(T):
        for j in range(nl):
            model_dso.addConstr(
                x[3 * nl + i * NX_dsn + j] * x[3 * nl + i * NX_dsn + j] <= x[3 * nl + i * NX_dsn + j + nl] * x[
                    3 * nl + i * NX_dsn + f[j] + 2 * nl])

    c_dso = cx
    c_dso = c_dso + model_distribution_network["Ax2y"].transpose().dot(lam) + ru * (
        (model_distribution_network["Ax2y"].transpose().dot(Ay2x)).dot(y0)).transpose()
    q_dso = model_distribution_network["Ax2y"].transpose().dot(model_distribution_network["Ax2y"]) * ru / 2

    # generate a list to store the non-zeros-index
    quadratic_item = []
    for i in range(nx):
        for j in range(nx):
            if q_dso[i, j] > 0:
                quadratic_item.append([i, j, q_dso[i, j]])

    obj_dso_linear = 0
    for i in range(nx):
        obj_dso_linear += c_dso[0][i] * x[i]  # only need to update this

    obj_dso_quadratic = 0
    for i in range(len(quadratic_item)):
        obj_dso_quadratic += quadratic_item[i][2] * x[quadratic_item[i][0]] * x[quadratic_item[i][1]]

    obj_dso = obj_dso_linear + obj_dso_quadratic
    model_dso.setObjective(obj_dso)
    model_dso.Params.OutputFlag = 0
    model_dso.Params.LogToConsole = 0
    model_dso.Params.DisplayInterval = 1
    model_dso.Params.MIPGap = 10 ** -2
    # model_dso.Params.Method = 2

    model_dso.Params.LogFile = " "

    model_dso.optimize()
    obj_dso = obj_dso.getValue()

    xx = []
    for v in model_dso.getVars():
        xx.append(v.x)
    xx = array(xx).reshape((nx, 1))  # convert the list to array
    # Calculate the real-objective function
    obj_dso_real = 0
    for i in range(nx):
        obj_dso_real += cx[i] * xx[i]

    # 2) Formulte the micro-girds problems
    model_mgs = Model("Model_Micro_grids")
    y = {}
    for i in range(ny):  # Decision making problem for the DSO
        if vtypes_y[i] == "c":
            y[i] = model_mgs.addVar(lb=ly[i], ub=uy[i], vtype=GRB.CONTINUOUS)
        elif vtypes_y[i] == "b":
            y[i] = model_mgs.addVar(lb=ly[i], ub=uy[i], vtype=GRB.BINARY)

    for i in range(neq_y):
        expr = 0
        for j in range(ny):
            if Aeq_y[i, j] != 0:
                expr += y[j] * Aeq_y[i, j]
        model_mgs.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq_y[i])

    for i in range(nineq_y):
        expr = 0
        for j in range(ny):
            if A_y[i, j] != 0:
                expr += y[j] * A_y[i, j]
        model_mgs.addConstr(lhs=expr, sense=GRB.LESS_EQUAL, rhs=b_y[i])

    q_mgs = Ay2x.transpose().dot(Ay2x) * ru / 2
    # generate a list to store the non-zeros-index
    quadratic_item_mgs = []
    for i in range(ny):
        for j in range(ny):
            if q_mgs[i, j] > 0:
                quadratic_item_mgs.append([i, j, q_mgs[i, j]])

    obj_mgs_quadratic = 0
    for i in range(len(quadratic_item_mgs)):
        obj_mgs_quadratic += quadratic_item_mgs[i][2] * y[quadratic_item_mgs[i][0]] * y[quadratic_item_mgs[i][1]]

    c_mgs = cy
    c_mgs = c_mgs + Ay2x.transpose().dot(lam) + ru * (
        (Ay2x.transpose().dot(model_distribution_network["Ax2y"])).dot(xx)).transpose()

    obj_mgs_linear = 0
    for i in range(ny):
        obj_mgs_linear += c_mgs[0][i] * y[i]  # only need to update this

    obj_mgs = obj_mgs_linear + obj_mgs_quadratic

    model_mgs.setObjective(obj_mgs)
    model_mgs.Params.OutputFlag = 0
    model_mgs.Params.LogToConsole = 0
    model_mgs.Params.DisplayInterval = 1
    model_mgs.Params.MIPGap = 10 ** -4
    model_mgs.optimize()
    obj_mgs = obj_mgs.getValue()

    yy = []
    for v in model_mgs.getVars():
        yy.append(v.x)
    yy = array(yy).reshape((ny, 1))  # convert the list to array
    # Calculate the real objective value
    obj_mgs_real = 0
    for i in range(ny):
        obj_mgs_real += cy[i] * yy[i]

    # Update the objective functions
    Iter = 0
    Iter_max = 1000
    obj_index = []
    while Iter <= Iter_max:
        # Update the lambda set
        lam += step * (model_distribution_network["Ax2y"].dot(xx[:, 0]) + Ay2x.dot(yy[:, 0]))
        #### Update the x part
        c_dso = cx
        c_dso = c_dso + model_distribution_network["Ax2y"].transpose().dot(lam) + ru * (
            (model_distribution_network["Ax2y"].transpose().dot(Ay2x)).dot(yy)).transpose()

        obj_dso_linear = 0
        for i in range(nx):
            obj_dso_linear += c_dso[0][i] * x[i]  # only need to update this

        obj_dso = obj_dso_linear + obj_dso_quadratic
        model_dso.setObjective(obj_dso)
        model_dso.optimize()
        xx = []
        for v in model_dso.getVars():
            xx.append(v.x)
        xx = array(xx).reshape((nx, 1))  # convert the list to array
        # Calculate the real-objective function
        obj_dso_real = 0
        for i in range(nx):
            obj_dso_real += cx[i] * xx[i]

        #### Update the y part
        c_mgs = cy
        c_mgs = c_mgs + Ay2x.transpose().dot(lam) + ru * (
            (Ay2x.transpose().dot(model_distribution_network["Ax2y"])).dot(xx)).transpose()

        obj_mgs_linear = 0
        for i in range(ny):
            obj_mgs_linear += c_mgs[0][i] * y[i]  # only need to update this

        obj_mgs = obj_mgs_linear + obj_mgs_quadratic
        model_mgs.setObjective(obj_mgs)
        model_mgs.optimize()
        yy = []
        for v in model_mgs.getVars():
            yy.append(v.x)
        yy = array(yy).reshape((ny, 1))  # convert the list to array
        # Calculate the real objective value
        obj_mgs_real = 0
        for i in range(ny):
            obj_mgs_real += cy[i] * yy[i]

        # Store the objective values
        obj_index.append(obj_mgs_real + obj_dso_real)
        print(obj_index[-1])
        Iter += 1

    print(yy)
