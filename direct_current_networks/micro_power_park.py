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

from copy import deepcopy

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
    Profile = array([
        [0.64, 0.63, 0.65, 0.64, 0.66, 0.69, 0.75, 0.91, 0.95, 0.97, 1.00, 0.97, 0.97, 0.95, 0.98, 0.99, 0.95, 0.95,
         0.94, 0.95, 0.97, 0.93, 0.85, 0.69],
        [0.78, 0.75, 0.74, 0.74, 0.75, 0.81, 0.91, 0.98, 0.99, 0.99, 1.00, 0.99, 0.99, 0.99, 0.98, 0.97, 0.96, 0.95,
         0.95, 0.95, 0.96, 0.95, 0.88, 0.82],
        [0.57, 0.55, 0.55, 0.56, 0.62, 0.70, 0.78, 0.83, 0.84, 0.89, 0.87, 0.82, 0.80, 0.80, 0.84, 0.89, 0.94, 0.98,
         1.00, 0.97, 0.87, 0.79, 0.72, 0.62]
    ])
    # import the information models of micro-grids
    micro_grid_1 = deepcopy(micro_grid)
    micro_grid_1["BUS"] = 2
    micro_grid_1["PD"]["AC_MAX"] = 100
    micro_grid_1["PD"]["DC_MAX"] = 100
    micro_grid_1["UG"]["PMIN"] = -100
    micro_grid_1["UG"]["PMAX"] = 100
    micro_grid_1["DG"]["PMAX"] = 100
    micro_grid_1["DG"]["COST_A"] = 0.015
    micro_grid_1["ESS"]["PDC_MAX"] = 100
    micro_grid_1["ESS"]["PCH_MAX"] = 100
    micro_grid_1["ESS"]["E0"] = 50
    micro_grid_1["ESS"]["EMIN"] = 10
    micro_grid_1["ESS"]["EMAX"] = 100
    micro_grid_1["BIC"]["PMAX"] = 100
    micro_grid_1["MG"]["PMAX"] = 500
    micro_grid_1["MG"]["PMIN"] = -500
    micro_grid_1["PD"]["AC"] = Profile[0] * micro_grid_1["PD"]["AC_MAX"]
    micro_grid_1["PD"]["DC"] = Profile[0] * micro_grid_1["PD"]["DC_MAX"]
    # micro_grid_1["MG"]["PMIN"] = 0
    # micro_grid_1["MG"]["PMAX"] = 0

    micro_grid_2 = deepcopy(micro_grid)
    micro_grid_2["BUS"] = 4
    micro_grid_2["PD"]["AC_MAX"] = 500
    micro_grid_2["PD"]["DC_MAX"] = 500
    micro_grid_2["UG"]["PMIN"] = -500
    micro_grid_2["UG"]["PMAX"] = 1000
    micro_grid_2["DG"]["PMAX"] = 500
    micro_grid_2["DG"]["COST_A"] = 0.01
    micro_grid_2["ESS"]["PDC_MAX"] = 500
    micro_grid_2["ESS"]["PCH_MAX"] = 500
    micro_grid_2["ESS"]["E0"] = 200
    micro_grid_2["ESS"]["EMIN"] = 100
    micro_grid_2["ESS"]["EMAX"] = 500
    micro_grid_2["BIC"]["PMAX"] = 500
    micro_grid_2["MG"]["PMAX"] = 500
    micro_grid_2["MG"]["PMIN"] = -500
    micro_grid_2["PD"]["AC"] = Profile[1] * micro_grid_2["PD"]["AC_MAX"]
    micro_grid_2["PD"]["DC"] = Profile[1] * micro_grid_2["PD"]["DC_MAX"]
    # micro_grid_2["MG"]["PMIN"] = 0
    # micro_grid_2["MG"]["PMAX"] = 0

    micro_grid_3 = deepcopy(micro_grid)
    micro_grid_3["BUS"] = 10
    micro_grid_3["PD"]["AC_MAX"] = 500
    micro_grid_3["PD"]["DC_MAX"] = 500
    micro_grid_3["UG"]["PMIN"] = -1000
    micro_grid_3["UG"]["PMAX"] = 1000
    micro_grid_3["DG"]["PMAX"] = 500
    micro_grid_3["DG"]["COST_A"] = 0.01
    micro_grid_3["ESS"]["PDC_MAX"] = 500
    micro_grid_3["ESS"]["PCH_MAX"] = 500
    micro_grid_3["ESS"]["E0"] = 200
    micro_grid_3["ESS"]["EMIN"] = 100
    micro_grid_3["ESS"]["EMAX"] = 500
    micro_grid_3["BIC"]["PMAX"] = 1000
    micro_grid_3["MG"]["PMAX"] = 1000
    micro_grid_3["MG"]["PMIN"] = -1000
    micro_grid_3["PD"]["AC"] = Profile[2] * micro_grid_3["PD"]["AC_MAX"]
    micro_grid_3["PD"]["DC"] = Profile[2] * micro_grid_3["PD"]["DC_MAX"]
    # micro_grid_3["MG"]["PMIN"] = 0
    # micro_grid_3["MG"]["PMAX"] = 0

    case_micro_grids = [micro_grid_1, micro_grid_2, micro_grid_3]
    # Test the unit commitment problem within each micro-grid
    nmg = len(case_micro_grids)
    # result = [0] * nmg
    model_micro_grids = [0] * nmg

    for i in range(nmg):
        model_micro_grids[i] = unit_commitment_power_park.micro_grid(case_micro_grids[i])
        # result[i] = milp(model_micro_grids[i]["c"], Aeq=model_micro_grids[i]["Aeq"], beq=model_micro_grids[i]["beq"],
        #                  A=model_micro_grids[i]["A"], b=model_micro_grids[i]["b"], xmin=model_micro_grids[i]["lb"],
        #                  xmax=model_micro_grids[i]["ub"], vtypes=model_micro_grids[i]["vtypes"])

    # check the network reconfiguration problem
    load_profile = array(
        [0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95,
         0.81, 0.59, 0.35, 0.14, 0.17, 0.41])
    load_profile = zeros(24)

    model_distribution_network = unit_commitment_power_park.problem_formulation(case=mpc, micro_grids=case_micro_grids,
                                                                                profile=load_profile)

    # result_distribution_network = milp(model_distribution_network["c"], Aeq=model_distribution_network["Aeq"],
    #                                    beq=model_distribution_network["beq"],
    #                                    A=model_distribution_network["A"], b=model_distribution_network["b"],
    #                                    xmin=model_distribution_network["lb"],
    #                                    xmax=model_distribution_network["ub"],
    #                                    vtypes=model_distribution_network["vtypes"])
    #
    #
    # print(result_distribution_network)
    # formulate connection matrix between the distribution network and micro-grids
    nVariables_distribution_network = len(model_distribution_network["c"])
    neq_distribution_network = model_distribution_network["Aeq"].shape[0]
    nineq_distribution_network = model_distribution_network["A"].shape[0]

    nVariables_micro_grid = zeros(nmg)
    neq_micro_grid = zeros(nmg)
    nineq_micro_grid = zeros(nmg)

    nVariables = int(nVariables_distribution_network)
    neq = int(neq_distribution_network)
    nineq = int(nineq_distribution_network)

    nVariables_index = zeros(nmg + 1)
    neq_index = zeros(nmg + 1)
    nineq_index = zeros(nmg + 1)

    nVariables_index[0] = int(nVariables_distribution_network)
    neq_index[0] = int(neq_distribution_network)
    nineq_index[0] = int(nineq_distribution_network)
    for i in range(nmg):
        nVariables_index[i + 1] = nVariables_index[i] + len(model_micro_grids[i]["c"])
        neq_index[i + 1] = neq_index[i] + model_micro_grids[0]["Aeq"].shape[0]
        nineq_index[i + 1] = nineq_index[i] + model_micro_grids[0]["A"].shape[0]
        nVariables += len(model_micro_grids[i]["c"])
        neq += int(model_micro_grids[0]["Aeq"].shape[0])
        nineq += int(model_micro_grids[0]["A"].shape[0])

    # Extract information from information models
    lx = model_distribution_network["lb"]
    ux = model_distribution_network["ub"]
    c = model_distribution_network["c"]
    vtypes = model_distribution_network["vtypes"]

    beq = model_distribution_network["beq"]
    b = model_distribution_network["b"]

    A = zeros((int(nineq_index[-1]), int(nVariables_index[-1])))
    Aeq = zeros((int(neq_index[-1]), int(nVariables_index[-1])))

    Aeq[0:neq_distribution_network, 0:nVariables_distribution_network] = model_distribution_network["Aeq"]
    A[0:nineq_distribution_network, 0:nVariables_distribution_network] = model_distribution_network["A"]

    for i in range(nmg):
        lx = concatenate([lx, model_micro_grids[i]["lb"]])
        ux = concatenate([ux, model_micro_grids[i]["ub"]])
        c = concatenate([c, model_micro_grids[i]["c"]])
        vtypes += model_micro_grids[i]["vtypes"]
        beq = concatenate([beq, model_micro_grids[i]["beq"]])
        b = concatenate([b, model_micro_grids[i]["b"]])
        Aeq[int(neq_index[i]):int(neq_index[i + 1]), int(nVariables_index[i]):int(nVariables_index[i + 1])] = \
            model_micro_grids[i]["Aeq"]
        A[int(nineq_index[i]):int(nineq_index[i + 1]), int(nVariables_index[i]):int(nVariables_index[i + 1])] = \
            model_micro_grids[i]["A"]

    # Add coupling constraints
    T = unit_commitment_power_park.T
    Ay2x = zeros((nmg * T, int(nVariables_index[-1] - nVariables_index[0])))
    for i in range(T):
        for j in range(nmg):
            Ay2x[i * nmg + j, int(nVariables_index[j] - nVariables_index[0]) + i * NX + PMG] = -1

    Aeq_temp = concatenate([model_distribution_network["Ax2y"], Ay2x], axis=1)
    beq_temp = zeros(nmg * T)

    Aeq = concatenate([Aeq, Aeq_temp])
    beq = concatenate([beq, beq_temp])

    model = Model("Centralized model")
    # # Define the decision variables
    x = {}
    for i in range(nVariables):
        if vtypes[i] == "c":
            x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)
        elif vtypes[i] == "b":
            x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.BINARY)

    neq = Aeq.shape[0]
    for i in range(neq):
        expr = 0
        for j in range(nVariables):
            expr += x[j] * Aeq[i, j]
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

    nineq = A.shape[0]
    for i in range(nineq):
        expr = 0
        for j in range(nVariables):
            expr += x[j] * A[i, j]
        model.addConstr(lhs=expr, sense=GRB.LESS_EQUAL, rhs=b[i])

    obj = 0
    for i in range(nVariables):
        obj += x[i] * c[i]

    # Add conic constraints
    nl = unit_commitment_power_park.nl
    f = model_distribution_network["f"]
    NX_dsn = model_distribution_network["NX"]
    for i in range(T):
        for j in range(nl):
            model.addConstr(
                x[3 * nl + i * NX_dsn + j] * x[3 * nl + i * NX_dsn + j] <= x[3 * nl + i * NX_dsn + j + nl] * x[
                    3 * nl + i * NX_dsn + f[j] + 2 * nl])

    model.setObjective(obj)
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    model.Params.DisplayInterval = 1
    model.Params.LogFile = ""

    model.optimize()
    obj = obj.getValue()

    xx = []
    for v in model.getVars():
        xx.append(v.x)
    xx = array(xx)  # convert the list to array
    # Obtain the solutions of distribution networks and micro-grids
    nb = unit_commitment_power_park.nb
    ng = unit_commitment_power_park.ng
    # 1) The network topology
    Alpha = xx[0:nl]
    Beta = xx[nl:2 * nl]
    Iij = xx[2 * nl:3 * nl]
    # 2) The distribution network operational plan
    Pij = zeros((nl, T))
    lij = zeros((nl, T))
    Vm = zeros((nb, T))
    Pg = zeros((ng, T))
    Pmg = zeros((nmg, T))
    for i in range(T):
        for j in range(nl):
            Pij[j, i] = xx[3 * nl + i * NX_dsn + j]
            lij[j, i] = xx[3 * nl + i * NX_dsn + nl + j]
        for j in range(nb):
            Vm[j, i] = xx[3 * nl + i * NX_dsn + 2 * nl + j]
        for j in range(ng):
            Pg[j, i] = xx[3 * nl + i * NX_dsn + 2 * nl + nb + j]
        for j in range(nmg):
            Pmg[j, i] = xx[3 * nl + i * NX_dsn + 2 * nl + nb + ng + j]
    # 3) The scheduling plan of each MG
    # 3.1) The energy storage system group
    Ich = zeros((nmg, T))
    Pess_dc = zeros((nmg, T))
    Pess_ch = zeros((nmg, T))
    Ress = zeros((nmg, T))
    Eess = zeros((nmg, T))
    # 3.2) The diesel generator group
    Ig = zeros((nmg, T))
    Pg_mg = zeros((nmg, T))
    Rg_mg = zeros((nmg, T))
    # 3.3) The utility grid group
    Iug_mg = zeros((nmg, T))
    Pug_mg = zeros((nmg, T))
    Rug_mg = zeros((nmg, T))
    # 3.4) Bi-directional converter group
    Ibic = zeros((nmg, T))
    Pbic_a2d = zeros((nmg, T))
    Pbic_d2a = zeros((nmg, T))
    # 3.5) Energy exchange part
    Pmg_mg = zeros((nmg, T))
    for i in range(T):
        for j in range(nmg):
            Ich[j, i] = xx[int(nVariables_index[j]) + i * NX + ICH]
            Pess_dc[j, i] = xx[int(nVariables_index[j]) + i * NX + PESS_DC]
            Pess_ch[j, i] = xx[int(nVariables_index[j]) + i * NX + PESS_CH]
            Ress[j, i] = xx[int(nVariables_index[j]) + i * NX + RESS]
            Eess[j, i] = xx[int(nVariables_index[j]) + i * NX + EESS]

            Ig[j, i] = xx[int(nVariables_index[j]) + i * NX + IG]
            Pg_mg[j, i] = xx[int(nVariables_index[j]) + i * NX + PG]
            Rg_mg[j, i] = xx[int(nVariables_index[j]) + i * NX + RG]

            Iug_mg[j, i] = xx[int(nVariables_index[j]) + i * NX + IUG]
            Pug_mg[j, i] = xx[int(nVariables_index[j]) + i * NX + PUG]
            Rug_mg[j, i] = xx[int(nVariables_index[j]) + i * NX + RUG]

            Ibic[j, i] = xx[int(nVariables_index[j]) + i * NX + IBIC_AC2DC]
            Pbic_a2d[j, i] = xx[int(nVariables_index[j]) + i * NX + PBIC_AC2DC]
            Pbic_d2a[j, i] = xx[int(nVariables_index[j]) + i * NX + PBIC_DC2AC]

            Pmg_mg[j, i] = xx[int(nVariables_index[j]) + i * NX + PMG]

    vol = zeros((nl, T))
    for i in range(T):
        for j in range(nl):
            if Alpha[j] > 0:
                vol[j, i] = Pij[j, i] ** 2 - Vm[int(f[j]), i] * lij[j, i]
    print(vol)
