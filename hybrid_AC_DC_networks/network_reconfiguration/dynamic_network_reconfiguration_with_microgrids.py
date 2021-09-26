"""
Dynamic distribution network reconfiguration with microgrids

"""

from distribution_system_optimization.test_cases import case33
from micro_grids.test_cases.cases_unit_commitment import micro_grid

from scipy import zeros, shape, ones, diag, concatenate, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack
from numpy import flatnonzero as find
from numpy import array, tile

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_STATUS, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, PD, VMAX, VMIN, QD
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int

from solvers.mixed_integer_quadratic_constrained_cplex import mixed_integer_quadratic_constrained_programming as miqcp
from copy import deepcopy
from solvers.mixed_integer_quadratic_solver_cplex import mixed_integer_quadratic_programming as milp

from distribution_system_optimization.data_format.idx_mpp import ICH, IG, IUG, IBIC_AC2DC, \
    PBIC_AC2DC, PG, PESS_DC, PBIC_DC2AC, PUG, PESS_CH, RUG, RESS, RG, EESS, NX_MG, QBIC, QUG, QG


class NetworkReconfiguration():
    def __init__(self):
        self.name = "Network reconfiguration"

    def main(self, case, microgrids, profile):
        """
        Main entrance for network reconfiguration problems
        :param case: electric network information
        :param profile: load profile within the distribution networks
        :param micrgrids: dictionary for microgrids
        :return: network reconfiguration, distribution network status, and microgrid status
        """

        T = len(profile)
        self.T = T

        nmg = len(microgrids)
        self.nmg = nmg

        # 1) Formulate the constraints for each system
        # 1.1) Distribution networks
        model_distribution_networks = self.problem_formualtion_distribution_networks(case=case, profile=profile,
                                                                                     micro_grids=microgrids)
        # 1.2) Microgrids
        model_microgrids = {}
        for i in range(nmg):
            model_microgrids[i] = self.problem_formulation_microgrid(micro_grid=microgrids[i])

        # 2) System level modelling
        nVariables_distribution_network = len(model_distribution_networks["c"])
        neq_distribution_network = model_distribution_networks["Aeq"].shape[0]
        nineq_distribution_network = model_distribution_networks["A"].shape[0]

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
            nVariables_index[i + 1] = nVariables_index[i] + len(model_microgrids[i]["c"])
            neq_index[i + 1] = neq_index[i] + model_microgrids[i]["Aeq"].shape[0]
            nineq_index[i + 1] = nineq_index[i] + model_microgrids[i]["A"].shape[0]
            nVariables += len(model_microgrids[i]["c"])
            neq += int(model_microgrids[i]["Aeq"].shape[0])
            nineq += int(model_microgrids[i]["A"].shape[0])

        lx = model_distribution_networks["lb"]
        ux = model_distribution_networks["ub"]
        c = model_distribution_networks["c"]
        vtypes = model_distribution_networks["vtypes"]

        beq = model_distribution_networks["beq"]
        b = model_distribution_networks["b"]
        Qc = model_distribution_networks["Qc"]
        q = model_distribution_networks["q"]

        A = zeros((int(nineq_index[-1]), int(nVariables_index[-1])))
        Aeq = zeros((int(neq_index[-1]), int(nVariables_index[-1])))

        Aeq[0:neq_distribution_network, 0:nVariables_distribution_network] = model_distribution_networks["Aeq"]
        A[0:nineq_distribution_network, 0:nVariables_distribution_network] = model_distribution_networks["A"]

        for i in range(nmg):
            lx = concatenate([lx, model_microgrids[i]["lb"]])
            ux = concatenate([ux, model_microgrids[i]["ub"]])
            c = concatenate([c, model_microgrids[i]["c"]])
            q = concatenate([q, model_microgrids[i]["q"]])
            vtypes += model_microgrids[i]["vtypes"]
            beq = concatenate([beq, model_microgrids[i]["beq"]])
            b = concatenate([b, model_microgrids[i]["b"]])
            Aeq[int(neq_index[i]):int(neq_index[i + 1]), int(nVariables_index[i]):int(nVariables_index[i + 1])] = \
                model_microgrids[i]["Aeq"]
            A[int(nineq_index[i]):int(nineq_index[i + 1]), int(nVariables_index[i]):int(nVariables_index[i + 1])] = \
                model_microgrids[i]["A"]

        # Add coupling constraints, between the microgrids and distribution networks
        Ay2x = zeros((2 * nmg * T, int(nVariables_index[-1] - nVariables_index[0])))
        for i in range(T):
            for j in range(nmg):
                Ay2x[i * nmg + j, int(nVariables_index[j] - nVariables_index[0]) + i * NX_MG + PUG] = -1
                Ay2x[nmg * T + i * nmg + j, int(nVariables_index[j] - nVariables_index[0]) + i * NX_MG + QUG] = -1

        Aeq_temp = concatenate([model_distribution_networks["Ax2y"], Ay2x], axis=1)
        beq_temp = zeros(2 * nmg * T)

        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        # 3) Solve the problem
        (xx, obj, success) = miqcp(c, q, Aeq=Aeq, beq=beq, A=A, b=b, Qc=Qc, vtypes=vtypes, xmin=lx, xmax=ux)

        # 4) Check the solutions, including microgrids and distribution networks
        # 4.1) Scheduling plan of distribution networks
        sol_distribution_network = self.solution_check_distribution_network(xx[0:nVariables_distribution_network])
        # 4.2) Scheduling plan of each MG
        # a) Energy storage system group
        Ich = zeros((nmg, T))
        Pess_dc = zeros((nmg, T))
        Pess_ch = zeros((nmg, T))
        Ress = zeros((nmg, T))
        Eess = zeros((nmg, T))
        # b) Diesel generator group
        Ig = zeros((nmg, T))
        Pg = zeros((nmg, T))
        Qg = zeros((nmg, T))
        Rg = zeros((nmg, T))
        # c) Utility grid group
        Iug = zeros((nmg, T))
        Pug = zeros((nmg, T))
        Qug = zeros((nmg, T))
        Rug = zeros((nmg, T))
        # d) Bi-directional converter group
        Ibic = zeros((nmg, T))
        Pbic_a2d = zeros((nmg, T))
        Pbic_d2a = zeros((nmg, T))
        Qbic = zeros((nmg, T))
        for i in range(T):
            for j in range(nmg):
                Ich[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + ICH]
                Pess_dc[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PESS_DC]
                Pess_ch[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PESS_CH]
                Ress[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + RESS]
                Eess[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + EESS]

                Ig[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + IG]
                Pg[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PG]
                Qg[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + QG]
                Rg[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + RG]

                Iug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + IUG]
                Pug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PUG]
                Qug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + QUG]
                Rug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + RUG]

                Ibic[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + IBIC_AC2DC]
                Pbic_a2d[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PBIC_AC2DC]
                Pbic_d2a[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PBIC_DC2AC]
                Qbic[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + QBIC]

        sol_microgrids = {
            "ICH": Ich,
            "PESS_DC": Pess_dc,
            "PESS_CH": Pess_ch,
            "RESS": Ress,
            "EESS": Eess,
            "IG": Ig,
            "PG": Pg,
            "QG": Qg,
            "RG": Rg,
            "IUG": Iug,
            "PUG": Pug,
            "QUG": Qug,
            "RUG": Rug,
            "IBIC": Ibic,
            "PBIC_AC2DC": Pbic_a2d,
            "PBIC_DC2AC": Pbic_d2a,
            "QBIC": Qbic,
        }

        print(sol_microgrids["QUG"] - sol_distribution_network["Qmg"] * 100)
        print(sol_microgrids["PUG"] - sol_distribution_network["Pmg"] * 100)

        return sol_distribution_network, sol_microgrids

    def problem_formualtion_distribution_networks(self, case, profile, micro_grids):
        T = self.T

        case["branch"][:, BR_STATUS] = ones(case["branch"].shape[0])  # All lines are on-line.
        mpc = ext2int(case)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections
        nmg = self.nmg
        self.nl = nl
        self.nb = nb
        self.ng = ng

        m = zeros(nmg)  ## list of integration index
        Pmg_l = zeros(nmg)  ## list of lower boundary
        Pmg_u = zeros(nmg)  ## list of upper boundary
        Qmg_l = zeros(nmg)  ## list of lower boundary
        Qmg_u = zeros(nmg)  ## list of upper boundary
        for i in range(nmg):
            m[i] = micro_grids[i]["BUS"]
            Pmg_l[i] = micro_grids[i]["UG"]["PMIN"] / baseMVA
            Pmg_u[i] = micro_grids[i]["UG"]["PMAX"] / baseMVA
            Qmg_l[i] = micro_grids[i]["UG"]["QMIN"] / baseMVA
            Qmg_u[i] = micro_grids[i]["UG"]["QMAX"] / baseMVA

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = range(nl)  ## double set of row indices
        self.f = f
        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
        Cmg = sparse((ones(nmg), (m, range(nmg))), (nb, nmg))

        Branch_R = branch[:, BR_R]
        Branch_X = branch[:, BR_X]
        Cf = Cf.T
        Ct = Ct.T
        # Obtain the boundary information
        Slmax = branch[:, RATE_A] / baseMVA

        Pij_l = -Slmax
        Qij_l = -Slmax
        Iij_l = zeros(nl)
        Vm_l = bus[:, VMIN] ** 2
        Pg_l = gen[:, PMIN] / baseMVA
        Qg_l = gen[:, QMIN] / baseMVA
        Alpha_l = zeros(nl)
        Beta_f_l = zeros(nl)
        Beta_t_l = zeros(nl)

        Pij_u = Slmax
        Qij_u = Slmax
        Iij_u = Slmax
        Vm_u = bus[:, VMAX] ** 2
        Pg_u = 2 * gen[:, PMAX] / baseMVA
        Qg_u = 2 * gen[:, QMAX] / baseMVA
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

        nx = int(3 * nl + nb + 2 * ng + 2 * nmg)
        self.nx = nx
        lx = concatenate(
            [Alpha_l, Beta_f_l, Beta_t_l, tile(concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l, Pmg_l, Qmg_l]), T)])
        ux = concatenate(
            [Alpha_u, Beta_f_u, Beta_t_u, tile(concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u, Pmg_u, Qmg_u]), T)])

        vtypes = ["b"] * 2 * nl + ["c"] * nl + ["c"] * nx * T

        # Define the decision variables
        NX = lx.shape[0]

        # Alpha = Beta_f + Beta_t
        Aeq_f = zeros((nl, NX))
        beq_f = zeros(nl)
        Aeq_f[:, 0: nl] = -eye(nl)
        Aeq_f[:, nl: 2 * nl] = eye(nl)
        Aeq_f[:, 2 * nl:3 * nl] = eye(nl)

        # sum(alpha)=nb-1
        Aeq_alpha = zeros((1, NX))
        beq_alpha = zeros(1)
        Aeq_alpha[0, 0: nl] = ones(nl)
        beq_alpha[0] = nb - 1

        # Span_f*Beta_f+Span_t*Beta_t = Spanning_tree
        Aeq_span = zeros((nb, NX))
        beq_span = ones(nb)
        beq_span[Root_node] = 0
        Aeq_span[:, nl:2 * nl] = Span_f
        Aeq_span[:, 2 * nl: 3 * nl] = Span_t

        # Add system level constraints
        # 1) Active power balance
        Aeq_p = zeros((nb * T, NX))
        beq_p = zeros(nb * T)
        for i in range(T):
            Aeq_p[i * nb:(i + 1) * nb, 3 * nl + i * nx:3 * nl + (i + 1) * nx] = hstack([Ct - Cf, zeros((nb, nl)),
                                                                                        -diag(Ct * Branch_R) * Ct,
                                                                                        zeros((nb, nb)), Cg,
                                                                                        zeros((nb, ng)), -Cmg,
                                                                                        zeros((nb, nmg))]).toarray()
            beq_p[i * nb:(i + 1) * nb] = profile[i] * bus[:, PD] / baseMVA

        # 2) Reactive power balance
        Aeq_q = zeros((nb * T, NX))
        beq_q = zeros(nb * T)
        for i in range(T):
            Aeq_q[i * nb:(i + 1) * nb, 3 * nl + i * nx:3 * nl + (i + 1) * nx] = hstack([zeros((nb, nl)), Ct - Cf,
                                                                                        -diag(Ct * Branch_X) * Ct,
                                                                                        zeros((nb, nb)),
                                                                                        zeros((nb, ng)), Cg,
                                                                                        zeros((nb, nmg)),
                                                                                        -Cmg]).toarray()
            beq_q[i * nb:(i + 1) * nb] = profile[i] * bus[:, QD] / baseMVA

        Aeq = vstack([Aeq_f, Aeq_alpha, Aeq_span, Aeq_p, Aeq_q]).toarray()
        beq = concatenate([beq_f, beq_alpha, beq_span, beq_p, beq_q])

        # Current constraint
        A = zeros((nl * T, NX))
        b = zeros(nl * T)
        for i in range(T):
            A[i * nl:(i + 1) * nl, 3 * nl + i * nx + 2 * nl: 3 * nl + i * nx + 3 * nl] = eye(nl)
            A[i * nl:(i + 1) * nl, 0: nl] = -diag(Iij_u)
        # Real power flow constraint
        A_temp = zeros((nl * T, NX))
        b_temp = zeros(nl * T)
        for i in range(T):
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx: 3 * nl + i * nx + nl] = eye(nl)
            A_temp[i * nl:(i + 1) * nl, 0: nl] = -diag(Pij_u)
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # Reactive power flow constraint
        A_temp = zeros((nl * T, NX))
        b_temp = zeros(nl * T)
        for i in range(T):
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx + nl: 3 * nl + i * nx + 2 * nl] = eye(nl)
            A_temp[i * nl:(i + 1) * nl, 0:nl] = -diag(Qij_u)
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # KCL equation
        A_temp = zeros((nl * T, NX))
        for i in range(T):
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx:3 * nl + i * nx + nl] = -2 * diag(Branch_R)
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx + nl:3 * nl + i * nx + 2 * nl] = -2 * diag(Branch_X)
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx + 2 * nl:3 * nl + i * nx + 3 * nl] = diag(Branch_R ** 2) + \
                                                                                             diag(Branch_X ** 2)
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx + 3 * nl:3 * nl + i * nx + 3 * nl + nb] = \
                (Cf.T - Ct.T).toarray()
            A_temp[i * nl:(i + 1) * nl, 0:nl] = eye(nl) * bigM
        b_temp = ones(nl * T) * bigM
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])

        A_temp = zeros((nl * T, NX))
        for i in range(T):
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx:3 * nl + i * nx + nl] = 2 * diag(Branch_R)
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx + nl:3 * nl + i * nx + 2 * nl] = 2 * diag(Branch_X)
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx + 2 * nl:3 * nl + i * nx + 3 * nl] = -diag(Branch_R ** 2) - \
                                                                                             diag(Branch_X ** 2)
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx + 3 * nl:3 * nl + i * nx + 3 * nl + nb] = \
                (-Cf.T + Ct.T).toarray()
            A_temp[i * nl:(i + 1) * nl, 0: nl] = eye(nl) * bigM
        b_temp = ones(nl * T) * bigM
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # Pij**2+Qij**2<=Vi*Iij
        Qc = dict()
        for t in range(T):
            for i in range(nl):
                Qc[t * nl + i] = [[int(3 * nl + t * nx + i), int(3 * nl + t * nx + i + nl),
                                   int(3 * nl + t * nx + i + 2 * nl), int(3 * nl + t * nx + f[i] + 3 * nl)],
                                  [int(3 * nl + t * nx + i), int(3 * nl + t * nx + i + nl),
                                   int(3 * nl + t * nx + f[i] + 3 * nl), int(3 * nl + t * nx + i + 2 * nl)],
                                  [1, 1, -1 / 2, -1 / 2]]
        c = zeros(NX)
        q = zeros(NX)
        c0 = 0
        for t in range(T):
            for i in range(ng):
                c[3 * nl + t * nx + i + 3 * nl + nb] = gencost[i, 5] * baseMVA
                q[3 * nl + t * nx + i + 3 * nl + nb] = gencost[i, 4] * baseMVA * baseMVA
                c0 += gencost[i, 6]

        # The boundary information
        Ax2y = zeros((2 * nmg * T, NX))  # The boundary information
        for i in range(T):
            for j in range(nmg):
                Ax2y[i * nmg + j, 3 * nl + i * nx + 3 * nl + nb + 2 * ng + j] = baseMVA  # Active power
                Ax2y[nmg * T + i * nmg + j,
                     3 * nl + i * nx + 3 * nl + nb + 2 * ng + nmg + j] = baseMVA  # Reactive power

        # sol = miqcp(c, q, Aeq=Aeq, beq=beq, A=A, b=b, Qc=Qc, vtypes=vtypes, xmin=lx, xmax=ux)

        model_distribution_grid = {"c": c,
                                   "q": q,
                                   "lb": lx,
                                   "ub": ux,
                                   "vtypes": vtypes,
                                   "A": A,
                                   "b": b,
                                   "Aeq": Aeq,
                                   "beq": beq,
                                   "Qc": Qc,
                                   "c0": c0,
                                   "Ax2y": Ax2y}

        return model_distribution_grid

    def solution_check_distribution_network(self, xx):
        """
        solution check for distribution networks solution
        :param xx:
        :return:
        """
        nl = self.nl
        nb = self.nb
        ng = self.ng
        T = self.T
        nx = self.nx
        nmg = self.nmg
        f = self.f

        Alpha = xx[0:nl]
        Beta_f = xx[nl:2 * nl]
        Beta_t = xx[2 * nl:3 * nl]

        Pij = zeros((nl, T))
        Qij = zeros((nl, T))
        Iij = zeros((nl, T))
        Vi = zeros((nb, T))
        Pg = zeros((ng, T))
        Qg = zeros((ng, T))
        Pmg = zeros((nmg, T))
        Qmg = zeros((nmg, T))
        for i in range(T):
            Pij[:, i] = xx[3 * nl + i * nx:3 * nl + i * nx + nl]
            Qij[:, i] = xx[3 * nl + i * nx + nl:3 * nl + i * nx + 2 * nl]
            Iij[:, i] = xx[3 * nl + i * nx + 2 * nl:3 * nl + i * nx + 3 * nl]
            Vi[:, i] = xx[3 * nl + i * nx + 3 * nl:3 * nl + i * nx + 3 * nl + nb]
            Pg[:, i] = xx[3 * nl + i * nx + 3 * nl + nb:3 * nl + i * nx + 3 * nl + nb + ng]
            Qg[:, i] = xx[3 * nl + i * nx + 3 * nl + nb + ng:3 * nl + i * nx + 3 * nl + nb + 2 * ng]
            Pmg[:, i] = xx[3 * nl + i * nx + 3 * nl + nb + 2 * ng:3 * nl + i * nx + 3 * nl + nb + 2 * ng + nmg]
            Qmg[:, i] = xx[3 * nl + i * nx + 3 * nl + nb + 2 * ng + nmg:
                           3 * nl + i * nx + 3 * nl + nb + 2 * ng + 2 * nmg]

        primal_residual = zeros((nl, T))
        for t in range(T):
            for i in range(nl):
                primal_residual[i, t] = Pij[i, t] * Pij[i, t] + Qij[i, t] * Qij[i, t] - Iij[i, t] * Vi[int(f[i]), t]

        sol = {"Pij": Pij,
               "Qij": Qij,
               "Iij": Iij,
               "Vi": Vi,
               "Pg": Pg,
               "Qg": Qg,
               "Pmg": Pmg,
               "Qmg": Qmg,
               "Alpha": Alpha,
               "Beta_f": Beta_f,
               "Beta_t": Beta_t,
               "residual": primal_residual}

        return sol

    def problem_formulation_microgrid(self, micro_grid):
        """
        Unit commitment problem formulation of single micro_grid
        :param micro_grid:
        :return:
        """

        try:
            T = self.T
        except:
            T = 24

        ## 1) boundary information and objective function
        nx = NX_MG * T
        lx = zeros(nx)
        ux = zeros(nx)
        c = zeros(nx)
        q = zeros(nx)
        vtypes = ["c"] * nx
        for i in range(T):
            ## 1.1) lower boundary
            lx[i * NX_MG + IG] = 0
            lx[i * NX_MG + PG] = 0
            lx[i * NX_MG + QG] = micro_grid["DG"]["QMIN"]
            lx[i * NX_MG + RG] = 0
            lx[i * NX_MG + IUG] = 0
            lx[i * NX_MG + PUG] = 0
            lx[i * NX_MG + QUG] = micro_grid["UG"]["QMIN"]
            lx[i * NX_MG + RUG] = 0
            lx[i * NX_MG + IBIC_AC2DC] = 0
            lx[i * NX_MG + PBIC_DC2AC] = 0
            lx[i * NX_MG + PBIC_AC2DC] = 0
            lx[i * NX_MG + QBIC] = -micro_grid["BIC"]["SMAX"]
            lx[i * NX_MG + ICH] = 0
            lx[i * NX_MG + PESS_CH] = 0
            lx[i * NX_MG + PESS_DC] = 0
            lx[i * NX_MG + RESS] = 0
            lx[i * NX_MG + EESS] = micro_grid["ESS"]["EMIN"]

            ## 1.2) upper boundary
            ux[i * NX_MG + IG] = 1
            ux[i * NX_MG + PG] = micro_grid["DG"]["PMAX"]
            ux[i * NX_MG + QG] = micro_grid["DG"]["QMAX"]
            ux[i * NX_MG + RG] = micro_grid["DG"]["PMAX"]
            ux[i * NX_MG + IUG] = 1
            ux[i * NX_MG + PUG] = micro_grid["UG"]["PMAX"]
            ux[i * NX_MG + QUG] = micro_grid["UG"]["QMAX"]
            ux[i * NX_MG + RUG] = micro_grid["UG"]["PMAX"]
            ux[i * NX_MG + IBIC_AC2DC] = 1
            ux[i * NX_MG + PBIC_DC2AC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX_MG + PBIC_AC2DC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX_MG + QBIC] = micro_grid["BIC"]["SMAX"]
            ux[i * NX_MG + ICH] = 1
            ux[i * NX_MG + PESS_CH] = micro_grid["ESS"]["PCH_MAX"]
            ux[i * NX_MG + PESS_DC] = micro_grid["ESS"]["PDC_MAX"]
            ux[i * NX_MG + RESS] = micro_grid["ESS"]["PCH_MAX"] + micro_grid["ESS"]["PDC_MAX"]
            ux[i * NX_MG + EESS] = micro_grid["ESS"]["EMAX"]

            ## 1.3) Objective functions
            c[i * NX_MG + PG] = micro_grid["DG"]["COST_A"]
            c[i * NX_MG + IG] = micro_grid["DG"]["COST_B"]
            # c[i * NX_MG + PUG] = micro_grid["UG"]["COST"][i]
            ## 1.4) Variable types
            vtypes[i * NX_MG + IG] = "b"
            vtypes[i * NX_MG + IUG] = "b"
            vtypes[i * NX_MG + IBIC_AC2DC] = "b"
            vtypes[i * NX_MG + ICH] = "b"
            if i == T:
                lx[i * NX_MG + EESS] = micro_grid["ESS"]["E0"]
                ux[i * NX_MG + EESS] = micro_grid["ESS"]["E0"]

        # 2) Formulate the equal constraints
        # 2.1) Power balance equation
        # a) AC bus equation
        Aeq = zeros((T, nx))
        beq = zeros(T)
        for i in range(T):
            Aeq[i, i * NX_MG + PG] = 1
            Aeq[i, i * NX_MG + PUG] = 1
            Aeq[i, i * NX_MG + PBIC_AC2DC] = -1
            Aeq[i, i * NX_MG + PBIC_DC2AC] = micro_grid["BIC"]["EFF_DC2AC"]
            beq[i] = micro_grid["PD"]["AC"][i]
        # b) DC bus equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX_MG + PBIC_AC2DC] = micro_grid["BIC"]["EFF_AC2DC"]
            Aeq_temp[i, i * NX_MG + PBIC_DC2AC] = -1
            Aeq_temp[i, i * NX_MG + PESS_CH] = -1
            Aeq_temp[i, i * NX_MG + PESS_DC] = 1
            beq_temp[i] = micro_grid["PD"]["DC"][i]
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # c) AC reactive power balance equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX_MG + QUG] = 1
            Aeq_temp[i, i * NX_MG + QBIC] = 1
            Aeq_temp[i, i * NX_MG + QG] = 1
            beq_temp[i] = micro_grid["QD"]["AC"][i]
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        # 2.2) Energy storage balance equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX_MG + EESS] = 1
            Aeq_temp[i, i * NX_MG + PESS_CH] = -micro_grid["ESS"]["EFF_CH"]
            Aeq_temp[i, i * NX_MG + PESS_DC] = 1 / micro_grid["ESS"]["EFF_DC"]
            if i == 0:
                beq_temp[i] = micro_grid["ESS"]["E0"]
            else:
                Aeq_temp[i, (i - 1) * NX_MG + EESS] = -1
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # 3) Formualte inequal constraints
        # 3.1) Pg+Rg<=Ig*Pgmax
        A = zeros((T, nx))
        b = zeros(T)
        for i in range(T):
            A[i, i * NX_MG + IG] = -micro_grid["DG"]["PMAX"]
            A[i, i * NX_MG + PG] = 1
            A[i, i * NX_MG + RG] = 1
        # 3.2) Pg-Rg>=Ig*Pgmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + IG] = micro_grid["DG"]["PMIN"]
            A_temp[i, i * NX_MG + RG] = 1
            A_temp[i, i * NX_MG + PG] = -1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.3) Qg<=Ig*Qgmax
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + IG] = -micro_grid["DG"]["QMAX"]
            A_temp[i, i * NX_MG + QG] = 1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.4) -Qg<=-Ig*Qgmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + IG] = micro_grid["DG"]["QMIN"]
            A_temp[i, i * NX_MG + QG] = -1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.5) Pbic_ac2dc<=Ibic*Pbic_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + IBIC_AC2DC] = -micro_grid["BIC"]["PMAX"]
            A_temp[i, i * NX_MG + PBIC_AC2DC] = 1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.6) Pbic_dc2sc<=(1-Ibic)*Pbic_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + IBIC_AC2DC] = micro_grid["BIC"]["PMAX"]
            A_temp[i, i * NX_MG + PBIC_DC2AC] = 1
            b_temp[i] = micro_grid["BIC"]["PMAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.7) Pess_ch<=Ich*Pess_ch_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + ICH] = -micro_grid["ESS"]["PCH_MAX"]
            A_temp[i, i * NX_MG + PESS_CH] = 1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.8) Pess_dc<=(1-Ich)*Pess_dc_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + ICH] = micro_grid["ESS"]["PDC_MAX"]
            A_temp[i, i * NX_MG + PESS_DC] = 1
            b_temp[i] = micro_grid["ESS"]["PDC_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.9) Pess_dc-Pess_ch+Ress<=Pess_dc_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + PESS_DC] = 1
            A_temp[i, i * NX_MG + PESS_CH] = -1
            A_temp[i, i * NX_MG + RESS] = 1
            b_temp[i] = micro_grid["ESS"]["PDC_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.10) Pess_ch-Pess_dc+Ress<=Pess_ch_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + PESS_CH] = 1
            A_temp[i, i * NX_MG + PESS_DC] = -1
            A_temp[i, i * NX_MG + RESS] = 1
            b_temp[i] = micro_grid["ESS"]["PCH_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.11) Pug+Rug<=Iug*Pugmax
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + IUG] = -micro_grid["UG"]["PMAX"]
            A_temp[i, i * NX_MG + PUG] = 1
            A_temp[i, i * NX_MG + RUG] = 1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.12) Pug-Rug>=Iug*Pugmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + IUG] = micro_grid["DG"]["PMIN"]
            A_temp[i, i * NX_MG + RUG] = 1
            A_temp[i, i * NX_MG + PUG] = -1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.13) Qug<=Iug*Qugmax
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + IUG] = -micro_grid["UG"]["QMAX"]
            A_temp[i, i * NX_MG + QUG] = 1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.14) -Qug<=-Iug*Pugmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + IUG] = micro_grid["UG"]["QMIN"]
            A_temp[i, i * NX_MG + QUG] = -1
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # sol = milp(c, q=q, Aeq=Aeq, beq=beq, A=A, b=b, vtypes=vtypes, xmin=lx, xmax=ux)

        model_micro_grid = {"c": c,
                            "q": q,
                            "lb": lx,
                            "ub": ux,
                            "vtypes": vtypes,
                            "A": A,
                            "b": b,
                            "Aeq": Aeq,
                            "beq": beq,
                            "NX": NX_MG,
                            "PG": PG,
                            "QG": QG}

        return model_micro_grid


if __name__ == "__main__":
    # Distribution network information
    mpc = case33.case33()  # Default test case
    load_profile = array(
        [0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95,
         0.81, 0.59, 0.35, 0.14, 0.17, 0.41])

    # Microgrid information
    Profile = array([
        [0.64, 0.63, 0.65, 0.64, 0.66, 0.69, 0.75, 0.91, 0.95, 0.97, 1.00, 0.97, 0.97, 0.95, 0.98, 0.99, 0.95, 0.95,
         0.94, 0.95, 0.97, 0.93, 0.85, 0.69],
        [0.78, 0.75, 0.74, 0.74, 0.75, 0.81, 0.91, 0.98, 0.99, 0.99, 1.00, 0.99, 0.99, 0.99, 0.98, 0.97, 0.96, 0.95,
         0.95, 0.95, 0.96, 0.95, 0.88, 0.82],
        [0.57, 0.55, 0.55, 0.56, 0.62, 0.70, 0.78, 0.83, 0.84, 0.89, 0.87, 0.82, 0.80, 0.80, 0.84, 0.89, 0.94, 0.98,
         1.00, 0.97, 0.87, 0.79, 0.72, 0.62]
    ])
    micro_grid_1 = deepcopy(micro_grid)
    micro_grid_1["BUS"] = 2
    micro_grid_1["PD"]["AC_MAX"] = 10
    micro_grid_1["PD"]["DC_MAX"] = 10
    micro_grid_1["UG"]["PMIN"] = -500
    micro_grid_1["UG"]["PMAX"] = 500
    micro_grid_1["UG"]["QMIN"] = -500
    micro_grid_1["UG"]["QMAX"] = 500
    micro_grid_1["DG"]["PMAX"] = 100
    micro_grid_1["DG"]["QMAX"] = 100
    micro_grid_1["DG"]["QMIN"] = -100
    micro_grid_1["DG"]["COST_A"] = 0.015
    micro_grid_1["ESS"]["PDC_MAX"] = 50
    micro_grid_1["ESS"]["PCH_MAX"] = 50
    micro_grid_1["ESS"]["E0"] = 50
    micro_grid_1["ESS"]["EMIN"] = 10
    micro_grid_1["ESS"]["EMAX"] = 100
    micro_grid_1["BIC"]["PMAX"] = 100
    micro_grid_1["BIC"]["QMAX"] = 100
    micro_grid_1["BIC"]["SMAX"] = 100
    micro_grid_1["PD"]["AC"] = Profile[0] * micro_grid_1["PD"]["AC_MAX"]
    micro_grid_1["QD"]["AC"] = Profile[0] * micro_grid_1["PD"]["AC_MAX"] * 0.2
    micro_grid_1["PD"]["DC"] = Profile[0] * micro_grid_1["PD"]["DC_MAX"]
    # micro_grid_1["MG"]["PMIN"] = 0
    # micro_grid_1["MG"]["PMAX"] = 0

    micro_grid_2 = deepcopy(micro_grid)
    micro_grid_2["BUS"] = 4
    micro_grid_2["PD"]["AC_MAX"] = 50
    micro_grid_2["PD"]["DC_MAX"] = 50
    micro_grid_2["UG"]["PMIN"] = -500
    micro_grid_2["UG"]["PMAX"] = 500
    micro_grid_1["UG"]["QMIN"] = -500
    micro_grid_1["UG"]["QMAX"] = 500
    micro_grid_2["DG"]["PMAX"] = 50
    micro_grid_1["DG"]["QMAX"] = 50
    micro_grid_1["DG"]["QMIN"] = -50
    micro_grid_2["DG"]["COST_A"] = 0.01
    micro_grid_2["ESS"]["PDC_MAX"] = 50
    micro_grid_2["ESS"]["PCH_MAX"] = 50
    micro_grid_2["ESS"]["E0"] = 15
    micro_grid_2["ESS"]["EMIN"] = 10
    micro_grid_2["ESS"]["EMAX"] = 50
    micro_grid_2["BIC"]["PMAX"] = 100
    micro_grid_2["BIC"]["QMAX"] = 100
    micro_grid_2["BIC"]["SMAX"] = 100
    micro_grid_2["PD"]["AC"] = Profile[1] * micro_grid_2["PD"]["AC_MAX"]
    micro_grid_2["QD"]["AC"] = Profile[1] * micro_grid_2["PD"]["AC_MAX"] * 0.2
    micro_grid_2["PD"]["DC"] = Profile[1] * micro_grid_2["PD"]["DC_MAX"]
    # micro_grid_2["MG"]["PMIN"] = 0
    # micro_grid_2["MG"]["PMAX"] = 0

    micro_grid_3 = deepcopy(micro_grid)
    micro_grid_3["BUS"] = 10
    micro_grid_3["PD"]["AC_MAX"] = 50
    micro_grid_3["PD"]["DC_MAX"] = 50
    micro_grid_3["UG"]["PMIN"] = -500
    micro_grid_3["UG"]["PMAX"] = 500
    micro_grid_3["UG"]["QMIN"] = -500
    micro_grid_3["UG"]["QMAX"] = 500
    micro_grid_3["DG"]["PMAX"] = 50
    micro_grid_3["DG"]["QMAX"] = 50
    micro_grid_3["DG"]["QMIN"] = -50
    micro_grid_3["DG"]["COST_A"] = 0.01
    micro_grid_3["ESS"]["PDC_MAX"] = 50
    micro_grid_3["ESS"]["PCH_MAX"] = 50
    micro_grid_3["ESS"]["E0"] = 20
    micro_grid_3["ESS"]["EMIN"] = 10
    micro_grid_3["ESS"]["EMAX"] = 50
    micro_grid_3["BIC"]["PMAX"] = 50
    micro_grid_3["BIC"]["QMAX"] = 100
    micro_grid_3["BIC"]["SMAX"] = 100
    micro_grid_3["PD"]["AC"] = Profile[2] * micro_grid_3["PD"]["AC_MAX"]
    micro_grid_3["QD"]["AC"] = Profile[2] * micro_grid_3["PD"]["AC_MAX"] * 0.2
    micro_grid_3["PD"]["DC"] = Profile[2] * micro_grid_3["PD"]["DC_MAX"]
    case_micro_grids = [micro_grid_1, micro_grid_2, micro_grid_3]

    network_reconfiguration = NetworkReconfiguration()

    sol = network_reconfiguration.main(case=mpc, profile=load_profile.tolist(), microgrids=case_micro_grids)

    print(max(sol["residual"][0]))
