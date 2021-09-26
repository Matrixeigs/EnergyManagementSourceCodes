"""
Dynamic distribution network reconfiguration
1) Dynamic refers to multiple periods
2)
"""

from distribution_system_optimization.test_cases import case33
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


class NetworkReconfiguration():
    def __init__(self):
        self.name = "Network reconfiguration"

    def main(self, case, profile):
        """
        Main entrance for network reconfiguration problems
        :param case:
        :return:
        """
        case["branch"][:, BR_STATUS] = ones(case["branch"].shape[0])
        mpc = ext2int(case)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = range(nl)  ## double set of row indices
        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
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

        T = len(profile)
        nx = int(3 * nl + nb + 2 * ng)
        lx = concatenate([Alpha_l, Beta_f_l, Beta_t_l, tile(concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l]), T)])
        ux = concatenate([Alpha_u, Beta_f_u, Beta_t_u, tile(concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u]), T)])
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
                                                                                        zeros((nb, ng))]).toarray()
            beq_p[i * nb:(i + 1) * nb] = profile[i] * bus[:, PD] / baseMVA

        # 2) Reactive power balance
        Aeq_q = zeros((nb * T, NX))
        beq_q = zeros(nb * T)
        for i in range(T):
            Aeq_q[i * nb:(i + 1) * nb, 3 * nl + i * nx:3 * nl + (i + 1) * nx] = hstack([zeros((nb, nl)), Ct - Cf,
                                                                                        -diag(Ct * Branch_X) * Ct,
                                                                                        zeros((nb, nb)),
                                                                                        zeros((nb, ng)), Cg]).toarray()
            beq_q[i * nb:(i + 1) * nb] = profile[i] * bus[:, QD] / baseMVA

        Aeq = vstack([Aeq_f, Aeq_alpha, Aeq_span, Aeq_p, Aeq_q]).toarray()
        beq = concatenate([beq_f, beq_alpha, beq_span, beq_p, beq_q])

        # Inequality constraints
        A = zeros((nl * T, NX))
        b = zeros(nl * T)
        for i in range(T):
            A[i * nl:(i + 1) * nl, 3 * nl + i * nx + 2 * nl: 3 * nl + i * nx + 3 * nl] = eye(nl)
            A[i * nl:(i + 1) * nl, 0: nl] = -diag(Iij_u)

        A_temp = zeros((nl * T, NX))
        b_temp = zeros(nl * T)
        for i in range(T):
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx: 3 * nl + i * nx + nl] = eye(nl)
            A_temp[i * nl:(i + 1) * nl, 0: nl] = -diag(Pij_u)
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        #
        A_temp = zeros((nl * T, NX))
        b_temp = zeros(nl * T)
        for i in range(T):
            A_temp[i * nl:(i + 1) * nl, 3 * nl + i * nx + nl: 3 * nl + i * nx + 2 * nl] = eye(nl)
            A_temp[i * nl:(i + 1) * nl, 0:nl] = -diag(Qij_u)
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])

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

        sol = miqcp(c, q, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lx, xmax=ux, vtypes=vtypes, Qc=Qc)
        xx = sol[0]
        Alpha = xx[0:nl]
        Beta_f = xx[nl:2 * nl]
        Beta_t = xx[2 * nl:3 * nl]
        Pij = zeros((nl, T))
        Qij = zeros((nl, T))
        Iij = zeros((nl, T))
        Vi = zeros((nb, T))
        Pg = zeros((ng, T))
        Qg = zeros((ng, T))
        for i in range(T):
            Pij[:, i] = xx[3 * nl + i * nx:3 * nl + i * nx + nl]
            Qij[:, i] = xx[3 * nl + i * nx + nl:3 * nl + i * nx + 2 * nl]
            Iij[:, i] = xx[3 * nl + i * nx + 2 * nl:3 * nl + i * nx + 3 * nl]
            Vi[:, i] = xx[3 * nl + i * nx + 3 * nl:3 * nl + i * nx + 3 * nl + nb]
            Pg[:, i] = xx[3 * nl + i * nx + 3 * nl + nb:3 * nl + i * nx + 3 * nl + nb + ng]
            Qg[:, i] = xx[3 * nl + i * nx + 3 * nl + nb + ng:3 * nl + i * nx + 3 * nl + nb + 2 * ng]

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
               "Alpha": Alpha,
               "Beta_f": Beta_f,
               "Beta_t": Beta_t,
               "residual": primal_residual,
               "obj": sol[1] + c0}

        return sol


if __name__ == "__main__":
    from pypower import runopf

    mpc = case33.case33()  # Default test case
    # profile = array([2, 2]) / 2
    profile = array(
        [1.75, 1.65, 1.58, 1.54, 1.55, 1.60, 1.73, 1.77, 1.86, 2.07, 2.29, 2.36, 2.42, 2.44, 2.49, 2.56, 2.56, 2.47,
         2.46, 2.37, 2.37, 2.33, 1.96, 1.96]) / 3

    network_reconfiguration = NetworkReconfiguration()

    sol = network_reconfiguration.main(case=mpc, profile=profile.tolist())

    print(max(sol["residual"][0]))
