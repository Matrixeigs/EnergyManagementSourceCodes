"""
Two-stage stochastic optimization for hybrid AC/DC microgrids under the uncertainty of loads
@author: Zhao Tianyang
@Date: 4 Aug 2019
@e-mail: matrixeigs@gmail.com

The hybrid AC/DC has multiple DGs, BICs, ESSs and renewable sources

"""
from scipy import zeros, shape, ones, diag, concatenate, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, lil_matrix

from micro_grids.idx_format_hybrid_AC_DC import PBIC_A2D, PBIC_D2A, PESS_CH0, PESS_DC0, PG0, PMESS, PPV0, PUG, NX_MG, \
    QBIC, QG0, QUG, NG, NESS, NRES, EESS0


class TwoStageStochasticHybirdACDCMG():

    def __init__(self):
        self.name = "Two stage stochastic optimization for hybrid AC/DC microgrids"
        self.ng = NG
        self.ness = NESS
        self.nres = NRES
        # Define the first-stage decision variable
        self.PG0 = 0
        self.RG0 = NG
        self.PUG = NG * 2
        self.nx_first_stage = NG * 2 + 1



    def main(self, microgrids):
        """
        Main function for hybrid AC DC microgrids
        :param microgrids:
        :return:
        """
        self.T = len(microgrids["PD"]["AC"])  # The number of time slots
        # First stage problem formulation

        # Second stage problem formulation

        # Merge the first stage and second stage problems

        # Solve the problem

        # Store the data into excel file

    def first_stage_problem_formulation_microgrid(self, mg):
        """
        First stage optimization problem formulation
        :param mg:
        :return:
        """



    def second_stage_problem_formulation_microgrid(self, mg):
        """
        Dynamic optimal power flow problem formulation of single micro_grid
        Might be extended with
        :param micro_grid:
        :return:
        """
        try:
            T = self.T
        except:
            T = 24

        ## 1) boundary information and objective function
        nv = NX_MG * T
        ng = self.ng
        npv = self.nres
        ness = self.ness

        lb = zeros(nv)
        ub = zeros(nv)
        c = zeros(nv)
        q = zeros(nv)
        vtypes = ["c"] * nv
        for t in range(T):
            ## 1.1) lower boundary
            for j in range(ng):
                lb[t * NX_MG + PG0] = 0
                lb[t * NX_MG + QG0] = mg["DG"]["QMIN"]
            lb[t * NX_MG + PUG] = 0
            lb[t * NX_MG + QUG] = mg["UG"]["QMIN"]
            lb[t * NX_MG + PBIC_D2A] = 0
            lb[t * NX_MG + PBIC_A2D] = 0
            lb[t * NX_MG + QBIC] = -mg["BIC"]["SMAX"]
            for j in range(ness):
                lb[t * NX_MG + PESS_CH0+j] = 0
                lb[t * NX_MG + PESS_DC0+j] = 0
                lb[t * NX_MG + EESS0+j] = mg["ESS"]["EMIN"]
            for j in range(npv):
                lb[t * NX_MG + PPV0+j] = 0

            # lb[t * NX_MG + PMESS] = pmess_l
            ## 1.2) upper boundary
            for j in range(ng):
                ub[t * NX_MG + PG0+j] = mg["DG"]["PMAX"]
                ub[t * NX_MG + QG0+j] = mg["DG"]["QMAX"]
            ub[t * NX_MG + PUG] = mg["UG"]["PMAX"]
            ub[t * NX_MG + QUG] = mg["UG"]["QMAX"]
            ub[t * NX_MG + PBIC_D2A] = mg["BIC"]["PMAX"]
            ub[t * NX_MG + PBIC_A2D] = mg["BIC"]["PMAX"]
            ub[t * NX_MG + QBIC] = mg["BIC"]["SMAX"]
            for j in range(ness):
                ub[t * NX_MG + PESS_CH0+j] = mg["ESS"]["PCH_MAX"]
                ub[t * NX_MG + PESS_DC0+j] = mg["ESS"]["PDC_MAX"]
                ub[t * NX_MG + EESS0+j] = mg["ESS"]["EMAX"]
            for j in range(npv):
                ub[t * NX_MG + PPV0] = mg["PV"]["PROFILE"][t]
            # ub[t * NX_MG + PMESS] = pmess_u
            ## 1.3) Objective functions
            for j in range(ng):
                c[t * NX_MG + PG0 + j] = mg["DG"]["COST_A"]
            for j in range(ness):
                c[t * NX_MG + PESS_CH0+j] = mg["ESS"]["COST_OP"]
                c[t * NX_MG + PESS_DC0+j] = mg["ESS"]["COST_OP"]

            for j in range(npv):
                c[t * NX_MG + PPV0] = mg["PV"]["COST"]
            ## 1.4) Upper and lower boundary information
            if t == T - 1:
                for j in range(ness):
                    lb[t * NX_MG + EESS0 + j] = mg["ESS"]["E0"]
                    ub[t * NX_MG + EESS0 + j] = mg["ESS"]["E0"]

        # 2) Formulate the equal constraints
        # 2.1) Power balance equation
        # a) AC bus equation
        Aeq = lil_matrix((T, nv))
        beq = zeros(T)
        for t in range(T):
            for j in range(ng):
                Aeq[t, t * NX_MG + PG0+j] = 1
            Aeq[t, t * NX_MG + PUG] = 1
            Aeq[t, t * NX_MG + PBIC_A2D] = -1
            Aeq[t, t * NX_MG + PBIC_D2A] = mg["BIC"]["EFF_DC2AC"]
            beq[t] = mg["PD"]["AC"][t]
        # b) DC bus equation
        Aeq_temp = lil_matrix((T, nv))
        beq_temp = zeros(T)
        for t in range(T):
            Aeq_temp[t, t * NX_MG + PBIC_A2D] = mg["BIC"]["EFF_AC2DC"]
            Aeq_temp[t, t * NX_MG + PBIC_D2A] = -1
            Aeq_temp[t, t * NX_MG + PESS_CH] = -1
            Aeq_temp[t, t * NX_MG + PESS_DC] = 1
            Aeq_temp[t, t * NX_MG + PPV] = 1
            Aeq_temp[t, t * NX_MG + PMESS] = 1  # The power injection from mobile energy storage systems
            beq_temp[t] = mg["PD"]["DC"][t]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # c) AC reactive power balance equation
        Aeq_temp = lil_matrix((T, nv))
        beq_temp = zeros(T)
        for t in range(T):
            Aeq_temp[t, t * NX_MG + QUG] = 1
            Aeq_temp[t, t * NX_MG + QBIC] = 1
            Aeq_temp[t, t * NX_MG + QG] = 1
            beq_temp[t] = mg["QD"]["AC"][t]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        # 2.2) Energy storage balance equation
        Aeq_temp = lil_matrix((T*ness, nv))
        beq_temp = zeros(T)
        for t in range(T):
            for j in range(ness):
                Aeq_temp[j*T+t, t * NX_MG + EESS0+j] = 1
                Aeq_temp[j*T+t, t * NX_MG + PESS_CH0+j] = -mg["ESS"]["EFF_CH"]
                Aeq_temp[j*T+t, t * NX_MG + PESS_DC0+j] = 1 / mg["ESS"]["EFF_DC"]
                if t == 0:
                    beq_temp[j*T+t] = mg["ESS"]["E0"]
                else:
                    Aeq_temp[j*T+t, (t - 1) * NX_MG + EESS0+j] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # 3) Formualte inequality constraints
        # There is no inequality constraint.

        # sol = milp(c, Aeq=Aeq, beq=beq, A=None, b=None, xmin=lb, xmax=ub)

        model_micro_grid = {"c": c,
                            "q": q,
                            "lb": lb,
                            "ub": ub,
                            "vtypes": vtypes,
                            "A": None,
                            "b": None,
                            "Aeq": Aeq,
                            "beq": beq
                            }

        return model_micro_grid
