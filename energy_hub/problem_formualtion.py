"""
Problem formulation class of energy hub optimization
Two types of problems will be formulated.
1) Day-ahead optimization

2) Real-time optimization

@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
@date: 10 Feb 2018
"""
from numpy import zeros, ones, inf, vstack, hstack, eye


class ProblemFormulation():
    """
    Two stage optimization problems formulation for energy hub systems
    """

    def __init__(self):
        self.name = 'EnergyHubProblem'

    def first_stage_problem(self, *args):
        # import data format
        from energy_hub.data_format import GAS, UG, PAC2DC, PDC2AC, PHVAC, EESS, PESSDC, PESSCH, NX
        # Parameter announcement
        PHVAC_max = args[0]
        eff_HVAC = args[1]
        Pess_ch_max = args[2]
        Pess_dc_max = args[3]
        eff_dc = args[4]
        eff_ch = args[5]
        E0 = args[6]
        Emax = args[7]
        Emin = args[8]
        BIC_cap = args[9]
        Gmax = args[10]
        eff_BIC = args[11]
        eff_CHP_e = args[12]
        eff_CHP_h = args[13]

        # Load profile
        AC_PD = args[14]
        DC_PD = args[15]
        HD = args[16]
        CD = args[17]
        PV_PG = args[18]

        # Price profile
        Gas_price = args[19]
        Electric_price = args[20]
        Eess_cost = args[21]

        Delta_t = args[22]
        T = int(args[23])
        # Formulate the boundary information
        lb = zeros((NX * T, 1))
        ub = zeros((NX * T, 1))
        for i in range(T):
            lb[i * NX + UG] = 0
            lb[i * NX + GAS] = 0
            lb[i * NX + PAC2DC] = 0
            lb[i * NX + PDC2AC] = 0
            lb[i * NX + EESS] = Emin
            lb[i * NX + PESSDC] = 0
            lb[i * NX + PESSCH] = 0
            lb[i * NX + PHVAC] = 0

            ub[i * NX + UG] = Gmax
            ub[i * NX + GAS] = Gmax
            ub[i * NX + PAC2DC] = BIC_cap
            ub[i * NX + PDC2AC] = BIC_cap
            ub[i * NX + EESS] = Emax
            ub[i * NX + PESSDC] = Pess_dc_max
            ub[i * NX + PESSCH] = Pess_ch_max
            ub[i * NX + PHVAC] = PHVAC_max
        # The constain set, all equal constraints
        Aeq = zeros((T, NX * T))
        beq = zeros((T, 1))
        for i in range(T):
            Aeq[i, i * NX + GAS] = eff_CHP_h
            beq[i] = HD[i]

        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            Aeq_temp[i, i * NX + PHVAC] = eff_HVAC
            beq_temp[i] = CD[i]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])

        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            Aeq_temp[i, i * NX + UG] = 1
            Aeq_temp[i, i * NX + GAS] = eff_CHP_e
            Aeq_temp[i, i * NX + PDC2AC] = eff_BIC
            Aeq_temp[i, i * NX + PAC2DC] = -1
            beq_temp[i] = AC_PD[i]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])

        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            Aeq_temp[i, i * NX + PESSDC] = 1
            Aeq_temp[i, i * NX + PESSCH] = -1
            Aeq_temp[i, i * NX + PAC2DC] = eff_BIC
            Aeq_temp[i, i * NX + PDC2AC] = -1
            beq_temp[i] = DC_PD[i] - PV_PG[i]

        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])

        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            if i == 1:
                Aeq_temp[i, i * NX + EESS] = 1
                Aeq_temp[i, i * NX + PESSCH] = -eff_ch * Delta_t
                Aeq_temp[i, i * NX + PESSDC] = eff_dc * Delta_t
                beq_temp[i] = E0
            else:
                Aeq_temp[i, i * NX + EESS] = 1
                Aeq_temp[i, (i - 1) * NX + EESS] = -1
                Aeq_temp[i, i * NX + PESSCH] = -eff_ch * Delta_t
                Aeq_temp[i, i * NX + PESSDC] = eff_dc * Delta_t
                beq_temp[i] = 0
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])

        c = zeros((NX * T, 1))
        for i in range(T):
            c[i * NX + UG] = Electric_price[i] * Delta_t
            c[i * NX + GAS] = Gas_price * Delta_t
            c[i * NX + PESSDC] = Eess_cost * Delta_t
            c[i * NX + PESSCH] = Eess_cost * Delta_t

        mathematical_model = {'c': c,
                              'Aeq': Aeq,
                              'beq': beq,
                              'A': None,
                              'b': None,
                              'lb': lb,
                              'ub': ub}
        return mathematical_model

    def second_stage_problem(self, *args):
        # import data format from the second stage optimization
        from energy_hub.data_format_second_stage import GAS, UG, PAC2DC, PDC2AC, PHVAC, EESS, PESSDC, PESSCH, \
            PH_ABS, PH_RELAX, PC_ABS, PC_RELAX, NX
        # Parameter announcement
        PHVAC_max = args[0]
        eff_HVAC = args[1]
        Pess_ch_max = args[2]
        Pess_dc_max = args[3]
        eff_dc = args[4]
        eff_ch = args[5]
        E0 = args[6]
        Emax = args[7]
        Emin = args[8]
        BIC_cap = args[9]
        Gmax = args[10]
        eff_BIC = args[11]
        eff_CHP_e = args[12]
        eff_CHP_h = args[13]

        # Load profile
        AC_PD = args[14]
        DC_PD = args[15]
        HD = args[16]
        CD = args[17]
        PV_PG = args[18]

        # Price profile
        Gas_price = args[19]
        Electric_price = args[20]
        Eess_cost = args[21]

        Delta_t = args[22]
        T = int(args[23])
        weight_factor = int(args[24])
        # Formulate the boundary information
        lb = zeros((NX * T, 1))
        ub = zeros((NX * T, 1))
        for i in range(T):
            lb[i * NX + UG] = 0
            lb[i * NX + GAS] = 0
            lb[i * NX + PAC2DC] = 0
            lb[i * NX + PDC2AC] = 0
            lb[i * NX + EESS] = Emin
            lb[i * NX + PESSDC] = 0
            lb[i * NX + PESSCH] = 0
            lb[i * NX + PHVAC] = 0

            ub[i * NX + UG] = Gmax
            ub[i * NX + GAS] = Gmax
            ub[i * NX + PAC2DC] = BIC_cap
            ub[i * NX + PDC2AC] = BIC_cap
            ub[i * NX + EESS] = Emax
            ub[i * NX + PESSDC] = Pess_dc_max
            ub[i * NX + PESSCH] = Pess_ch_max
            ub[i * NX + PHVAC] = PHVAC_max
        # The constain set, all equal constraints
        # 1) AC side power balance equation
        Aeq = zeros((T, NX * T))
        beq = zeros((T, 1))
        for i in range(T):
            Aeq[i, i * NX + UG] = 1
            Aeq[i, i * NX + GAS] = eff_CHP_e
            Aeq[i, i * NX + PDC2AC] = eff_BIC
            Aeq[i, i * NX + PAC2DC] = -1
            beq[i] = AC_PD[i]
        # 2) DC side power balance equation
        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            Aeq_temp[i, i * NX + PESSDC] = 1
            Aeq_temp[i, i * NX + PESSCH] = -1
            Aeq_temp[i, i * NX + PAC2DC] = eff_BIC
            Aeq_temp[i, i * NX + PDC2AC] = -1
            beq_temp[i] = DC_PD[i] - PV_PG[i]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])
        # 3) Heat and cooling energy balance equation
        # 3.1) Heat energy balance
        N_compress = 4
        T_compress = int(T / N_compress)
        Aeq_temp = zeros((T_compress, NX * T))
        beq_temp = zeros((T_compress, 1))
        for i in range(T_compress):
            Aeq_temp[i, i * 4 * NX + GAS] = eff_CHP_h * Delta_t
            Aeq_temp[i, (i * 4 + 1) * NX + GAS] = eff_CHP_h * Delta_t
            Aeq_temp[i, (i * 4 + 2) * NX + GAS] = eff_CHP_h * Delta_t
            Aeq_temp[i, (i * 4 + 3) * NX + GAS] = eff_CHP_h * Delta_t
            beq_temp[i] = HD[i]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])
        # 3.2) Cooling energy balance
        Aeq_temp = zeros((T_compress, NX * T))
        beq_temp = zeros((T_compress, 1))
        for i in range(T_compress):
            Aeq_temp[i, i * 4 * NX + PHVAC] = eff_HVAC * Delta_t
            Aeq_temp[i, (i * 4 + 1) * NX + PHVAC] = eff_HVAC * Delta_t
            Aeq_temp[i, (i * 4 + 2) * NX + PHVAC] = eff_HVAC * Delta_t
            Aeq_temp[i, (i * 4 + 3) * NX + PHVAC] = eff_HVAC * Delta_t
            beq_temp[i] = CD[i]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])
        # 4) ESS SOC dynamic equation
        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            if i == 1:
                Aeq_temp[i, i * NX + EESS] = 1
                Aeq_temp[i, i * NX + PESSCH] = -eff_ch * Delta_t
                Aeq_temp[i, i * NX + PESSDC] = eff_dc * Delta_t
                beq_temp[i] = E0
            else:
                Aeq_temp[i, i * NX + EESS] = 1
                Aeq_temp[i, (i - 1) * NX + EESS] = -1
                Aeq_temp[i, i * NX + PESSCH] = -eff_ch * Delta_t
                Aeq_temp[i, i * NX + PESSDC] = eff_dc * Delta_t
                beq_temp[i] = 0
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])

        c = zeros((NX * T,1))
        for i in range(T):
            c[i * NX + UG] = Electric_price[int(i * Delta_t)] * Delta_t
            c[i * NX + GAS] = Gas_price * Delta_t
            c[i * NX + PESSDC] = Eess_cost * Delta_t
            c[i * NX + PESSCH] = Eess_cost * Delta_t
            # The relaxation of second stage optimization
            c[i * NX + PH_ABS] = weight_factor * Delta_t
            c[i * NX + PC_ABS] = weight_factor * Delta_t

        mathematical_model = {'c': c,
                              'Aeq': Aeq,
                              'beq': beq,
                              'A': None,
                              'b': None,
                              'lb': lb,
                              'ub': ub}
        # Formulating the standard format problem, removing the upper and lower boundary information to equality constraints.
        neq = Aeq.shape[0]
        nx = Aeq.shape[1]
        # Aeq_extended = zeros((neq + 2 * nx, 3 * nx))
        lb_extended = zeros((3 * nx, 1))
        ub_extended = inf * ones((3 * nx, 1))
        beq_extended = vstack([beq, lb, ub])
        c_extended = vstack([c, zeros((2 * nx, 1))])
        Aeq_extended = hstack([Aeq, zeros((neq, 2 * nx))])
        Aeq_extended = vstack([Aeq_extended, hstack([eye(nx), -eye(nx), zeros((nx, nx))]),
                               hstack([eye(nx), zeros((nx, nx)), eye(nx)])])

        mathematical_model_extended = {'c': c_extended,
                                       'Aeq': Aeq_extended,
                                       'beq': beq_extended,
                                       'A': None,
                                       'b': None,
                                       'lb': lb_extended,
                                       'ub': ub_extended}

        return mathematical_model, mathematical_model_extended

    def coupling_constraints(self, *args):
        """
        The coupling constraints for the first stage optimization problem and second stage problem
        :param args: The mathematical models of two stage optimization problem
        :return:
        """
        model_first_stage = args[0]
        model_second_stage = args[1]

        return model_first_stage
