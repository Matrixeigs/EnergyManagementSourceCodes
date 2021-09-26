"""
Day-ahead deterministic bidding strategy
@author:Tianyang Zhao
@e-mail:zhaoty@ntu.edu.sg
The parameters are obtained from the following papers.
[1] Energy flow modeling and optimal operation analysis of the micro energy grid based on energy hub
The ice maker and ice storage are adopted

[2]

"""
from numpy import array, arange, zeros, ones, concatenate
import numpy as np

from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp

class EnergyHubManagement():
    def __init__(self):
        self.name = "day-ahead modelling"

    def problem_formulation(self, ELEC=None, BIC=None, ESS=None, CCHP=None, HVAC=None, THERMAL=None, CHIL=None,
                            BOIL=None, T=None):
        """
        Problem formulation for energy hub management
        :param ELEC: Electrical system with the load and utility grid information
        :param BIC: Bi-directional converter information
        :param ESS: Energy storage system information (Battery ESS and Thermal ESS)
        :param CCHP: Combined heat and power units information
        :param HVAC: Heat, ventilation and air-conditioning information
        :param THERMAL: Thermal load information
        :return:
        """
        from energy_hub.bidding_strategy.data_format import PUG, PCHP, PAC2DC, PDC2AC, PIAC, EESS, PESS_CH, PESS_DC, \
            PPV, PCS, QCHP, QGAS, EHSS, QHS_DC, QHS_CH, QAC, QTD, QCE, QIAC, ECSS, QCS_DC, QCS_CH, QCD, VCHP, VGAS, NX
        self.T = T
        # 1） Formulate the day-ahead operation plan
        # 1.1) The decision variables
        nx = NX * T
        lb = zeros((nx, 1))  # The lower boundary
        ub = zeros((nx, 1))  # The upper boundary
        # Update the boundary information
        for i in range(T):
            lb[i * NX + PUG] = ELEC["UG_MIN"]
            lb[i * NX + PCHP] = 0
            lb[i * NX + PAC2DC] = 0
            lb[i * NX + PDC2AC] = 0
            lb[i * NX + PIAC] = 0
            lb[i * NX + EESS] = ESS["BESS"]["E_MIN"]
            lb[i * NX + PESS_DC] = 0
            lb[i * NX + PESS_CH] = 0
            lb[i * NX + PPV] = 0
            lb[i * NX + PCS] = 0
            lb[i * NX + QCHP] = 0
            lb[i * NX + QGAS] = 0
            lb[i * NX + EHSS] = ESS["TESS"]["E_MIN"]
            lb[i * NX + QHS_DC] = 0
            lb[i * NX + QHS_CH] = 0
            lb[i * NX + QAC] = 0
            lb[i * NX + QTD] = 0
            lb[i * NX + QCE] = 0
            lb[i * NX + QIAC] = 0
            lb[i * NX + ECSS] = ESS["CESS"]["E_MIN"]
            lb[i * NX + QCS_DC] = 0
            lb[i * NX + QCS_CH] = 0
            lb[i * NX + QCD] = 0
            lb[i * NX + VCHP] = 0
            lb[i * NX + VGAS] = 0

            ub[i * NX + PUG] = ELEC["UG_MAX"]
            ub[i * NX + PCHP] = CCHP["MAX"] * CCHP["EFF_E"]
            ub[i * NX + PAC2DC] = BIC["CAP"]
            ub[i * NX + PDC2AC] = BIC["CAP"]
            ub[i * NX + PIAC] = HVAC["CAP"]
            ub[i * NX + EESS] = ESS["BESS"]["E_MAX"]
            ub[i * NX + PESS_DC] = ESS["BESS"]["PC_MAX"]
            ub[i * NX + PESS_CH] = ESS["BESS"]["PD_MAX"]
            ub[i * NX + PPV] = ELEC["PV_PG"][i]
            ub[i * NX + PCS] = ESS["CESS"]["PMAX"]

            ub[i * NX + QCHP] = CCHP["MAX"] * CCHP["EFF_H"]
            ub[i * NX + QGAS] = CCHP["MAX"] * CCHP["EFF_H"]
            ub[i * NX + EHSS] = ESS["TESS"]["E_MAX"]
            ub[i * NX + QHS_DC] = ESS["TESS"]["TD_MAX"]
            ub[i * NX + QHS_CH] = ESS["TESS"]["TC_MAX"]
            ub[i * NX + QAC] = CHIL["CAP"]
            ub[i * NX + QTD] = THERMAL["HD"][i]

            ub[i * NX + QCE] = CHIL["CAP"]
            ub[i * NX + QIAC] = HVAC["CAP"] * HVAC["EFF"]
            ub[i * NX + ECSS] = ESS["CESS"]["E_MAX"]
            ub[i * NX + QCS_DC] = ESS["CESS"]["TD_MAX"]
            ub[i * NX + QCS_CH] = ESS["CESS"]["TC_MAX"]
            ub[i * NX + QCD] = THERMAL["CD"][i]
            ub[i * NX + VCHP] = CCHP["MAX"]
            ub[i * NX + VGAS] = BOIL["CAP"]
            # Add the energy status constraints
            if i == T - 1:
                lb[i * NX + EESS] = ESS["BESS"]["E0"]
                ub[i * NX + EESS] = ESS["BESS"]["E0"]
                lb[i * NX + EHSS] = ESS["TESS"]["E0"]
                ub[i * NX + EHSS] = ESS["TESS"]["E0"]
                lb[i * NX + ECSS] = ESS["CESS"]["E0"]
                ub[i * NX + ECSS] = ESS["CESS"]["E0"]
        # 1.2 Formulate the equality constraint set
        # 1.2.1) The constraints for the battery energy storage systems
        Aeq_bess = zeros((T, nx))
        beq_bess = zeros((T, 1))
        for i in range(T):
            Aeq_bess[i, i * NX + EESS] = 1
            Aeq_bess[i, i * NX + PESS_CH] = -ESS["BESS"]["EFF_CH"]
            Aeq_bess[i, i * NX + PESS_DC] = 1 / ESS["BESS"]["EFF_DC"]
            if i != 0:
                Aeq_bess[i, (i - 1) * NX + EESS] = -1
                beq_bess[i, 0] = 0
            else:
                beq_bess[i, 0] = ESS["BESS"]["E0"]

        # 1.2.2) The constraints for the heat storage
        Aeq_tess = zeros((T, nx))
        beq_tess = zeros((T, 1))
        for i in range(T):
            Aeq_tess[i, i * NX + EHSS] = 1
            Aeq_tess[i, i * NX + QHS_CH] = -ESS["TESS"]["EFF_CH"]
            Aeq_tess[i, i * NX + QHS_DC] = 1 / ESS["TESS"]["EFF_DC"]
            if i != 0:
                Aeq_tess[i, (i - 1) * NX + EHSS] = -ESS["TESS"]["EFF_SD"]
                beq_tess[i, 0] = 0
            else:
                beq_tess[i, 0] = ESS["TESS"]["EFF_SD"] * ESS["TESS"]["E0"]

        # 1.2.3) The constraints for the cooling storage
        Aeq_cess = zeros((T, nx))
        beq_cess = zeros((T, 1))
        for i in range(T):
            Aeq_cess[i, i * NX + ECSS] = 1
            Aeq_cess[i, i * NX + QCS_CH] = -ESS["CESS"]["EFF_CH"]
            Aeq_cess[i, i * NX + QCS_DC] = 1 / ESS["CESS"]["EFF_DC"]
            if i != 0:  # Not the first period
                Aeq_cess[i, (i - 1) * NX + ECSS] = -ESS["CESS"]["EFF_SD"]
                beq_cess[i, 0] = 0
            else:
                beq_cess[i, 0] = ESS["CESS"]["EFF_SD"] * ESS["CESS"]["E0"]

        # 1.2.4) Energy conversion relationship
        # 1.2.4.1）For the combined heat and power unit, electricity
        Aeq_chp_e = zeros((T, nx))
        beq_chp_e = zeros((T, 1))
        for i in range(T):
            Aeq_chp_e[i, i * NX + VCHP] = CCHP["EFF_E"]
            Aeq_chp_e[i, i * NX + PCHP] = -1
        # 1.2.4.2）For the combined heat and power unit, heat
        Aeq_chp_h = zeros((T, nx))
        beq_chp_h = zeros((T, 1))
        for i in range(T):
            Aeq_chp_h[i, i * NX + VCHP] = CCHP["EFF_H"]
            Aeq_chp_h[i, i * NX + QCHP] = -1
        # 1.2.4.3) For the Gas boiler
        Aeq_boil = zeros((T, nx))
        beq_boil = zeros((T, 1))
        for i in range(T):
            Aeq_boil[i, i * NX + VGAS] = BOIL["EFF"]
            Aeq_boil[i, i * NX + QGAS] = -1
        # 1.2.4.4) For the absorption chiller
        Aeq_chil = zeros((T, nx))
        beq_chil = zeros((T, 1))
        for i in range(T):
            Aeq_chil[i, i * NX + QAC] = CHIL["EFF"]
            Aeq_chil[i, i * NX + QCE] = -1
        # 1.2.4.5) For the inverter air-conditioning
        Aeq_iac = zeros((T, nx))
        beq_iac = zeros((T, 1))
        for i in range(T):
            Aeq_iac[i, i * NX + PIAC] = HVAC["EFF"]
            Aeq_iac[i, i * NX + QIAC] = -1
        # 1.2.4.6) For the ice-maker
        Aeq_ice = zeros((T, nx))
        beq_ice = zeros((T, 1))
        for i in range(T):
            Aeq_ice[i, i * NX + PCS] = ESS["CESS"]["ICE"]
            Aeq_ice[i, i * NX + QCS_CH] = -1

        # 1.2.5) The power balance for the AC bus in the hybrid AC/DC micro-grid
        Aeq_ac = zeros((T, nx))
        beq_ac = zeros((T, 1))
        for i in range(T):
            Aeq_ac[i, i * NX + PUG] = 1
            Aeq_ac[i, i * NX + PCHP] = 1
            Aeq_ac[i, i * NX + PAC2DC] = -1
            Aeq_ac[i, i * NX + PDC2AC] = BIC["EFF"]

            beq_ac[i, 0] = ELEC["AC_PD"][i]
        # 1.2.6) The power balance for the DC bus in the hybrid AC/DC micro-grid
        Aeq_dc = zeros((T, nx))
        beq_dc = zeros((T, 1))
        for i in range(T):
            Aeq_dc[i, i * NX + PIAC] = -1  # Provide cooling service
            Aeq_dc[i, i * NX + PAC2DC] = BIC["EFF"]  #
            Aeq_dc[i, i * NX + PDC2AC] = -1
            Aeq_dc[i, i * NX + PESS_CH] = -1
            Aeq_dc[i, i * NX + PESS_DC] = 1
            Aeq_dc[i, i * NX + PPV] = 1
            Aeq_ac[i, i * NX + PCS] = -1
            beq_dc[i, 0] = ELEC["DC_PD"][i]

        # 1.2.7) heating hub balance
        Aeq_hh = zeros((T, nx))
        beq_hh = zeros((T, 1))
        for i in range(T):
            Aeq_hh[i, i * NX + QCHP] = 1
            Aeq_hh[i, i * NX + QGAS] = 1
            Aeq_hh[i, i * NX + QHS_DC] = 1
            Aeq_hh[i, i * NX + QHS_CH] = -1
            Aeq_hh[i, i * NX + QAC] = -1
            Aeq_hh[i, i * NX + QTD] = -1
            beq_hh[i, 0] = THERMAL["HD"][i]
        # 1.2.8) Cooling hub balance
        Aeq_ch = zeros((T, nx))
        beq_ch = zeros((T, 1))
        for i in range(T):
            Aeq_ch[i, i * NX + QIAC] = 1
            Aeq_ch[i, i * NX + QCE] = 1
            Aeq_ch[i, i * NX + QCS_DC] = 1
            Aeq_ch[i, i * NX + QCD] = -1
            beq_ch[i, 0] = THERMAL["CD"][i]
        # 1.3) For the inequality constraints
        # In this version, it seems that, there is none inequality constraints

        # 1.4) For the objective function
        c = zeros((nx, 1))
        for i in range(T):
            c[i * NX + PUG] = ELEC["UG_PRICE"][i]
            c[i * NX + PCHP] = 0
            c[i * NX + PAC2DC] = 0
            c[i * NX + PDC2AC] = 0
            c[i * NX + PIAC] = 0
            c[i * NX + EESS] = 0
            c[i * NX + PESS_DC] = ESS["BESS"]["COST"]
            c[i * NX + PESS_CH] = ESS["BESS"]["COST"]
            c[i * NX + PPV] = 0

            c[i * NX + QCHP] = 0
            c[i * NX + QGAS] = 0
            c[i * NX + EHSS] = 0
            c[i * NX + QHS_DC] = ESS["TESS"]["COST"]
            c[i * NX + QHS_CH] = ESS["TESS"]["COST"]
            c[i * NX + QAC] = 0
            c[i * NX + QTD] = 0

            c[i * NX + QCE] = 0
            c[i * NX + QIAC] = 0
            c[i * NX + ECSS] = 0
            c[i * NX + QCS_DC] = ESS["CESS"]["COST"]
            c[i * NX + QCS_CH] = ESS["CESS"]["COST"]
            c[i * NX + QCD] = 0

            c[i * NX + VCHP] = CCHP["COST"]
            c[i * NX + VGAS] = CCHP["COST"]

        # Combine the constraint set
        Aeq = concatenate(
            [Aeq_bess, Aeq_tess, Aeq_cess, Aeq_chp_e, Aeq_chp_h, Aeq_boil, Aeq_chil, Aeq_iac, Aeq_ice, Aeq_ac, Aeq_dc,
             Aeq_hh, Aeq_ch])
        beq = concatenate(
            [beq_bess, beq_tess, beq_cess, beq_chp_e, beq_chp_h, beq_boil, beq_chil, beq_iac, beq_ice, beq_ac, beq_dc,
             beq_hh, beq_ch])

        model = {"Aeq": Aeq,
                 "beq": beq,
                 "A": None,
                 "b": None,
                 "c": c,
                 "lb": lb,
                 "ub": ub,
                 "ac_eq": [9 * T, 10 * T],
                 "nx": NX,
                 "pug": PUG}

        return model

    def problem_solving(self, model):
        """
        problem solving of the day-ahead energy hub
        :param model:
        :return:
        """
        from energy_hub.bidding_strategy.data_format import PUG, PCHP, PAC2DC, PDC2AC, PIAC, EESS, PESS_CH, PESS_DC, \
            PPV, PCS, QCHP, QGAS, EHSS, QHS_DC, QHS_CH, QAC, QTD, QCE, QIAC, ECSS, QCS_DC, QCS_CH, QCD, VCHP, VGAS, NX

        # Try to solve the linear programing problem
        (x, objvalue, status) = lp(model["c"], Aeq=model["Aeq"], beq=model["beq"], xmin=model["lb"], xmax=model["ub"])
        T = self.T
        # decouple the solution
        pug = zeros((T, 1))
        pchp = zeros((T, 1))
        pac2dc = zeros((T, 1))
        pdc2ac = zeros((T, 1))
        piac = zeros((T, 1))
        eess = zeros((T, 1))
        pess_ch = zeros((T, 1))
        pess_dc = zeros((T, 1))
        ppv = zeros((T, 1))
        qchp = zeros((T, 1))
        qgas = zeros((T, 1))
        etss = zeros((T, 1))
        qes_dc = zeros((T, 1))
        qes_ch = zeros((T, 1))
        qac = zeros((T, 1))
        qtd = zeros((T, 1))
        qce = zeros((T, 1))
        qiac = zeros((T, 1))
        ecss = zeros((T, 1))
        qcs_dc = zeros((T, 1))
        qcs_ch = zeros((T, 1))
        qcd = zeros((T, 1))
        vchp = zeros((T, 1))
        vgas = zeros((T, 1))
        for i in range(T):
            pug[i, 0] = x[i * NX + PUG]
            pchp[i, 0] = x[i * NX + PCHP]
            pac2dc[i, 0] = x[i * NX + PAC2DC]
            pdc2ac[i, 0] = x[i * NX + PDC2AC]
            piac[i, 0] = x[i * NX + PIAC]
            eess[i, 0] = x[i * NX + EESS]
            pess_ch[i, 0] = x[i * NX + PESS_CH]
            pess_dc[i, 0] = x[i * NX + PESS_DC]
            ppv[i, 0] = x[i * NX + PPV]
            qchp[i, 0] = x[i * NX + QCHP]
            qgas[i, 0] = x[i * NX + QGAS]
            etss[i, 0] = x[i * NX + EHSS]
            qes_dc[i, 0] = x[i * NX + QHS_DC]
            qes_ch[i, 0] = x[i * NX + QHS_CH]
            qac[i, 0] = x[i * NX + QAC]
            qtd[i, 0] = x[i * NX + QTD]
            qce[i, 0] = x[i * NX + QCE]
            qiac[i, 0] = x[i * NX + QIAC]
            ecss[i, 0] = x[i * NX + ECSS]
            qcs_dc[i, 0] = x[i * NX + QCS_DC]
            qcs_ch[i, 0] = x[i * NX + QCS_CH]
            qcd[i, 0] = x[i * NX + QCD]
            vchp[i, 0] = x[i * NX + VCHP]
            vgas[i, 0] = x[i * NX + VGAS]

        # Formulate the solution
        sol = {"obj": objvalue,
               "PUG": pug,
               "PCHP": pchp,
               "PAC2DC": pac2dc,
               "PDC2AC": pdc2ac,
               "PIAC": piac,
               "EESS": eess,
               "PESS_CH": pess_ch,
               "PESS_DC": pess_dc,
               "PPV": ppv,
               "QCHP": qchp,
               "QGAS": qgas,
               "ETSS": etss,
               "QES_CH": qes_ch,
               "QES_DC": qes_dc,
               "QAC": qac,
               "QTD": qtd,
               "QCE": qce,
               "QIAC": qiac,
               "ECSS": ecss,
               "QCS_CH": qcs_ch,
               "QCS_DC": qcs_dc,
               "QCD": qcd,
               "VCHP": vchp,
               "VGAS": vgas,
               }

        # pyplot.plot(bic_relaxation)
        # pyplot.plot(ess_relaxation)
        # pyplot.plot(tes_relaxation)
        # pyplot.plot(ces_relaxation)
        # pyplot.show()

        return sol

    def solution_check(self, sol):
        # Check the relaxations
        bic_relaxation = np.multiply(sol["PAC2DC"], sol["PDC2AC"])
        ess_relaxation = np.multiply(sol["PESS_DC"], sol["PESS_CH"])
        tes_relaxation = np.multiply(sol["QES_CH"], sol["QES_DC"])
        ces_relaxation = np.multiply(sol["QCS_CH"], sol["QCS_DC"])

        sol_check = {"bic": bic_relaxation,
                     "ess": ess_relaxation,
                     "tes": tes_relaxation,
                     "ces": ces_relaxation}

        return sol_check


if __name__ == "__main__":
    # A test system
    # 1) System level configuration
    T = 24
    Delta_t = 1
    delat_t = 1
    T_second_stage = int(T / delat_t)

    # For the HVAC system
    # 2) Thermal system configuration
    QHVAC_max = 100
    eff_HVAC = 4

    CD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])
    HD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])

    # 3) Electricity system configuration
    PUG_MAX = 200
    PV_CAP = 50
    AC_PD_cap = 50
    DC_PD_cap = 50
    HD_cap = 100
    CD_cap = 100

    PESS_CH_MAX = 100
    PESS_DC_MAX = 100
    EFF_DC = 0.9
    EFF_CH = 0.9
    E0 = 50
    Emax = 100
    Emin = 20

    BIC_CAP = 100
    eff_BIC = 0.95

    electricity_price = array(
        [6.01, 73.91, 71.31, 69.24, 68.94, 70.56, 75.16, 73.19, 79.70, 85.76, 86.90, 88.60, 90.62, 91.26, 93.70, 90.94,
         91.26, 80.39, 76.25, 76.80, 81.22, 83.75, 76.16, 72.69])

    AC_PD = array([323.0284, 308.2374, 318.1886, 307.9809, 331.2170, 368.6539, 702.0040, 577.7045, 1180.4547, 1227.6240,
                   1282.9344, 1311.9738, 1268.9502, 1321.7436, 1323.9218, 1327.1464, 1386.9117, 1321.6387, 1132.0476,
                   1109.2701, 882.5698, 832.4520, 349.3568, 299.9920])
    DC_PD = array([287.7698, 287.7698, 287.7698, 287.7698, 299.9920, 349.3582, 774.4047, 664.0625, 1132.6996, 1107.7366,
                   1069.6837, 1068.9819, 1027.3295, 1096.3820, 1109.4778, 1110.7039, 1160.1270, 1078.7839, 852.2514,
                   791.5814, 575.4085, 551.1441, 349.3568, 299.992])
    PV_PG = array(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])

    ELEC_PRICE = electricity_price / 300

    Eess_cost = 0.01

    PV_PG = PV_PG * PV_CAP
    # Modify the first stage profiles
    AC_PD = (AC_PD / max(AC_PD)) * AC_PD_cap
    DC_PD = (DC_PD / max(DC_PD)) * DC_PD_cap
    HD = (HD / max(HD)) * HD_cap
    CD = (CD / max(CD)) * CD_cap

    # Generate the second stage profiles using spline of scipy
    Time_first_stage = arange(0, T, Delta_t)
    Time_second_stage = arange(0, T, delat_t)

    # AC_PD_tck = interpolate.splrep(Time_first_stage, AC_PD, s=0)
    # DC_PD_tck = interpolate.splrep(Time_first_stage, DC_PD, s=0)
    # PV_PG_tck = interpolate.splrep(Time_first_stage, PV_PG, s=0)
    #
    # AC_PD_second_stage = interpolate.splev(Time_second_stage, AC_PD_tck, der=0)
    # DC_PD_second_stage = interpolate.splev(Time_second_stage, DC_PD_tck, der=0)
    # PV_PG_second_stage = interpolate.splev(Time_second_stage, PV_PG_tck, der=0)
    #
    # for i in range(T_second_stage):
    #     if AC_PD_second_stage[i] < 0:
    #         AC_PD_second_stage[i] = 0
    #     if DC_PD_second_stage[i] < 0:
    #         DC_PD_second_stage[i] = 0
    #     if PV_PG_second_stage[i] < 0:
    #         PV_PG_second_stage[i] = 0

    # CCHP system
    Gas_price = 0.1892
    Gmax = 200
    eff_CHP_e = 0.3
    eff_CHP_h = 0.4
    # Boiler information
    Boil_max = 100
    eff_boil = 0.9
    # Chiller information
    Chiller_max = 100
    eff_chiller = 1.2

    CCHP = {"MAX": Gmax,
            "EFF_E": eff_CHP_e,
            "EFF_C": eff_CHP_h,
            "EFF_H": eff_CHP_h,
            "COST": Gas_price}

    HVAC = {"CAP": QHVAC_max,
            "EFF": eff_HVAC}

    THERMAL = {"HD": HD,
               "CD": CD, }

    ELEC = {"UG_MAX": PUG_MAX,
            "UG_MIN": -PUG_MAX,
            "UG_PRICE": ELEC_PRICE,
            "AC_PD": AC_PD,
            "DC_PD": DC_PD,
            "PV_PG": PV_PG
            }

    BIC = {"CAP": BIC_CAP,
           "EFF": eff_BIC,
           }

    BESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "PC_MAX": PESS_CH_MAX,
            "PD_MAX": PESS_DC_MAX,
            "EFF_CH": EFF_CH,
            "EFF_DC": EFF_DC,
            "COST": Eess_cost,
            }

    TESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "TC_MAX": PESS_CH_MAX,
            "TD_MAX": PESS_DC_MAX,
            "EFF_CH": EFF_CH,
            "EFF_DC": EFF_DC,
            "EFF_SD": 0.98,  # The self discharging
            "COST": Eess_cost,
            }

    CESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "TC_MAX": PESS_CH_MAX,
            "TD_MAX": PESS_DC_MAX,
            "EFF_CH": EFF_CH,
            "EFF_DC": EFF_DC,
            "EFF_SD": 0.98,  # The self discharging
            "COST": Eess_cost,
            "PMAX": PESS_CH_MAX * 10,
            "ICE": 3.5,
            }

    ESS = {"BESS": BESS,
           "TESS": TESS,
           "CESS": CESS}

    BOIL = {"CAP": Boil_max,
            "EFF": eff_boil}

    CHIL = {"CAP": Chiller_max,
            "EFF": eff_chiller}

    energy_hub_management = EnergyHubManagement()

    model = energy_hub_management.problem_formulation(ELEC=ELEC, CCHP=CCHP, THERMAL=THERMAL, BIC=BIC, ESS=ESS,
                                                      HVAC=HVAC, BOIL=BOIL, CHIL=CHIL, T=T)
    sol = energy_hub_management.problem_solving(model)

    sol_check = energy_hub_management.solution_check(sol)

    print(sol_check)
