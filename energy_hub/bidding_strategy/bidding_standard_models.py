"""
Standard bidding model for the energy hub problems
"""

from numpy import array, arange, zeros, ones, concatenate, multiply


class EnergyHubManagement():
    def __init__(self):
        self.name = "bidding modelling"

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
        Beq = concatenate(
            [Aeq_bess, Aeq_tess, Aeq_cess, Aeq_chp_e, Aeq_chp_h, Aeq_boil, Aeq_chil, Aeq_iac, Aeq_ice, Aeq_dc, Aeq_hh,
             Aeq_ch])
        f = concatenate(
            [beq_bess, beq_tess, beq_cess, beq_chp_e, beq_chp_h, beq_boil, beq_chil, beq_iac, beq_ice, beq_dc,
             beq_hh, beq_ch])

        D = zeros((2 * T, NX * T))
        h = zeros((2 * T, 1))
        h[0:T] = ELEC["UG_MAX"]
        h[T:] = -ELEC["UG_MIN"]

        for i in range(T):
            D[i, i * NX + PUG] = 1
            D[i + T, i * NX + PUG] = -1

        model = {"D": D,
                 "h": h,
                 "F": Aeq_ac,
                 "g": beq_ac,
                 "B": None,
                 "d": None,
                 "Beq": Beq,
                 "f": f,
                 "q": c,
                 "lb": lb,
                 "ub": ub,
                 }
        # for i in model:
        #     if model[i] is not None:
        #         model[i]=matrix(model[i])

        if model["h"] is None:
            model["nh"] = 0
        else:
            model["nh"] = model["h"].shape[0]

        if model["g"] is None:
            model["ng"] = 0
        else:
            model["ng"] = model["g"].shape[0]

        if model["d"] is None:
            model["nd"] = 0
        else:
            model["nd"] = model["d"].shape[0]

        if model["f"] is None:
            model["nf"] = 0
        else:
            model["nf"] = model["f"].shape[0]

        if model["q"] is None:
            model["nx"] = 0
        else:
            model["nx"] = model["q"].shape[0]

        return model

    def solution_update(self, model):
        """
        problem solving of the day-ahead energy hub
        :param model:
        :return:
        """
        from energy_hub.bidding_strategy.data_format import PUG, PCHP, PAC2DC, PDC2AC, PIAC, EESS, PESS_CH, PESS_DC, \
            PPV, PCS, QCHP, QGAS, EHSS, QHS_DC, QHS_CH, QAC, QTD, QCE, QIAC, ECSS, QCS_DC, QCS_CH, QCD, VCHP, VGAS, NX

        # Try to solve the linear programing problem
        try:
            T = self.T
        except:
            T = model["T"]
        x = model["x"]
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
        sol = {"PUG": pug,
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
        return sol

    def solution_check(self, sol):
        # Check the relaxations
        bic_relaxation = multiply(sol["PAC2DC"], sol["PDC2AC"])
        ess_relaxation = multiply(sol["PESS_DC"], sol["PESS_CH"])
        tes_relaxation = multiply(sol["QES_CH"], sol["QES_DC"])
        ces_relaxation = multiply(sol["QCS_CH"], sol["QCS_DC"])

        sol_check = {"bic": bic_relaxation,
                     "ess": ess_relaxation,
                     "tes": tes_relaxation,
                     "ces": ces_relaxation}

        return sol_check
