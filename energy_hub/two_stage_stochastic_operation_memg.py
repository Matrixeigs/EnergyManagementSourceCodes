"""
Two stage stochastic optimization problem for multi energy microgrid
@author: Zhao Tianyang
@e-mail:matrixeigs@gmail.com
"""
from gurobipy import *
import numpy as np
import os, platform
import pandas as pd


class TwoStageStochastic():
    def __init__(self):
        self.pwd = os.getcwd()

    def day_ahead_scheduling(self, AC_LOAD_MEAN, AC_LOAD_STD, DC_LOAD_MEAN, DC_LOAD_STD, PV_MEAN, PV_STD, HD_MEAN,
                             HD_STD, CD_MEAN, CD_STD, AT_MEAN, AT_STD, SCENARIO_UPDATE=1, N_S=100,
                             condifential_level=0.05, beta=0.05):
        """
        Day-ahead scheduling problem for memg
        :return:
        """
        bigM = 1e5
        T = len(AC_LOAD_MEAN)
        self.T = T
        # 1) Generate scenarios
        if SCENARIO_UPDATE > 0:  # We need to update the scenarios
            # The scenarios are stored by row.
            if platform.system() == "Windows":
                writer = pd.ExcelWriter(self.pwd + "\input_data.xlsx", float_format="10.4%f", index=False)
            else:
                writer = pd.ExcelWriter(self.pwd + "/input_data.xlsx", float_format="10.4%f", index=False)

            ac_load = self.scenario_generation(AC_LOAD_MEAN, AC_LOAD_STD, N_S)
            df = pd.DataFrame(ac_load)
            df.to_excel(writer, sheet_name="ac_load")
            dc_load = self.scenario_generation(DC_LOAD_MEAN, DC_LOAD_STD, N_S)
            df = pd.DataFrame(dc_load)
            df.to_excel(writer, sheet_name="dc_load")
            pv = self.scenario_generation(PV_MEAN, PV_STD, N_S)
            df = pd.DataFrame(pv)
            df.to_excel(writer, sheet_name="pv")
            hd = self.scenario_generation(HD_MEAN, HD_STD, N_S)
            df = pd.DataFrame(hd)
            df.to_excel(writer, sheet_name="hd")
            cd = self.scenario_generation(CD_MEAN, CD_STD, N_S)
            df = pd.DataFrame(cd)
            df.to_excel(writer, sheet_name="cd")
            at = self.scenario_generation(AT_MEAN, AT_STD, N_S)
            df = pd.DataFrame(at)
            df.to_excel(writer, sheet_name="ambinent_temprature")
            # Sample the vehicle driving pattern, to generate the arrival pattern and minimal departure pattern
            Arrival_curve = np.zeros((N_S, T))
            Minimal_departure_curve = np.zeros((N_S, T))
            ev_scale = np.zeros((N_S, T))
            for i in range(N_S):
                ev_sample_arr = np.zeros((NEV, T))
                ev_sample_dep = np.zeros((NEV, T))
                ev_sample = np.zeros((NEV, T))
                for j in range(NEV):
                    arrival_time = max(int(np.ceil(np.random.normal(Arrival_mean, Arrival_std))), 8)
                    departure_time = int(min(np.ceil(np.random.normal(Dep_mean, Dep_std)), T - 1))
                    departure_time = max(arrival_time, departure_time)
                    energy_ev = min(np.random.normal(SOC_mean, SOC_std) * EEV,
                                    (departure_time - arrival_time) * PEV_max)
                    ev_sample[j, arrival_time:departure_time] = 1
                    for k in range(arrival_time, departure_time):
                        ev_sample_arr[j, k] = min(energy_ev, PEV_max)
                        ev_sample_dep[j, departure_time - k + arrival_time] = min(energy_ev, PEV_max)
                        energy_ev = max(energy_ev - PEV_max, 0)

                Arrival_curve[i, :] = np.cumsum(np.sum(ev_sample_arr, axis=0))
                Minimal_departure_curve[i, :] = np.cumsum(np.sum(ev_sample_dep, axis=0))
                ev_scale[i, :] = np.sum(ev_sample, axis=0)
                # Gap = Arrival_curve-Minimal_departure_curve
            df = pd.DataFrame(Arrival_curve)
            df.to_excel(writer, sheet_name="Arrival_curve")
            df = pd.DataFrame(Minimal_departure_curve)
            df.to_excel(writer, sheet_name="Minimal_departure_curve")
            df = pd.DataFrame(ev_scale)
            df.to_excel(writer, sheet_name="ev_scale")
            # df = pd.DataFrame(Gap)
            # df.to_excel(writer, sheet_name="Gap")

            writer.save()
        else:
            # Load pre-generated sources
            ac_load = pd.read_excel(self.pwd + "/input_data.xlsx", sheet_name="ac_load", index_col=0).values
            dc_load = pd.read_excel(self.pwd + "/input_data.xlsx", sheet_name="dc_load", index_col=0).values
            pv = pd.read_excel(self.pwd + "/input_data.xlsx", sheet_name="pv", index_col=0).values
            hd = pd.read_excel(self.pwd + "/input_data.xlsx", sheet_name="hd", index_col=0).values
            cd = pd.read_excel(self.pwd + "/input_data.xlsx", sheet_name="cd", index_col=0).values
            at = pd.read_excel(self.pwd + "/input_data.xlsx", sheet_name="ambinent_temprature", index_col=0).values
            Arrival_curve = pd.read_excel(self.pwd + "/input_data.xlsx", sheet_name="Arrival_curve", index_col=0).values
            Minimal_departure_curve = pd.read_excel(self.pwd + "/input_data.xlsx", sheet_name="Minimal_departure_curve",
                                                    index_col=0).values
            ev_scale = pd.read_excel(self.pwd + "/input_data.xlsx", sheet_name="ev_scale", index_col=0).values
            N_S = ev_scale.shape[0]

        # 2) Problem formulation
        model = Model("MEMG")
        # 2.1) Decision variables
        PDA = {}  # Day-ahead bidding strategy
        ICHP = {}  # Start-up and shut-down of gas turbine
        ALPHA_CHP = {}  # Start-up and shut-down of gas turbine
        BETA_CHP = {}  # Start-up and shut-down of gas turbine

        pRT = {}  # Real-time prices
        pRT_positive = {}  # Real-time prices
        pRT_negative = {}  # Real-time prices
        pA2D = {}  # Real-time power transferred from AC to DC
        pD2A = {}  # Real-time power transferred from DC to AC
        IA2D = {}  # AC to DC index
        eESS = {}  # Real-time energy status
        pES_DC = {}  # ESS discharging rate
        IES_DC = {}  # ESS discharging rate
        pES_CH = {}  # ESS charging rate
        pHVAC = {}  # HVAC consumption
        pPV = {}  # PV output
        pP2G = {}  # Power to Gas rate
        pEV = {}  # Power to Gas rate
        ## Group 2: Heating ##
        eHSS = {}
        qHS_DC = {}
        qHS_CH = {}
        IHS_DC = {}
        qHD = {}
        qAC = {}
        ## Group 3: Cooling ##
        eCSS = {}
        qCS_DC = {}
        qCS_CH = {}
        ICS_DC = {}
        qCD = {}
        ## Group 4: Gas ##
        vCHP = {}
        v = {}
        vBoil = {}
        vGS_DC = {}
        vGS_CH = {}
        IGS_DC = {}
        eGSS = {}
        eGV = {}
        # Group 5: Temperature in side the routine
        temperature_in = {}
        # Group 6: feasible control
        Temperature_index = {}
        EV_index = {}
        GV_index = {}

        # Group CVaR
        ru = model.addVar(name="ru")
        var = {}
        ## The first stage decision variables
        for i in range(T):
            PDA[i] = model.addVar(lb=-PUG_MAX, ub=PUG_MAX, name="PDA{0}".format(i))
            ALPHA_CHP[i] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="ALPHA_CHP{0}".format(i))
            BETA_CHP[i] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="BETA_CHP{0}".format(i))
            ICHP[i] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="ICHP{0}".format(i))
        ## The second stage decision variables
        for i in range(T):
            for j in range(N_S):
                # Electrical
                pRT[i, j] = model.addVar(lb=-PUG_MAX, ub=PUG_MAX, name="pRT{0}".format(i * N_S + j))
                pRT_positive[i, j] = model.addVar(lb=0, ub=PUG_MAX, name="pRT_positive{0}".format(i * N_S + j))
                pRT_negative[i, j] = model.addVar(lb=0, ub=PUG_MAX, name="pRT_negative{0}".format(i * N_S + j))
                pA2D[i, j] = model.addVar(lb=0, ub=BIC_CAP, name="pA2D{0}".format(i * N_S + j))
                pD2A[i, j] = model.addVar(lb=0, ub=BIC_CAP, name="pD2A{0}".format(i * N_S + j))
                IA2D[i, j] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="IA2D{0}".format(i * N_S + j))
                eESS[i, j] = model.addVar(lb=EESS_MIN, ub=EESS_MAX, name="eESS{0}".format(i * N_S + j))
                pES_DC[i, j] = model.addVar(lb=0, ub=PESS_DC_MAX, name="pES_DC{0}".format(i * N_S + j))
                pES_CH[i, j] = model.addVar(lb=0, ub=PESS_CH_MAX, name="pES_CH{0}".format(i * N_S + j))
                IES_DC[i, j] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="IES_DC{0}".format(i * N_S + j))
                pHVAC[i, j] = model.addVar(lb=0, ub=QHVAC_max, name="pHVAC{0}".format(i * N_S + j))  # using AC machines
                pPV[i, j] = model.addVar(lb=0, ub=pv[j, i], name="pPV{0}".format(i * N_S + j))
                pP2G[i, j] = model.addVar(lb=0, ub=PP2G_cap, name="pP2G{0}".format(i * N_S + j))
                pEV[i, j] = model.addVar(lb=0, ub=PEV_max * NEV, name="pEV{0}".format(i * N_S + j))
                # Heating
                eHSS[i, j] = model.addVar(lb=EHSS_MIN, ub=EHSS_MAX, name="eHSS{0}".format(i * N_S + j))
                qHS_DC[i, j] = model.addVar(lb=0, ub=PHSS_DC_MAX, name="qHS_DC{0}".format(i * N_S + j))
                qHS_CH[i, j] = model.addVar(lb=0, ub=PHSS_CH_MAX, name="qHS_CH{0}".format(i * N_S + j))
                IHS_DC[i, j] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="IHS_DC{0}".format(i * N_S + j))
                qHD[i, j] = model.addVar(lb=0, ub=QHVAC_max, name="qHD{0}".format(i * N_S + j))
                qAC[i, j] = model.addVar(lb=0, ub=Chiller_max, name="qAC{0}".format(i * N_S + j))
                # Cooling
                eCSS[i, j] = model.addVar(lb=ECSS_MIN, ub=ECSS_MAX, name="eCSS{0}".format(i * N_S + j))
                qCS_DC[i, j] = model.addVar(lb=0, ub=PCSS_DC_MAX, name="qCS_DC{0}".format(i * N_S + j))
                qCS_CH[i, j] = model.addVar(lb=0, ub=PCSS_CH_MAX, name="qCS_CH{0}".format(i * N_S + j))
                ICS_DC[i, j] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="ICS_DC{0}".format(i * N_S + j))
                qCD[i, j] = model.addVar(lb=0, ub=QHVAC_max, name="qCD{0}".format(i * N_S + j))
                # Gas
                v[i, j] = model.addVar(lb=0, ub=Gmax * 10, name="v{0}".format(i * N_S + j))
                vBoil[i, j] = model.addVar(lb=0, ub=G_boiler_cap, name="vBoil{0}".format(i * N_S + j))
                vCHP[i, j] = model.addVar(lb=0, ub=Gmax, name="vCHP{0}".format(i * N_S + j))
                vGS_DC[i, j] = model.addVar(lb=0, ub=PGSS_DC_MAX, name="vGS_DC{0}".format(i * N_S + j))
                vGS_CH[i, j] = model.addVar(lb=0, ub=PGSS_CH_MAX, name="vGS_CH{0}".format(i * N_S + j))
                eGSS[i, j] = model.addVar(lb=EGSS_MIN, ub=EGSS_MAX, name="eGSS{0}".format(i * N_S + j))
                IGS_DC[i, j] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="IGS_DC{0}".format(i * N_S + j))
                eGV[i, j] = model.addVar(lb=0, ub=PGV_max * NGV, name="eGV{0}".format(i * N_S + j))
                # Temperature
                temperature_in[i, j] = model.addVar(lb=-1e3, ub=1e3, name="temperature_in{0}".format(i * N_S + j))

        for i in range(N_S):
            var[i] = model.addVar(lb=0, name="var{0}".format(i))
            Temperature_index[i] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="Temperature_index{0}".format(i))
            EV_index[i] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="EV_index{0}".format(i))
            GV_index[i] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="GV_index{0}".format(i))

        # 2.2) First-stage constraint set
        # 1) Eq.(4) is satisfied automatically
        # 2) Eq.(5)
        for i in range(T):
            if i == 0:
                model.addConstr(ALPHA_CHP[i] - BETA_CHP[i] == ICHP[i] - ICHP0)
            else:
                model.addConstr(ALPHA_CHP[i] - BETA_CHP[i] == ICHP[i] - ICHP[i - 1])
        # 3) Eq.(6):
        for i in range(UT, T):
            expr = 0
            for j in range(i - UT, i):
                expr += ALPHA_CHP[j]
            model.addConstr(expr <= ICHP[i])
        # 4) Eq.(7):
        for i in range(DT, T):
            expr = 0
            for j in range(i - DT, i):
                expr += BETA_CHP[j]
            model.addConstr(expr + ICHP[i] <= 1)
        # 2.3) Second-stage constraint set
        ### Electrical part
        # Eq.(8):
        for i in range(T):
            for j in range(N_S):
                model.addConstr(
                    PDA[i] + pRT[i, j] + eff_CHP_e * vCHP[i, j] + eff_BIC_D2A * pD2A[i, j] == ac_load[j, i] + pA2D[
                        i, j] + pHVAC[i, j] + pP2G[i, j])
        # Eq.(9):
        for i in range(T):
            for j in range(N_S):
                model.addConstr(
                    pES_DC[i, j] - pES_CH[i, j] + eff_BIC_A2D * pA2D[i, j] + pPV[i, j] == dc_load[j, i] + pD2A[i, j] +
                    pEV[i, j])
        # Eq.(10)-(11):
        for i in range(T):
            for j in range(N_S):
                model.addConstr(pA2D[i, j] <= IA2D[i, j] * BIC_CAP)
                model.addConstr(pD2A[i, j] <= (1 - IA2D[i, j]) * BIC_CAP)
        # Eq.(12)-(16):
        for i in range(T):
            for j in range(N_S):
                model.addConstr(pES_DC[i, j] <= IES_DC[i, j] * PESS_DC_MAX)
                model.addConstr(pES_CH[i, j] <= (1 - IES_DC[i, j]) * PESS_CH_MAX)
                if i == 0:
                    model.addConstr(eESS[i, j] - EESS0 == pES_CH[i, j] * eff_PESS_CH - pES_DC[i, j] / eff_PESS_DC)
                else:
                    model.addConstr(
                        eESS[i, j] - eESS[i - 1, j] == pES_CH[i, j] * eff_PESS_CH - pES_DC[i, j] / eff_PESS_DC)

        for i in range(N_S): model.addConstr(eESS[T - 1, i] == EESS0)
        # EV part
        for i in range(T):
            for j in range(N_S):
                rhs = 0
                for k in range(i + 1):
                    rhs += pEV[k, j]
                model.addConstr(rhs <= Arrival_curve[j, i])

                if i == T - 1:
                    model.addConstr(rhs >= Minimal_departure_curve[j, i])
                else:
                    model.addConstr(rhs >= Minimal_departure_curve[j, i] - EV_index[j] * bigM)

        for i in range(T):
            for j in range(N_S):
                model.addConstr(pEV[i, j] <= ev_scale[j, i] * PEV_max + EV_index[j] * bigM)

        ###Thermal part
        # Eq.(21)
        for i in range(T):
            for j in range(N_S):
                model.addConstr(
                    qHD[i, j] + hd[j, i] + qAC[i, j] + qHS_CH[i, j] == eff_CHP_h * vCHP[i, j] + qHS_DC[i, j] + vBoil[
                        i, j] * eff_boiler)
        # Eq.(22)
        for i in range(T):
            for j in range(N_S):
                model.addConstr(
                    qCD[i, j] + cd[j, i] + qCS_CH[i, j] == qCS_DC[i, j] + eff_HVAC * pHVAC[i, j] + eff_chiller * qAC[
                        i, j])
        # Additional constraints for heating storage and cooling storage
        for i in range(T):
            for j in range(N_S):
                model.addConstr(qHS_DC[i, j] <= IHS_DC[i, j] * PESS_DC_MAX)
                model.addConstr(qHS_CH[i, j] <= (1 - IHS_DC[i, j]) * PESS_CH_MAX)
                if i == 0:
                    model.addConstr(
                        eHSS[i, j] - (1 - eff_HSS) * EHSS0 == qHS_CH[i, j] * eff_HSS_CH - qHS_DC[i, j] / eff_HSS_DC)
                else:
                    model.addConstr(
                        eHSS[i, j] - (1 - eff_HSS) * eHSS[i - 1, j] == qHS_CH[i, j] * eff_HSS_CH - qHS_DC[
                            i, j] / eff_HSS_DC)

        for i in range(N_S): model.addConstr(eHSS[T - 1, i] == EHSS0)

        for i in range(T):
            for j in range(N_S):
                model.addConstr(qCS_DC[i, j] <= ICS_DC[i, j] * PESS_DC_MAX)
                model.addConstr(qCS_CH[i, j] <= (1 - ICS_DC[i, j]) * PESS_CH_MAX)
                if i == 0:
                    model.addConstr(
                        eCSS[i, j] - (1 - eff_CSS) * ECSS0 == qCS_CH[i, j] * eff_CSS_CH - qCS_DC[i, j] / eff_CSS_DC)
                else:
                    model.addConstr(
                        eCSS[i, j] - (1 - eff_CSS) * eCSS[i - 1, j] == qCS_CH[i, j] * eff_CSS_CH - qCS_DC[
                            i, j] / eff_CSS_DC)

        for i in range(N_S): model.addConstr(eCSS[T - 1, i] == ECSS0)

        # Eq.(23)
        for i in range(T):
            for j in range(N_S):
                if i > 0:
                    model.addConstr(
                        qHD[i, j] - qCD[i, j] == c_air * (temperature_in[i, j] - temperature_in[i - 1, j]) - (
                                at[j, i] - temperature_in[i, j]) / r_t)
                else:
                    model.addConstr(temperature_in[i, j] == (temprature_in_min + temprature_in_max) / 2)
        # Eq.(24)
        for i in range(T):
            for j in range(N_S):
                model.addConstr(temperature_in[i, j] >= temprature_in_min - Temperature_index[j] * bigM)
                model.addConstr(temperature_in[i, j] <= Temperature_index[j] * bigM + temprature_in_max)

        ###Gas part
        # Eq.(25)
        for i in range(T):
            for j in range(N_S):
                model.addConstr(eff_P2G * pP2G[i, j] + v[i, j] + vGS_DC[i, j] == vCHP[i, j] + vGS_CH[i, j] + eGV[i, j])
        # GV part
        for i in range(T):
            for j in range(N_S):
                rhs = 0
                for k in range(i + 1):
                    rhs += eGV[k, j]
                model.addConstr(rhs <= Arrival_curve[j, i])
                if i == T - 1:
                    model.addConstr(rhs >= Minimal_departure_curve[j, i])
                else:
                    model.addConstr(rhs >= Minimal_departure_curve[j, i] - GV_index[j] * bigM)

        for i in range(T):
            for j in range(N_S):
                model.addConstr(eGV[i, j] <= ev_scale[j, i] * PGV_max + GV_index[j] * bigM)

        # Gas storage
        for i in range(T):
            for j in range(N_S):
                model.addConstr(vGS_DC[i, j] <= IGS_DC[i, j] * PESS_DC_MAX)
                model.addConstr(vGS_CH[i, j] <= (1 - IGS_DC[i, j]) * PESS_CH_MAX)
                if i == 0:
                    model.addConstr(
                        eGSS[i, j] - (1 - eff_GSS) * EGSS0 == vGS_CH[i, j] * eff_GSS_CH - vGS_DC[i, j] / eff_GSS_CH)
                else:
                    model.addConstr(
                        eGSS[i, j] - (1 - eff_GSS) * eGSS[i - 1, j] == vGS_CH[i, j] * eff_GSS_CH - vGS_DC[
                            i, j] / eff_GSS_CH)

        for i in range(N_S): model.addConstr(eGSS[T - 1, i] == EGSS0)
        # Eq.(28)
        for i in range(T):
            for j in range(N_S):
                model.addConstr(vCHP[i, j] <= ICHP[i] * Gmax)
                model.addConstr(vCHP[i, j] >= ICHP[i] * 0)

        # Relaxed Temperature constraints
        rhs = 0
        for i in range(N_S):
            rhs += Temperature_index[i]
            rhs += EV_index[i]
            rhs += GV_index[i]
        model.addConstr(rhs <= beta * N_S)

        for i in range(N_S):
            model.addConstr(Temperature_index[i] + EV_index[i] + GV_index[i] <= 1)

        model.optimize()
        # Objective function
        for i in range(T):
            for j in range(N_S):
                model.addConstr(pRT_negative[i, j] >= -pRT[i, j])
                model.addConstr(pRT_positive[i, j] >= pRT[i, j])

        obj_DA = 0
        for i in range(T):
            obj_DA += electricity_price_DA[i] * PDA[i] + ALPHA_CHP[i] * SU + BETA_CHP[i] * SD + ICOST * ICHP[i]
        obj_RT = 0
        for i in range(T):
            for j in range(N_S):
                obj_RT += electricity_price[i] * pRT[i, j] / N_S
                obj_RT += electricity_price[i] * pRT_positive[i, j] / N_S
                obj_RT += electricity_price[i] * pRT_negative[i, j] / N_S
                obj_RT += Gas_price * v[i, j] / N_S
        # Add CVaR
        for i in range(N_S):
            rhs = 0
            for j in range(T):
                rhs += electricity_price[j] * pRT[j, i]
                rhs += electricity_price[j] * pRT_positive[j, i]
                rhs += electricity_price[j] * pRT_negative[j, i]
                rhs += Gas_price * v[j, i]
            model.addConstr(rhs + obj_DA - ru <= var[i])

        obj_CVaR = ru
        for i in range(N_S):
            obj_CVaR += var[i] / (1 - condifential_level)
        # 3) Problem solving
        model.setObjective(obj_DA + obj_RT + 0 * obj_CVaR)
        model.Params.MIPGap = 5 * 10 ** -3
        model.optimize()
        # 4) Save the results
        PDA = np.zeros(T)  # Day-ahead bidding strategy
        ICHP = np.zeros(T)  # Start-up and shut-down of gas turbine
        ALPHA_CHP = np.zeros(T)  # Start-up and shut-down of gas turbine
        BETA_CHP = np.zeros(T)  # Start-up and shut-down of gas turbine
        for i in range(T):
            PDA[i] = model.getVarByName("PDA{0}".format(i)).X
            ICHP[i] = model.getVarByName("ICHP{0}".format(i)).X
            ALPHA_CHP[i] = model.getVarByName("ALPHA_CHP{0}".format(i)).X
            BETA_CHP[i] = model.getVarByName("BETA_CHP{0}".format(i)).X

        pRT = np.zeros((T, N_S))  # Real-time prices
        pRT_positive = np.zeros((T, N_S))  # Real-time prices
        pRT_negative = np.zeros((T, N_S))  # Real-time prices
        pA2D = np.zeros((T, N_S))  # Real-time power transferred from AC to DC
        pD2A = np.zeros((T, N_S))  # Real-time power transferred from DC to AC
        IA2D = np.zeros((T, N_S))  # AC to DC index
        eESS = np.zeros((T, N_S))  # Real-time energy status
        pES_DC = np.zeros((T, N_S))  # ESS discharging rate
        IES_DC = np.zeros((T, N_S))  # ESS discharging rate
        pES_CH = np.zeros((T, N_S))  # ESS charging rate
        pHVAC = np.zeros((T, N_S))  # HVAC consumption
        pP2G = np.zeros((T, N_S))  # PV output
        pPV = np.zeros((T, N_S))  # PV output
        pEV = np.zeros((T, N_S))  # EV charging
        ## Group 2: Heating ##
        eHSS = np.zeros((T, N_S))
        qHS_DC = np.zeros((T, N_S))
        qHS_CH = np.zeros((T, N_S))
        IHS_DC = np.zeros((T, N_S))
        qHD = np.zeros((T, N_S))
        qAC = np.zeros((T, N_S))
        ## Group 3: Cooling ##
        eCSS = np.zeros((T, N_S))
        qCS_DC = np.zeros((T, N_S))
        qCS_CH = np.zeros((T, N_S))
        ICS_DC = np.zeros((T, N_S))
        qCD = np.zeros((T, N_S))
        ## Group 4: Gas ##
        vCHP = np.zeros((T, N_S))
        v = np.zeros((T, N_S))
        vGS_DC = np.zeros((T, N_S))
        vGS_CH = np.zeros((T, N_S))
        IGS_DC = np.zeros((T, N_S))
        eGSS = np.zeros((T, N_S))
        eGV = np.zeros((T, N_S))
        # Group 5: Temperature in side the routine
        temperature_in = np.zeros((T, N_S))
        Temperature_index = np.zeros(N_S)
        EV_index = np.zeros(N_S)
        GV_index = np.zeros(N_S)

        for i in range(T):
            for j in range(N_S):
                pRT[i, j] = model.getVarByName("pRT{0}".format(i * N_S + j)).X
                pRT_positive[i, j] = model.getVarByName("pRT_positive{0}".format(i * N_S + j)).X
                pRT_negative[i, j] = model.getVarByName("pRT_negative{0}".format(i * N_S + j)).X
                pA2D[i, j] = model.getVarByName("pA2D{0}".format(i * N_S + j)).X
                pD2A[i, j] = model.getVarByName("pD2A{0}".format(i * N_S + j)).X
                IA2D[i, j] = model.getVarByName("IA2D{0}".format(i * N_S + j)).X
                eESS[i, j] = model.getVarByName("eESS{0}".format(i * N_S + j)).X
                pES_DC[i, j] = model.getVarByName("pES_DC{0}".format(i * N_S + j)).X
                IES_DC[i, j] = model.getVarByName("IES_DC{0}".format(i * N_S + j)).X
                pES_CH[i, j] = model.getVarByName("pES_CH{0}".format(i * N_S + j)).X
                pHVAC[i, j] = model.getVarByName("pHVAC{0}".format(i * N_S + j)).X
                pPV[i, j] = model.getVarByName("pPV{0}".format(i * N_S + j)).X
                pEV[i, j] = model.getVarByName("pEV{0}".format(i * N_S + j)).X
                pP2G[i, j] = model.getVarByName("pP2G{0}".format(i * N_S + j)).X

                eHSS[i, j] = model.getVarByName("eHSS{0}".format(i * N_S + j)).X
                qHS_DC[i, j] = model.getVarByName("qHS_DC{0}".format(i * N_S + j)).X
                qHS_CH[i, j] = model.getVarByName("qHS_CH{0}".format(i * N_S + j)).X
                qHD[i, j] = model.getVarByName("qHD{0}".format(i * N_S + j)).X
                qAC[i, j] = model.getVarByName("qAC{0}".format(i * N_S + j)).X
                IHS_DC[i, j] = model.getVarByName("IHS_DC{0}".format(i * N_S + j)).X
                eCSS[i, j] = model.getVarByName("eCSS{0}".format(i * N_S + j)).X
                qCS_DC[i, j] = model.getVarByName("qCS_DC{0}".format(i * N_S + j)).X
                qCS_CH[i, j] = model.getVarByName("qCS_CH{0}".format(i * N_S + j)).X
                ICS_DC[i, j] = model.getVarByName("ICS_DC{0}".format(i * N_S + j)).X
                qCD[i, j] = model.getVarByName("qCD{0}".format(i * N_S + j)).X
                vCHP[i, j] = model.getVarByName("vCHP{0}".format(i * N_S + j)).X
                v[i, j] = model.getVarByName("v{0}".format(i * N_S + j)).X
                vGS_DC[i, j] = model.getVarByName("vGS_DC{0}".format(i * N_S + j)).X
                vGS_CH[i, j] = model.getVarByName("vGS_CH{0}".format(i * N_S + j)).X
                IGS_DC[i, j] = model.getVarByName("IGS_DC{0}".format(i * N_S + j)).X
                eGSS[i, j] = model.getVarByName("eGSS{0}".format(i * N_S + j)).X
                eGV[i, j] = model.getVarByName("eGV{0}".format(i * N_S + j)).X

                temperature_in[i, j] = model.getVarByName("temperature_in{0}".format(i * N_S + j)).X

        for i in range(N_S):
            Temperature_index[i] = model.getVarByName("Temperature_index{0}".format(i)).X
            EV_index[i] = model.getVarByName("EV_index{0}".format(i)).X
            GV_index[i] = model.getVarByName("GV_index{0}".format(i)).X

        obj_DA = np.zeros(1)
        for i in range(T):
            obj_DA += electricity_price_DA[i] * PDA[i] + ALPHA_CHP[i] * SU + BETA_CHP[i] * SD + ICOST * ICHP[i]

        obj_RT = np.zeros(N_S)
        for i in range(T):
            for j in range(N_S):
                obj_RT[j] += electricity_price[i] * pRT[i, j] / N_S
                obj_RT[j] += electricity_price[i] * pRT_positive[i, j] / N_S
                obj_RT[j] += electricity_price[i] * pRT_negative[i, j] / N_S
                obj_RT[j] += Gas_price * v[i, j] / N_S

        # save results to the files
        if platform.system() == "Windows":
            writer = pd.ExcelWriter(self.pwd + r"\result.xlsx", float_format="10.4%f", index=False)
        else:
            writer = pd.ExcelWriter(self.pwd + "/result.xlsx", float_format="10.4%f", index=False)
        df = pd.DataFrame(obj_DA + sum(obj_RT))
        df.to_excel(writer, sheet_name="obj")
        df = pd.DataFrame(obj_DA)
        df.to_excel(writer, sheet_name="obj_DA")
        df = pd.DataFrame(obj_RT)
        df.to_excel(writer, sheet_name="obj_RT")
        df = pd.DataFrame(Temperature_index)
        df.to_excel(writer, sheet_name="Temperature_index")
        df = pd.DataFrame(EV_index)
        df.to_excel(writer, sheet_name="EV_index")
        df = pd.DataFrame(GV_index)
        df.to_excel(writer, sheet_name="GV_index")

        df = pd.DataFrame(PDA)
        df.to_excel(writer, sheet_name="PDA")
        df = pd.DataFrame(ICHP)
        df.to_excel(writer, sheet_name="ICHP")
        df = pd.DataFrame(ALPHA_CHP)
        df.to_excel(writer, sheet_name="ALPHA_CHP")
        df = pd.DataFrame(BETA_CHP)
        df.to_excel(writer, sheet_name="BETA_CHP")

        df = pd.DataFrame(pRT)
        df.to_excel(writer, sheet_name="pRT")
        df = pd.DataFrame(pA2D)
        df.to_excel(writer, sheet_name="pA2D")
        df = pd.DataFrame(pD2A)
        df.to_excel(writer, sheet_name="pD2A")
        df = pd.DataFrame(IA2D)
        df.to_excel(writer, sheet_name="IA2D")
        df = pd.DataFrame(pP2G)
        df.to_excel(writer, sheet_name="pP2G")
        df = pd.DataFrame(eESS)
        df.to_excel(writer, sheet_name="eESS")
        df = pd.DataFrame(pES_DC)
        df.to_excel(writer, sheet_name="pES_DC")
        df = pd.DataFrame(pES_CH)
        df.to_excel(writer, sheet_name="pES_CH")
        df = pd.DataFrame(qHD)
        df.to_excel(writer, sheet_name="qHD")
        df = pd.DataFrame(qAC)
        df.to_excel(writer, sheet_name="qAC")
        df = pd.DataFrame(pPV)
        df.to_excel(writer, sheet_name="pPV")
        df = pd.DataFrame(pEV)
        df.to_excel(writer, sheet_name="pEV")
        df = pd.DataFrame(eHSS)
        df.to_excel(writer, sheet_name="eHSS")
        df = pd.DataFrame(qHS_DC)
        df.to_excel(writer, sheet_name="qHS_DC")
        df = pd.DataFrame(qHS_CH)
        df.to_excel(writer, sheet_name="qHS_CH")
        df = pd.DataFrame(eCSS)
        df.to_excel(writer, sheet_name="eCSS")
        df = pd.DataFrame(IHS_DC)
        df.to_excel(writer, sheet_name="IHS_DC")
        df = pd.DataFrame(qCS_DC)
        df.to_excel(writer, sheet_name="qCS_DC")
        df = pd.DataFrame(qCS_CH)
        df.to_excel(writer, sheet_name="qCS_CH")
        df = pd.DataFrame(ICS_DC)
        df.to_excel(writer, sheet_name="ICS_DC")
        df = pd.DataFrame(qCD)
        df.to_excel(writer, sheet_name="qCD")
        df = pd.DataFrame(vCHP)
        df.to_excel(writer, sheet_name="vCHP")
        df = pd.DataFrame(v)
        df.to_excel(writer, sheet_name="v")
        df = pd.DataFrame(vGS_DC)
        df.to_excel(writer, sheet_name="vGS_DC")
        df = pd.DataFrame(vGS_CH)
        df.to_excel(writer, sheet_name="vGS_CH")
        df = pd.DataFrame(IGS_DC)
        df.to_excel(writer, sheet_name="IGS_DC")
        df = pd.DataFrame(eGSS)
        df.to_excel(writer, sheet_name="eGSS")
        df = pd.DataFrame(eGV)
        df.to_excel(writer, sheet_name="eGV")
        df = pd.DataFrame(temperature_in)
        df.to_excel(writer, sheet_name="temperature_in")

        writer.save()

        return

    def scenario_generation(self, MEAN_VALUE, STD, N_S):
        scenarios = np.zeros((N_S, self.T))
        for i in range(N_S):
            for j in range(self.T):
                scenarios[i, j] = max(np.random.normal(MEAN_VALUE[j], STD[j]), 0)

        return scenarios


if __name__ == "__main__":
    ## Capacity Configuration
    PV_cap = 10000
    AC_PD_cap = 5000
    DC_PD_cap = 5000
    HD_cap = 8000
    CD_cap = 5000
    GAS_cap = 5000
    NEV = 1000
    NGV = 1000
    # Chiller information
    Chiller_max = 5000
    eff_chiller = 1.2

    # Electricity system configuration
    PUG_MAX = 5000

    BIC_CAP = 5000
    eff_BIC_A2D = 0.95
    eff_BIC_D2A = 0.95

    G_boiler_cap = 8000
    eff_boiler = 0.75

    PP2G_cap = 14000
    eff_P2G = 0.6

    PCHP_cap = 6000
    eff_CHP_e = 0.35
    eff_CHP_h = 0.35

    PESS_CH_MAX = 800
    PESS_DC_MAX = 1000
    eff_PESS_CH = 0.9
    eff_PESS_DC = 0.9
    eff_PESS = 0.001
    EESS_MIN = 0.2 * 2000
    EESS_MAX = 0.9 * 2000
    EESS0 = 0.5 * 2000

    PHSS_CH_MAX = 1500
    PHSS_DC_MAX = 2100
    eff_HSS_CH = 0.85
    eff_HSS_DC = 0.85
    eff_HSS = 0.005
    EHSS_MIN = 0.2 * 2500
    EHSS_MAX = 0.9 * 2500
    EHSS0 = 0.5 * 2500

    PCSS_CH_MAX = 1500
    PCSS_DC_MAX = 2100
    eff_CSS_CH = 0.85
    eff_CSS_DC = 0.85
    eff_CSS = 0.005
    ECSS_MIN = 0.2 * 2500
    ECSS_MAX = 0.9 * 2500
    ECSS0 = 0.5 * 2500

    PGSS_CH_MAX = 1250
    PGSS_DC_MAX = 1500
    eff_GSS_CH = 0.95
    eff_GSS_DC = 0.95
    eff_GSS = 0.003
    EGSS_MIN = 0.2 * 3000
    EGSS_MAX = 0.9 * 3000
    EGSS0 = 0.5 * 3000

    # AC electrical demand
    AC_PD = np.array(
        [323.0284, 308.2374, 318.1886, 307.9809, 331.2170, 368.6539, 702.0040, 577.7045, 1180.4547, 1227.6240,
         1282.9344, 1311.9738, 1268.9502, 1321.7436, 1323.9218, 1327.1464, 1386.9117, 1321.6387, 1132.0476,
         1109.2701, 882.5698, 832.4520, 349.3568, 299.9920])
    AC_PD = AC_PD / max(AC_PD) * AC_PD_cap

    # DC electrical demand
    DC_PD = np.array(
        [287.7698, 287.7698, 287.7698, 287.7698, 299.9920, 349.3582, 774.4047, 664.0625, 1132.6996, 1107.7366,
         1069.6837, 1068.9819, 1027.3295, 1096.3820, 1109.4778, 1110.7039, 1160.1270, 1078.7839, 852.2514,
         791.5814, 575.4085, 551.1441, 349.3568, 299.992])
    DC_PD = DC_PD / max(DC_PD) * DC_PD_cap

    # Heating demand
    HD = np.array(
        [16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
         86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
         0, 0])
    HD = HD / max(HD) * HD_cap

    # Cooling demand
    CD = np.array(
        [16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
         86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
         0, 0])
    CD = CD / max(CD) * CD_cap

    # PV load profile
    PV_PG = np.array(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])
    PV_PG = PV_PG / max(PV_PG) * PV_cap

    electricity_price_DA = np.array(
        [6.01, 75.91, 73.31, 71.24, 70.94, 69.56, 74.16, 72.19, 80.70, 86.76, 85.90, 87.60, 91.62, 90.26, 95.70, 87.94,
         91.26, 82.39, 75.25, 76.80, 81.22, 83.75, 76.16, 72.69]) / 300

    electricity_price = np.array(
        [6.01, 73.91, 71.31, 69.24, 68.94, 70.56, 75.16, 73.19, 79.70, 85.76, 86.90, 88.60, 90.62, 91.26, 93.70, 90.94,
         91.26, 80.39, 76.25, 76.80, 81.22, 83.75, 76.16, 72.69]) / 300

    # 2) Thermal system configuration
    QHVAC_max = 5000
    eff_HVAC = 4
    c_air = 185
    r_t = 0.013
    ambinent_temprature = np.array(
        [27, 27, 26, 26, 26, 26, 26, 25, 27, 28, 30, 31, 32, 32, 32, 32, 32, 32, 31, 30, 29, 28, 28, 27])
    temprature_in_min = 20
    temprature_in_max = 20
    # 3) For the vehicles, the driving patterns are assumed to be the same, following the normal distribution, in the day-time
    # 3.1) Electric vehicles
    Arrival_mean = 8.92
    Arrival_std = 3.24
    Dep_mean = 17.47
    Dep_std = 3.41
    SOC_mean = 0.4
    SOC_std = 0.1
    EEV = 50
    PEV_max = 10
    # 3.2) Gas vehicles
    EGV = 50
    PGV_max = 700

    # 4) CCHP system
    # CCHP system
    Gas_price = 0.1892
    Gmax = 20000
    UT = 2
    DT = 1
    ICHP0 = 1
    SU = 10
    SD = 10
    ICOST = 100

    two_stage_stochastic = TwoStageStochastic()
    two_stage_stochastic.day_ahead_scheduling(SCENARIO_UPDATE=0, AC_LOAD_MEAN=AC_PD, DC_LOAD_MEAN=DC_PD,
                                              DC_LOAD_STD=DC_PD * 0.03, AC_LOAD_STD=0.03 * AC_PD, PV_MEAN=PV_PG,
                                              PV_STD=PV_PG * 0.05, HD_MEAN=HD, HD_STD=HD * 0.03, CD_MEAN=CD,
                                              CD_STD=CD * 0.03, AT_MEAN=ambinent_temprature,
                                              AT_STD=ambinent_temprature * 0.05)
