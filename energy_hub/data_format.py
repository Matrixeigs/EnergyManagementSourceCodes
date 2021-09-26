"""The data format for microgrid embedded energy hub
The configuration of energy hub is imported from the following reference:
[1] Energy flow modeling and optimal operation analysis of the micro energy grid based on energy hub

The following groups of variables are considered.
1) Electricity P (nine variables)
1.1) PUG
1.2) PCHP
1.3) PAC2DC
1.4) PDC2AC
1.5) EESS
1.6) PESS_DC
1.7) PESS_CH
1.8) PIAC
1.9) PPV

2) Heating Q (seven variables)
2.1) QCHP
2.2) QGAS
2.3) ETSS
2.4) QES_DC
2.5) QES_CH
2.6) QAC
2.7) QTD

3) Cooling Q (six variables)
3.1) QCE
3.2) QIAC
3.3) ECSS
3.4) QCS_DC
3.5) QCS_CH
3.6) QCD

4) Gas V (two variables)
4.1) VCHP
4.2) VGAS

In total, there are 24 variables in each time slot
"""

## Group 1: Electricity ##
PUG = 0  # CHP input
PCHP = 1  # Utility grid input
PAC2DC = 2  # Electrical energy from AC to DC
PDC2AC = 3  # Electrical energy from AC to DC
EESS = 4
PESS_DC = 5  # ESS discharging rate
PESS_CH = 6  # ESS charging rate
PIAC = 7  # HVAC consumption
PPV = 8  # HVAC consumption
## Group 2: Heating ##
QCHP = 9
QGAS = 10
EHSS = 11
QHS_DC = 12
QHS_CH = 13
QAC = 14
QTD = 15
## Group 3: Cooling ##
QCE = 16
QIAC = 17
ECSS = 18
QCS_DC = 19
QCS_CH = 20
QCD = 21
## Group 4: Gas ##
VCHP = 22
VGAS = 23
## Total number of decision variables
NX = 24  # Number of decision variables