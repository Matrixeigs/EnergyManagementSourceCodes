"""
Hybrid AC/DC micro grids data format

Case format for two-stage stochastic optimization with multiple DGs, RESs and ESSs.
"""
NG = 1
NRES = 1
NESS = 1
PG0 = 0
QG0 = PG0 + NG
PUG = QG0 + NG
QUG = PUG + 1
PBIC_A2D = QUG + 1
PBIC_D2A = PBIC_A2D + 1
QBIC = PBIC_D2A + 1
PESS_CH0 = QBIC + 1
PESS_DC0 = PESS_CH0 + NESS
EESS0 = PESS_DC0 + 1
PPV0 = EESS0 + 1
PMESS = PPV0 + 1
NX_MG = PMESS + 1