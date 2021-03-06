"""
Units data for wind hydro dispatch
Data is obtained from real test systems
"""
from numpy import array


def case14():
    """Power flow data for real wind hydro power systems

    @return: Power flow data for jointed wind hydro power systems
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95],
        [2, 2, 0, 0, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95],
        [3, 2, 0, 0, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95],
        [4, 2, 0, 0, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95],
        [5, 1, 0, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
        [6, 1, 15, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
        [7, 1, 20, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
        [8, 1, 30, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
        [9, 1, 28, 0, 0, 19, 1, 1, 0, 110, 1, 1.05, 0.95],
        [10, 1, 10, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
        [11, 1, 15, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
        [12, 1, 15, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
        [13, 1, 13, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95],
        [14, 1, 0, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [1, 0, 0, 900, -900, 1, 100, 1, 900, -900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 100, 0, 47.6, -47.6, 1, 100, 1, 238, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 100, 0, 49.5, -49.5, 1, 100, 1, 247.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 70, 0, 194, -30, 1, 100, 1, 194, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 100, 0, 29.8, -29.8, 1, 100, 1, 149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 20, 0, 60, -2, 1, 100, 1, 60, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [12, 20, 0, 9.5, -9.5, 1, 100, 1, 47.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [12, 20, 0, 70, -9, 1, 100, 1, 70, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [13, 20, 0, 70, -4, 1, 100, 1, 70, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [14, 50, 0, 20, -20, 1, 100, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [1, 2,   0.0038, 0.0194, 0.0368,   300, 300, 300, 0,    0, 1, -360, 360],
        [1, 3,   0.0698, 0.3819, 0.0726,   300, 300, 300, 0,    0, 1, -360, 360],
        [1, 4,   0.0500, 0.1900, 0.0200,   300, 300, 300, 0,    0, 1, -360, 360],
        [1, 5,   0.0000, 0.0923, 0.0000,   300, 300, 300, 0.69, 0, 1, -360, 360],
        [1, 5,   0.0000, 0.0923, 0.0000,   300, 300, 300, 0.69, 0, 1, -360, 360],
        [5, 6,   0.0036, 0.0116, 0.00015,  130, 130, 130, 0,    0, 1, -360, 360],
        [6, 7,   0.0045, 0.0115, 0.0014,   130, 130, 130, 0,    0, 0, -360, 360],
        [7, 8,   0.0054, 0.0137, 0.0017,   100, 100, 100, 0,    0, 1, -360, 360],
        [5, 8,   0.0010, 0.0027, 0.000334, 300, 300, 300, 0,    0, 1, -360, 360],
        [5, 9,   0.0049, 0.0125, 0.0016,   130, 130, 130, 0,    0, 1, -360, 360],
        [9, 10,  0.0028, 0.0070, 0.000866, 300, 300, 300, 0,    0, 1, -360, 360],
        [5, 11,  0.0045, 0.0115, 0.0014,   200, 200, 200, 0,    0, 1, -360, 360],
        [11, 13, 0.0024, 0.0061, 0.000656, 100, 100, 100, 0,    0, 1, -360, 360],
        [5, 12,  0.0017, 0.0043, 0.000534, 100, 100, 100, 0,    0, 1, -360, 360],
        [5, 14,  0.0029, 0.0029, 0.0016,   100, 100, 100, 0,    0, 1, -360, 360]
    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 0, 0, 3, 0.0430293, 20, 0],
        [2, 0, 0, 3, 0.25, 20, 0],
        [2, 0, 0, 3, 0.01, 40, 0],
        [2, 0, 0, 3, 0.01, 40, 0],
        [2, 0, 0, 3, 0.01, 40, 0],
        [2, 0, 0, 3, 0.0430293, 20, 0],
        [2, 0, 0, 3, 0.25, 20, 0],
        [2, 0, 0, 3, 0.01, 40, 0],
        [2, 0, 0, 3, 0.01, 40, 0],
        [2, 0, 0, 3, 0.01, 40, 0]
    ])

    return ppc
