# Copyright (c) 2017-2018. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
Power flow data for 123 bus test system under three phase balance scenario.
"""

from numpy import array
import pandas as pd

def case123():
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 1000.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = pd.read_csv('case123bus.csv', header=None).values
    ppc["branch"] = pd.read_csv('case123branch.csv', header=None).values
    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf, start-up time, shut-down time and initial condition!
    ppc["gen"] = array([
        [1, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [10, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [20, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [30, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [40, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [50, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [60, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [70, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [80, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [90, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [100, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [110, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [120, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [15, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [25, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [35, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [45, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [55, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [65, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [75, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [85, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [95, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [105, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
        [115, 23.54, 0, 15000, -20000, 1, 100, 1, 80000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1],
    ])

    ## branch data

    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = array([
        [1, 8],
        [2, 23],
        [3, 26],
    ])

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
        [2, 0, 0, 3, 1 / 1000 ** 2, 2 / 1000, 0],
    ])

    return ppc
