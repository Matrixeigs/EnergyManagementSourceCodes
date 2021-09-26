"""
Interval unit commitment
@author:Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
"""
from pypower import loadcase, ext2int, makeBdc
from scipy.sparse import csr_matrix as sparse
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, concatenate, array, diag, eye
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp
import pandas as pd


def problem_formulation(case, BETA=0.15, BETA_HYDRO=0.05, BETA_LOAD=0.03):
    """
    :param case: The test case for unit commitment problem
    :return:
    """
    CAP_WIND = 1  # The capacity of wind farm
    # The disturbance range of wind farm
    # The disturbance range of wind farm

    CAPVALUE = 10  # The capacity value
    Price_energy = r_[ones(8), 3 * ones(8), ones(8)]

    from pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
    from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
    from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I
    from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN

    mpc = ext2int.ext2int(case)
    baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

    nb = shape(mpc['bus'])[0]  ## number of buses
    nl = shape(mpc['branch'])[0]  ## number of branches
    ng = shape(mpc['gen'])[0]  ## number of dispatchable injections

    # Bbus = makeBdc.makeBdc(baseMVA, bus, branch)
    # Distribution_factor = Bbus[1] * inv(Bbus[0])

    Distribution_factor = array([
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.005, -0.005, -0.005, -1.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,
         -0.005, ],
        [0.47, 0.47, 0.47, 0.47, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
        [0.47, 0.47, 0.47, 0.47, -0.03, - 0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
        [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, -0.68, -0.68, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32],
        [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, -0.68, -0.68, 0.32, 0.32, 0.32, 0.32],
        [0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, -0.84, 0.16, 0.16, 0.16, 0.16],
        [-0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -1.16, -0.16, -1.16, -0.16],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        [-0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -1.16, -0.16, -0.16],
        [-0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -1.08],
    ])
    Distribution_factor = sparse(Distribution_factor)
    # Formulate connection matrix for wind farms
    i = []
    PWMAX = []
    PWMIN = []
    for index in range(ng):
        if gen[index, PMIN] == 0:
            i.append(index)
            PWMAX.append(gen[index, PMAX])
            PWMIN.append(gen[index, PMIN])
    i = array(i)
    nw = i.shape[0]
    Cw = sparse((ones(nw), (gen[i, GEN_BUS], arange(nw))), shape=(nb, nw))
    PWMAX = array(PWMAX).reshape((len(PWMAX), 1))
    PWMIN = array(PWMIN).reshape((len(PWMIN), 1))
    # Formulate the connection matrix for hydro power plants
    i = []
    PHMAX = []
    PHMIN = []
    for index in range(ng):
        if gen[index, PMIN] > 0:
            i.append(index)
            PHMAX.append(gen[index, PMAX])
            PHMIN.append(gen[index, PMIN])
    i = array(i)
    nh = i.shape[0]
    Ch = sparse((ones(nh), (gen[i, GEN_BUS], arange(nh))), shape=(nb, nh))
    PHMAX = array(PHMAX).reshape((len(PHMAX), 1))
    PHMIN = array(PHMIN).reshape((len(PHMIN), 1))

    # Formulate the external power systems
    i = []
    PEXMAX = []
    PEXMIN = []
    for index in range(ng):
        if gen[index, PMIN] < 0:
            i.append(index)
            PEXMAX.append(gen[index, PMAX])
            PEXMIN.append(gen[index, PMIN])
    i = array(i)
    nex = i.shape[0]
    Cex = sparse((ones(nex), (gen[i, GEN_BUS], arange(nex))), shape=(nb, nex))
    PEXMAX = array(PEXMAX).reshape((len(PEXMAX), 1))
    PEXMIN = array(PEXMIN).reshape((len(PEXMIN), 1))
    PLMAX = branch[:, RATE_A].reshape((nl, 1))  # The power flow limitation

    T = 24
    ## Profiles
    # Wind profile
    WIND_PROFILE = array(
        [591.35, 714.50, 1074.49, 505.06, 692.78, 881.88, 858.48, 609.11, 559.95, 426.86, 394.54, 164.47, 27.15, 4.47,
         54.08, 109.90, 111.50, 130.44, 111.59, 162.38, 188.16, 216.98, 102.94, 229.53]).reshape((T, 1))
    WIND_PROFILE = WIND_PROFILE / WIND_PROFILE.max()
    WIND_PROFILE_FORECAST = zeros((T * nw, 1))
    Delta_wind = zeros((T * nw, 1))
    for i in range(T):
        WIND_PROFILE_FORECAST[i * nw:(i + 1) * nw, :] = WIND_PROFILE[i] * PWMAX
        Delta_wind[i * nw:(i + 1) * nw, :] = WIND_PROFILE[i] * PWMAX * BETA

    # Load profile
    LOAD_PROFILE = array([0.632596195634005, 0.598783973523217, 0.580981513054525, 0.574328051348912, 0.584214221241601,
                          0.631074282084712, 0.708620833751212, 0.797665730618795, 0.877125330124026, 0.926981579915087,
                          0.947428654208872, 0.921588439808779, 0.884707317888543, 0.877717046100358, 0.880387289807107,
                          0.892056129442049, 0.909233443653261, 0.926748403704075, 0.968646575067696, 0.999358974358974,
                          0.979169591816267, 0.913517534182463, 0.806453715775750, 0.699930632166617]).reshape((T, 1))
    LOAD_FORECAST = zeros((T * nb, 1))
    Delta_load = zeros((T * nb, 1))
    load_base = bus[:, PD].reshape(nb, 1)
    for i in range(T):
        LOAD_FORECAST[i * nb:(i + 1) * nb, :] = load_base * LOAD_PROFILE[i]
        Delta_load[i * nb:(i + 1) * nb, :] = load_base * BETA_LOAD

    # Hydro information
    HYDRO_INJECT = array([6, 2, 4, 3]).reshape((nh, 1))
    HYDRO_INJECT_FORECAST = zeros((T * nh, 1))
    Delta_hydro = zeros((T * nh, 1))
    for i in range(T):
        HYDRO_INJECT_FORECAST[i * nh:(i + 1) * nh, :] = HYDRO_INJECT
        Delta_hydro[i * nh:(i + 1) * nh, :] = HYDRO_INJECT * BETA_HYDRO

    MIN_DOWN = ones((nh, 1))
    MIN_UP = ones((nh, 1))

    QMIN = array([1.5, 1, 1, 1]).reshape((nh, 1))
    QMAX = array([20, 10, 10, 10]).reshape((nh, 1))
    VMIN = array([70, 50, 70, 40]).reshape((nh, 1))
    VMAX = array([160, 140, 150, 130]).reshape((nh, 1))
    V0 = array([110, 90, 100, 80]).reshape((nh, 1))
    M_transfer = diag(array([8.8649, 6.4444, 6.778, 7.3333]))
    C_TEMP = array([30, 2, 9, 4]).reshape((4, 1))
    Q_TEMP = array([1.5, 1, 1, 1]).reshape((4, 1))
    # Define the first stage decision variables
    ON = 0
    OFF = 1
    IHG = 2
    PHG = 3
    RUHG = 4
    RDHG = 5
    QHG = 6
    QUHG = 7
    QDHG = 8
    V = 9
    S = 10
    PWC = 11
    PLC = 12
    PEX = 13
    CEX = 14
    NX = PWC * nh * T + nw * T + nb * T + nex * T + 1
    lb = zeros((NX, 1))
    ub = zeros((NX, 1))
    c = zeros((NX, 1))
    vtypes = ["c"] * NX
    for i in range(T):
        for j in range(nh):
            # lower boundary information
            lb[ON * nh * T + i * nh + j] = 0
            lb[OFF * nh * T + i * nh + j] = 0
            lb[IHG * nh * T + i * nh + j] = 0
            lb[PHG * nh * T + i * nh + j] = 0
            lb[RUHG * nh * T + i * nh + j] = 0
            lb[RDHG * nh * T + i * nh + j] = 0
            lb[QHG * nh * T + i * nh + j] = 0
            lb[QUHG * nh * T + i * nh + j] = 0
            lb[QDHG * nh * T + i * nh + j] = 0
            lb[V * nh * T + i * nh + j] = VMIN[j]
            lb[S * nh * T + i * nh + j] = 0
            # upper boundary information
            ub[ON * nh * T + i * nh + j] = 1
            ub[OFF * nh * T + i * nh + j] = 1
            ub[IHG * nh * T + i * nh + j] = 1
            ub[PHG * nh * T + i * nh + j] = PHMAX[j]
            ub[RUHG * nh * T + i * nh + j] = PHMAX[j]
            ub[RDHG * nh * T + i * nh + j] = PHMAX[j]
            ub[QHG * nh * T + i * nh + j] = QMAX[j]
            ub[QUHG * nh * T + i * nh + j] = QMAX[j]
            ub[QDHG * nh * T + i * nh + j] = QMAX[j]
            ub[V * nh * T + i * nh + j] = VMAX[j]
            ub[S * nh * T + i * nh + j] = 10 ** 8
            # objective value
            c[S * nh * T + i * nh + j] = 1
            c[RUHG * nh * T + i * nh + j] = -Price_energy[j]
            c[RDHG * nh * T + i * nh + j] = Price_energy[j]

            # variables types
            vtypes[ON * nh * T + i * nh + j] = "D"
            vtypes[OFF * nh * T + i * nh + j] = "D"
            vtypes[IHG * nh * T + i * nh + j] = "D"
            if i == T - 1:
                lb[V * nh * T + i * nh + j] = V0[j]
                ub[V * nh * T + i * nh + j] = V0[j]

        for j in range(nw):
            # lower boundary information
            lb[PWC * nh * T + i * nw + j] = 0
            # upper boundary information
            ub[PWC * nh * T + i * nw + j] = WIND_PROFILE_FORECAST[i * nw + j]
            # objective value
            c[PWC * nh * T + i * nw + j] = 1
        for j in range(nb):
            # lower boundary information
            lb[PWC * nh * T + nw * T + i * nb + j] = 0
            # upper boundary information
            ub[PWC * nh * T + nw * T + i * nb + j] = bus[j, PD] * LOAD_PROFILE[i]
            # objective value
            c[PWC * nh * T + nw * T + i * nb + j] = 10 ** 8
        for j in range(nex):
            # lower boundary information
            lb[PWC * nh * T + nw * T + nb * T + i * nex + j] = PEXMIN[j]
            # upper boundary information
            ub[PWC * nh * T + nw * T + nb * T + i * nex + j] = PEXMAX[j]
            # objective value
            c[PWC * nh * T + nw * T + nb * T + i * nex + j] = -Price_energy[i]
    # lower boundary information
    lb[PWC * nh * T + nw * T + nb * T + nex * T] = PEXMIN[0]
    # upper boundary information
    ub[PWC * nh * T + nw * T + nb * T + nex * T] = PEXMAX[0]
    # objective value
    # c[PWC * nh * T + nw * T + nb * T + nex * T] = -CAPVALUE

    # 2) Constraint set
    # 2.1) Power balance equation
    Aeq = zeros((T, NX))
    beq = zeros((T, 1))
    for i in range(T):
        # For the hydro units
        for j in range(nh):
            Aeq[i, PHG * nh * T + i * nh + j] = 1
        # For the wind farms
        for j in range(nw):
            Aeq[i, PWC * nh * T + i * nw + j] = -1
        # For the loads
        for j in range(nb):
            Aeq[i, PWC * nh * T + nw * T + i * nb + j] = 1
        # For the power exchange
        for j in range(nex):
            Aeq[i, PWC * nh * T + nw * T + nb * T + i * nex + j] = -1

        beq[i] = sum(load_base) * LOAD_PROFILE[i] - sum(WIND_PROFILE_FORECAST[i * nw:(i + 1) * nw])

    # 2.2) Status transformation of each unit
    Aeq_temp = zeros((T * nh, NX))
    beq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aeq_temp[i * nh + j, ON * nh * T + i * nh + j] = -1
            Aeq_temp[i * nh + j, OFF * nh * T + i * nh + j] = 1
            Aeq_temp[i * nh + j, IHG * nh * T + i * nh + j] = 1
            if i != 0:
                Aeq_temp[i * nh + j, IHG * nh * T + (i - 1) * nh + j] = -1
            else:
                beq_temp[i * T + j] = 0

    Aeq = concatenate((Aeq, Aeq_temp), axis=0)
    beq = concatenate((beq, beq_temp), axis=0)
    # 2.3) water status change
    Aeq_temp = zeros((T * nh, NX))
    beq_temp = HYDRO_INJECT_FORECAST
    for i in range(T):
        for j in range(nh):
            Aeq_temp[i * nh + j, V * nh * T + i * nh + j] = 1
            Aeq_temp[i * nh + j, S * nh * T + i * nh + j] = 1
            Aeq_temp[i * nh + j, QHG * nh * T + i * nh + j] = 1
            if i != 0:
                Aeq_temp[i * nh + j, V * nh * T + (i - 1) * nh + j] = -1
            else:
                beq_temp[i * T + j] += V0[j]

    Aeq = concatenate((Aeq, Aeq_temp), axis=0)
    beq = concatenate((beq, beq_temp), axis=0)

    # 2.4) Power water transfering
    Aeq_temp = zeros((T * nh, NX))
    beq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aeq_temp[i * nh + j, PHG * nh * T + i * nh + j] = 1
            Aeq_temp[i * nh + j, QHG * nh * T + i * nh + j] = -M_transfer[j, j]
            Aeq_temp[i * nh + j, IHG * nh * T + i * nh + j] = -C_TEMP[j] + M_transfer[j, j] * Q_TEMP[j]
    Aeq = concatenate((Aeq, Aeq_temp), axis=0)
    beq = concatenate((beq, beq_temp), axis=0)

    # 2.5) Power range limitation
    Aineq = zeros((T * nh, NX))
    bineq = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq[i * nh + j, ON * nh * T + i * nh + j] = 1
            Aineq[i * nh + j, OFF * nh * T + i * nh + j] = 1
            bineq[i * nh + j] = 1

    Aineq_temp = zeros((T * nh, NX))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, IHG * nh * T + i * nh + j] = PHMIN[j]
            Aineq_temp[i * nh + j, PHG * nh * T + i * nh + j] = -1
            Aineq_temp[i * nh + j, RDHG * nh * T + i * nh + j] = 1
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    Aineq_temp = zeros((T * nh, NX))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, IHG * nh * T + i * nh + j] = -PHMAX[j]
            Aineq_temp[i * nh + j, PHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, RUHG * nh * T + i * nh + j] = 1
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    # 2.6) Water reserve constraints
    Aineq_temp = zeros((T * nh, NX))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, PHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, RUHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, IHG * nh * T + i * nh + j] = -C_TEMP[j] + M_transfer[j, j] * Q_TEMP[j]
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = -M_transfer[j, j]
            Aineq_temp[i * nh + j, QUHG * nh * T + i * nh + j] = -M_transfer[j, j]

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    Aineq_temp = zeros((T * nh, NX))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, PHG * nh * T + i * nh + j] = -1
            Aineq_temp[i * nh + j, RDHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, IHG * nh * T + i * nh + j] = C_TEMP[j] - M_transfer[j, j] * Q_TEMP[j]
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = M_transfer[j, j]
            Aineq_temp[i * nh + j, QDHG * nh * T + i * nh + j] = -M_transfer[j, j]

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    # 2.7) water flow constraints
    Aineq_temp = zeros((T * nh, NX))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, IHG * nh * T + i * nh + j] = QMIN[j]
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = -1
            Aineq_temp[i * nh + j, QDHG * nh * T + i * nh + j] = 1
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    Aineq_temp = zeros((T * nh, NX))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, IHG * nh * T + i * nh + j] = -QMAX[j]
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, QUHG * nh * T + i * nh + j] = 1

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    # 2.8) Water reserve limitation
    Aineq_temp = zeros((T * nh, NX))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, V * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = -1
            Aineq_temp[i * nh + j, QDHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, S * nh * T + i * nh + j] = -1
            bineq_temp[i * nh + j] = VMAX[j] - HYDRO_INJECT_FORECAST[i * nh + j]

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    Aineq_temp = zeros((T * nh, NX))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, V * nh * T + i * nh + j] = -1
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, QUHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, S * nh * T + i * nh + j] = 1
            bineq_temp[i * nh + j] = -VMIN[j] + HYDRO_INJECT_FORECAST[i * nh + j]

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    # 2.9) Line flow limitation
    Aineq_temp = zeros((T * nl, NX))
    bineq_temp = zeros((T * nl, 1))
    for i in range(T):
        Aineq_temp[i * nl:(i + 1) * nl, PHG * nh * T + i * nh:PHG * nh * T + (i + 1) * nh] = -(
                Distribution_factor * Ch).todense()
        Aineq_temp[i * nl:(i + 1) * nl, PWC * nh * T + i * nw:PWC * nh * T + (i + 1) * nw] = (
                Distribution_factor * Cw).todense()
        Aineq_temp[i * nl:(i + 1) * nl,
        PWC * nh * T + nw * T + i * nb:PWC * nh * T + nw * T + (i + 1) * nb] = -Distribution_factor.todense()

        Aineq_temp[i * nl:(i + 1) * nl,
        PWC * nh * T + nw * T + nb * T + i * nex:PWC * nh * T + nw * T + nb * T + (i + 1) * nex] = (
                Distribution_factor * Cex).todense()

        bineq_temp[i * nl:(i + 1) * nl, :] = PLMAX - Distribution_factor * (
                (bus[:, PD] * LOAD_PROFILE[i]).reshape(nb, 1) - Cw * WIND_PROFILE_FORECAST[i * nw:(i + 1) * nw])
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    Aineq_temp = zeros((T * nl, NX))
    bineq_temp = zeros((T * nl, 1))
    for i in range(T):
        Aineq_temp[i * nl:(i + 1) * nl, PHG * nh * T + i * nh:PHG * nh * T + (i + 1) * nh] = (
                Distribution_factor * Ch).todense()

        Aineq_temp[i * nl:(i + 1) * nl, PWC * nh * T + i * nw:PWC * nh * T + (i + 1) * nw] = -(
                Distribution_factor * Cw).todense()

        Aineq_temp[i * nl:(i + 1) * nl,
        PWC * nh * T + nw * T + i * nb:PWC * nh * T + nw * T + (i + 1) * nb] = Distribution_factor.todense()

        Aineq_temp[i * nl:(i + 1) * nl,
        PWC * nh * T + nw * T + nb * T + i * nex:PWC * nh * T + nw * T + nb * T + (i + 1) * nex] = -(
                Distribution_factor * Cex).todense()

        bineq_temp[i * nl:(i + 1) * nl, :] = PLMAX + Distribution_factor * (
                (bus[:, PD] * LOAD_PROFILE[i]).reshape(nb, 1) - Cw * WIND_PROFILE_FORECAST[i * nw:(i + 1) * nw])

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)
    # 2.10)  Capacity limitation
    Aineq_temp = zeros((T, NX))
    bineq_temp = zeros((T, 1))
    for i in range(T):
        Aineq_temp[i, PWC * nh * T + nw * T + nb * T + nex * T] = 1
        Aineq_temp[i, PWC * nh * T + nw * T + nb * T + i * nex] = -1

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)
    # 2.11)  Up and down reserve for the forecasting errors
    # Up reserve limitation
    Aineq_temp = zeros((T, NX))
    bineq_temp = zeros((T, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i, RUHG * nh * T + i * nh + j] = -1
        for j in range(nw):
            bineq_temp[i] -= Delta_wind[i * nw + j]
        for j in range(nb):
            bineq_temp[i] -= Delta_load[i * nb + j]

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)
    # Down reserve limitation
    Aineq_temp = zeros((T, NX))
    bineq_temp = zeros((T, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i, RDHG * nh * T + i * nh + j] = -1
        for j in range(nw):
            bineq_temp[i] -= Delta_wind[i * nw + j]
        for j in range(nb):
            bineq_temp[i] -= Delta_load[i * nb + j]
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    model_first_stage = {"c": c,
                         "lb": lb,
                         "ub": ub,
                         "A": Aineq,
                         "b": bineq,
                         "Aeq": Aeq,
                         "beq": beq,
                         "vtypes": vtypes}

    ## Formualte the second stage decision making problem
    phg = 0
    qhg = 1
    v = 2
    s = 3
    pwc = 4
    plc = 5
    pex = 6
    cex = 7
    nx = pwc * nh * T + nw * T + nb * T + nex * T + 1
    # Generate the lower and boundary for the first stage decision variables
    lb = zeros((nx, 1))
    ub = zeros((nx, 1))
    c = zeros((nx, 1))
    vtypes = ["c"] * nx
    nu = nh * T + nw * T + nb * T
    u_mean = concatenate([HYDRO_INJECT_FORECAST, WIND_PROFILE_FORECAST, LOAD_FORECAST])
    u_delta = concatenate([Delta_hydro, Delta_wind, Delta_load])
    for i in range(T):
        for j in range(nh):
            # lower boundary information
            lb[phg * nh * T + i * nh + j] = 0
            lb[qhg * nh * T + i * nh + j] = 0
            lb[v * nh * T + i * nh + j] = VMIN[j]
            lb[s * nh * T + i * nh + j] = 0
            # upper boundary information
            ub[phg * nh * T + i * nh + j] = PHMAX[j]
            ub[qhg * nh * T + i * nh + j] = QMAX[j]
            ub[v * nh * T + i * nh + j] = VMAX[j]
            ub[s * nh * T + i * nh + j] = 10 ** 8
            # objective value
            c[s * nh * T + i * nh + j] = 1
            if i == T - 1:
                lb[v * nh * T + i * nh + j] = V0[j]
                ub[v * nh * T + i * nh + j] = V0[j]
        for j in range(nw):
            # lower boundary information
            lb[pwc * nh * T + i * nw + j] = 0
            # upper boundary information
            ub[pwc * nh * T + i * nw + j] = 10 ** 4
            # objective value
            c[pwc * nh * T + i * nw + j] = 1
        for j in range(nb):
            # lower boundary information
            lb[pwc * nh * T + nw * T + i * nb + j] = 0
            # upper boundary information
            ub[pwc * nh * T + nw * T + i * nb + j] = 10 ** 4
            # objective value
            c[pwc * nh * T + nw * T + i * nb + j] = 10 ** 6
        for j in range(nex):
            # lower boundary information
            lb[pwc * nh * T + nw * T + nb * T + i * nex + j] = PEXMIN[j]
            # upper boundary information
            ub[pwc * nh * T + nw * T + nb * T + i * nex + j] = PEXMAX[j]
            # objective value
            # c[pwc * nh * T + nw * T + nb * T + i * nex + j] = -Price_energy[i]
    # lower boundary information
    lb[pwc * nh * T + nw * T + nb * T + nex * T] = PEXMIN[0]
    # upper boundary information
    ub[pwc * nh * T + nw * T + nb * T + nex * T] = PEXMAX[0]
    # objective value
    c[pwc * nh * T + nw * T + nb * T + nex * T] = -CAPVALUE
    # Generate correlate constraints
    # 3.1) Power balance constraints
    E = zeros((T, NX))
    M = zeros((T, nu))
    G = zeros((T, nx))
    h = beq[0:T]
    for i in range(T):
        # For the hydro units
        for j in range(nh):
            G[i, phg * nh * T + i * nh + j] = 1
        # For the wind farms
        for j in range(nw):
            G[i, pwc * nh * T + i * nw + j] = -1
        # For the loads
        for j in range(nb):
            G[i, pwc * nh * T + nw * T + i * nb + j] = 1
        # For the power exchange
        for j in range(nex):
            G[i, pwc * nh * T + nw * T + nb * T + i * nex + j] = -1

    # Update G,M,E,h
    G = concatenate([G, -G])
    M = concatenate([M, -M])
    E = concatenate([E, -E])
    h = concatenate([h, -h])
    # 3.2) water status change
    E_temp = zeros((T * nh, NX))
    M_temp = zeros((T * nh, nu))
    G_temp = zeros((T * nh, nx))
    h_temp = HYDRO_INJECT_FORECAST
    for i in range(T):
        for j in range(nh):
            G_temp[i * nh + j, v * nh * T + i * nh + j] = 1
            G_temp[i * nh + j, s * nh * T + i * nh + j] = 1
            G_temp[i * nh + j, qhg * nh * T + i * nh + j] = 1
            if i != 0:
                G_temp[i * nh + j, v * nh * T + (i - 1) * nh + j] = -1
            else:
                h_temp[i * T + j] = V0[j]

            # M_temp[i * nh + j, i * nh + j] = -1
    G = concatenate([G, G_temp, -G_temp])
    M = concatenate([M, M_temp, -M_temp])
    E = concatenate([E, E_temp, -E_temp])
    h = concatenate([h, h_temp, -h_temp])

    # 3.3) Power water transfering
    E_temp = zeros((T * nh, NX))
    M_temp = zeros((T * nh, nu))
    G_temp = zeros((T * nh, nx))
    h_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            G_temp[i * nh + j, phg * nh * T + i * nh + j] = 1
            G_temp[i * nh + j, qhg * nh * T + i * nh + j] = -M_transfer[j, j]
            E_temp[i * nh + j, IHG * nh * T + i * nh + j] = -C_TEMP[j] + M_transfer[j, j] * Q_TEMP[j]
    G = concatenate([G, G_temp, -G_temp])
    M = concatenate([M, M_temp, -M_temp])
    E = concatenate([E, E_temp, -E_temp])
    h = concatenate([h, h_temp, -h_temp])

    # 3.4) Power range limitation
    # Some problem found
    E_temp = zeros((T * nh, NX))
    M_temp = zeros((T * nh, nu))
    G_temp = zeros((T * nh, nx))
    h_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            G_temp[i * nh + j, phg * nh * T + i * nh + j] = 1
            E_temp[i * nh + j, PHG * nh * T + i * nh + j] = -1
            E_temp[i * nh + j, RDHG * nh * T + i * nh + j] = 1
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    E_temp = zeros((T * nh, NX))
    M_temp = zeros((T * nh, nu))
    G_temp = zeros((T * nh, nx))
    h_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            G_temp[i * nh + j, phg * nh * T + i * nh + j] = -1
            E_temp[i * nh + j, PHG * nh * T + i * nh + j] = 1
            E_temp[i * nh + j, RUHG * nh * T + i * nh + j] = 1
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    # 3.5) Water flow constraints
    E_temp = zeros((T * nh, NX))
    M_temp = zeros((T * nh, nu))
    G_temp = zeros((T * nh, nx))
    h_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            G_temp[i * nh + j, qhg * nh * T + i * nh + j] = 1
            E_temp[i * nh + j, QHG * nh * T + i * nh + j] = -1
            E_temp[i * nh + j, QDHG * nh * T + i * nh + j] = 1
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    E_temp = zeros((T * nh, NX))
    M_temp = zeros((T * nh, nu))
    G_temp = zeros((T * nh, nx))
    h_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            G_temp[i * nh + j, qhg * nh * T + i * nh + j] = -1
            E_temp[i * nh + j, QHG * nh * T + i * nh + j] = 1
            E_temp[i * nh + j, QUHG * nh * T + i * nh + j] = 1

    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])
    # 3.6) Line flow constraints
    E_temp = zeros((T * nl, NX))
    M_temp = zeros((T * nl, nu))
    G_temp = zeros((T * nl, nx))
    h_temp = zeros((T * nl, 1))

    for i in range(T):
        G_temp[i * nl:(i + 1) * nl, phg * nh * T + i * nh:phg * nh * T + (i + 1) * nh] = (
                Distribution_factor * Ch).todense()

        G_temp[i * nl:(i + 1) * nl, pwc * nh * T + i * nw: pwc * nh * T + (i + 1) * nw] = -(
                Distribution_factor * Cw).todense()

        G_temp[i * nl:(i + 1) * nl,
        pwc * nh * T + nw * T + i * nb:pwc * nh * T + nw * T + (i + 1) * nb] = Distribution_factor.todense()

        G_temp[i * nl:(i + 1) * nl,
        pwc * nh * T + nw * T + nb * T + i * nex:pwc * nh * T + nw * T + nb * T + (i + 1) * nex] = -(
                Distribution_factor * Cex).todense()

        M_temp[i * nl:(i + 1) * nl, nh * T + i * nw: nh * T + (i + 1) * nw] = (
                Distribution_factor * Cw).todense()

        M_temp[i * nl:(i + 1) * nl,
        nh * T + nw * T + i * nb: nh * T + nw * T + (i + 1) * nb] = -Distribution_factor.todense()

        h_temp[i * nl:(i + 1) * nl, :] = -PLMAX

    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    E_temp = zeros((T * nl, NX))
    M_temp = zeros((T * nl, nu))
    G_temp = zeros((T * nl, nx))
    h_temp = zeros((T * nl, 1))

    for i in range(T):
        G_temp[i * nl:(i + 1) * nl, phg * nh * T + i * nh:phg * nh * T + (i + 1) * nh] = -(
                Distribution_factor * Ch).todense()

        G_temp[i * nl:(i + 1) * nl, pwc * nh * T + i * nw: pwc * nh * T + (i + 1) * nw] = (
                Distribution_factor * Cw).todense()

        G_temp[i * nl:(i + 1) * nl,
        pwc * nh * T + nw * T + i * nb:pwc * nh * T + nw * T + (i + 1) * nb] = -Distribution_factor.todense()

        G_temp[i * nl:(i + 1) * nl,
        pwc * nh * T + nw * T + nb * T + i * nex:pwc * nh * T + nw * T + nb * T + (i + 1) * nex] = (
                Distribution_factor * Cex).todense()

        M_temp[i * nl:(i + 1) * nl, nh * T + i * nw: nh * T + (i + 1) * nw] = -(
                Distribution_factor * Cw).todense()

        M_temp[i * nl:(i + 1) * nl,
        nh * T + nw * T + i * nb: nh * T + nw * T + (i + 1) * nb] = Distribution_factor.todense()

        h_temp[i * nl:(i + 1) * nl, :] = -PLMAX

    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])
    # 3.7) Capacity constraints
    E_temp = zeros((T, NX))
    M_temp = zeros((T, nu))
    G_temp = zeros((T, nx))
    h_temp = zeros((T, 1))

    for i in range(T):
        G_temp[i, pwc * nh * T + nw * T + nb * T + nex * T] = -1
        G_temp[i, pwc * nh * T + nw * T + nb * T + i * nex] = 1

    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])
    # 3.8) Dispatch range constraints
    # Wind curtailment range
    E_temp = zeros((T * nw, NX))
    M_temp = zeros((T * nw, nu))
    G_temp = zeros((T * nw, nx))
    h_temp = zeros((T * nw, 1))
    for i in range(T):
        for j in range(nw):
            G_temp[i * nw + j, pwc * nh * T + i * nw + j] = -1
            M_temp[i * nw + j, nh * T + i * nw + j] = 1
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    # Load shedding range
    E_temp = zeros((T * nb, NX))
    M_temp = zeros((T * nb, nu))
    G_temp = zeros((T * nb, nx))
    h_temp = zeros((T * nb, 1))
    for i in range(T):
        for j in range(nb):
            G_temp[i * nb + j, pwc * nh * T + T * nw + i * nb + j] = -1
            M_temp[i * nb + j, nh * T + T * nw + i * nb + j] = 1
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])
    # 3.9) Upper boundary and lower boundary information
    E_temp = zeros((nx, NX))
    M_temp = zeros((nx, nu))
    G_temp = eye(nx)
    h_temp = lb
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])

    E_temp = zeros((nx, NX))
    M_temp = zeros((nx, nu))
    G_temp = -eye(nx)
    h_temp = -ub
    G = concatenate([G, G_temp])
    M = concatenate([M, M_temp])
    E = concatenate([E, E_temp])
    h = concatenate([h, h_temp])
    d = c
    # Test the second stage problem
    u = u_mean - u_delta
    M_positive = zeros((M.shape[0], M.shape[1]))
    M_negative = zeros((M.shape[0], M.shape[1]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                M_positive[i, j] = M[i, j]
            else:
                M_negative[i, j] = M[i, j]

    h_second_stage = M_negative.dot(u_mean + u_delta) + M_positive.dot(u_mean - u_delta) - h
    # Reformulate the first stage optimization problem
    c_compact = concatenate([model_first_stage["c"], d])
    Aeq_compact = concatenate([model_first_stage["Aeq"], zeros((model_first_stage["Aeq"].shape[0], nx))], axis=1)
    beq_compact = model_first_stage["beq"]
    A_compact = concatenate([model_first_stage["A"], zeros((model_first_stage["A"].shape[0], nx))], axis=1)
    b_compact = model_first_stage["b"]
    lb_compact = concatenate([model_first_stage["lb"], ones((nx, 1)) * (-10 ** 8)])
    ub_compact = concatenate([model_first_stage["ub"], ones((nx, 1)) * 10 ** 8])
    vtypes_compact = model_first_stage["vtypes"] + ["c"] * nx
    A_compact_temp = zeros((E.shape[0], NX + nx))
    A_compact_temp[:, 0:NX] = -E
    A_compact_temp[:, NX:] = -G
    A_compact = concatenate([A_compact, A_compact_temp])
    b_compact = concatenate([b_compact, h_second_stage])

    # solve the compact model
    (xx, obj, success) = lp(c_compact,
                            Aeq=Aeq_compact,
                            beq=beq_compact,
                            A=A_compact,
                            b=b_compact,
                            xmin=lb_compact,
                            xmax=ub_compact,
                            vtypes=vtypes_compact,
                            objsense="min")
    xx = array(xx).reshape((len(xx), 1))

    # Decompose the first stage decision varialbes
    On = zeros((T, nh))
    Off = zeros((T, nh))
    Ihg = zeros((T, nh))
    Phg = zeros((T, nh))
    Ruhg = zeros((T, nh))
    Rdhg = zeros((T, nh))
    Qhg = zeros((T, nh))
    Quhg = zeros((T, nh))
    Qdhg = zeros((T, nh))
    v = zeros((T, nh))
    s = zeros((T, nh))
    Pwc = zeros((T, nw))
    Plc = zeros((T, nb))
    Pex = zeros((T, nex))
    Cex = zeros((T, 1))
    for i in range(T):
        for j in range(nh):
            On[i, j] = xx[ON * nh * T + i * nh + j]
            Off[i, j] = xx[OFF * nh * T + i * nh + j]
            Ihg[i, j] = xx[IHG * nh * T + i * nh + j]
            Phg[i, j] = xx[PHG * nh * T + i * nh + j]
            Ruhg[i, j] = xx[RUHG * nh * T + i * nh + j]
            Rdhg[i, j] = xx[RDHG * nh * T + i * nh + j]
            Qhg[i, j] = xx[QHG * nh * T + i * nh + j]
            Quhg[i, j] = xx[QUHG * nh * T + i * nh + j]
            Qdhg[i, j] = xx[QDHG * nh * T + i * nh + j]
            v[i, j] = xx[V * nh * T + i * nh + j]
            s[i, j] = xx[S * nh * T + i * nh + j]
    for i in range(T):
        for j in range(nw):
            Pwc[i, j] = xx[PWC * nh * T + i * nw + j]

    for i in range(T):
        for j in range(nb):
            Plc[i, j] = xx[PWC * nh * T + nw * T + i * nb + j]

    for i in range(T):
        for j in range(nex):
            Pex[i, j] = xx[PWC * nh * T + nw * T + nb * T + i * nex + j]

    Cex = xx[-1]
    Ruhg_agg = zeros((T, 1))
    Rdhg_agg = zeros((T, 1))
    for i in range(T):
        for j in range(nh):
            Ruhg_agg[i] += Ruhg[i, j]
            Rdhg_agg[i] += Rdhg[i, j]

    sol = {"START_UP": On,
           "SHUT_DOWN": Off,
           "I": Ihg,
           "PG": Phg,
           "RU": Ruhg,
           "RD": Rdhg,
           "Q": Qhg,
           "QU": Quhg,
           "QD": Qdhg,
           "V": v,
           "S": s,
           "PEX": Pex,
           "PEX_UP": Pex + Ruhg_agg,
           "PEX_DOWN": Pex - Rdhg_agg,
           "obj": obj}

    # df = pd.DataFrame(sol["I"])
    # filepath = './Ih.xlsx'
    # df.to_excel(filepath, index=False)
    #
    # df = pd.DataFrame(sol["PG"])
    # filepath = './Pg.xlsx'
    # df.to_excel(filepath, index=False)
    #
    # df = pd.DataFrame(sol["RU"])
    # filepath = './Ru.xlsx'
    # df.to_excel(filepath, index=False)
    #
    # df = pd.DataFrame(sol["RD"])
    # filepath = './Rd.xlsx'
    # df.to_excel(filepath, index=False)
    #
    # df = pd.DataFrame(sol["Q"])
    # filepath = './q.xlsx'
    # df.to_excel(filepath, index=False)
    #
    # df = pd.DataFrame(sol["PEX"])
    # filepath = './Pex.xlsx'
    # df.to_excel(filepath, index=False)
    #
    # df = pd.DataFrame(sol["PEX_UP"])
    # filepath = './Pex_up.xlsx'
    # df.to_excel(filepath, index=False)
    #
    # df = pd.DataFrame(sol["PEX_DOWN"])
    # filepath = './Pex_down.xlsx'
    # df.to_excel(filepath, index=False)

    return sol


if __name__ == "__main__":
    from unit_commitment.test_cases.case14 import case14

    case = loadcase.loadcase(case14())
    result = zeros((5, 15))
    for i in range(5):
        print(i)
        for j in range(15):
            model = problem_formulation(case, BETA=j * 0.01 + 0.05, BETA_LOAD=i * 0.01 + 0.01)
            result[i, j] = -model["obj"]
            print(j)

    df = pd.DataFrame(result)
    filepath = './sensitive.xlsx'
    df.to_excel(filepath, index=False)

    print(result)
