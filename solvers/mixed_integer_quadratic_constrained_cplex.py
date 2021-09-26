"""
Mixed-integer quadratic constrained programming solvers
"""
import cplex  # import the cplex solver package
from numpy import ones, nonzero, concatenate, zeros
from cplex.exceptions import CplexError


def mixed_integer_quadratic_constrained_programming(c, q, Aeq=None, beq=None, A=None, b=None, xmin=None, xmax=None,
                                                    vtypes=None, Qc=None, rc=None,
                                                    opt=None, objsense=None):
    if type(c) == list:
        nx = len(c)
    else:
        nx = c.shape[0]  # number of decision variables

    if A is not None:
        if A.shape[0] != None:
            nineq = A.shape[0]  # number of equality constraints
        else:
            nineq = 0
    else:
        nineq = 0

    if Aeq is not None:
        if Aeq.shape[0] != None:
            neq = Aeq.shape[0]  # number of inequality constraints
        else:
            neq = 0
    else:
        neq = 0

    if Qc is not None:
        nqc = len(Qc)
    else:
        nqc = 0

    if rc is not None:
        try:
            rc = rc[:, 0]
            rc = rc.tolist()
        except IndexError:
            rc = rc.tolist()
        except:
            pass

    # Fulfilling the missing information
    if beq is None or len(beq) == 0: beq = -cplex.infinity * ones(neq)
    if b is None or len(b) == 0: b = cplex.infinity * ones(nineq)
    if xmin is None or len(xmin) == 0: xmin = -cplex.infinity * ones(nx)
    if xmax is None or len(xmax) == 0: xmax = cplex.infinity * ones(nx)
    # Convert the data format
    try:
        c = c[:, 0]
        c = c.tolist()
    except IndexError:
        c = c.tolist()
    except:
        pass

    try:
        q = q[:, 0]
        q = q.tolist()
    except IndexError:
        q = q.tolist()
    except:
        pass

    try:
        b = b[:, 0]
        b = b.tolist()
    except IndexError:
        b = b.tolist()
    except:
        pass

    try:
        beq = beq[:, 0]
        beq = beq.tolist()
    except IndexError:
        beq = beq.tolist()
    except:
        pass

    try:
        xmin = xmin[:, 0]
        xmin = xmin.tolist()
    except IndexError:
        xmin = xmin.tolist()
    except:
        pass

    try:
        xmax = xmax[:, 0]
        xmax = xmax.tolist()
    except IndexError:
        xmax = xmax.tolist()
    except:
        pass

    if neq == 0: beq = []
    if nineq == 0: b = []

    try:
        prob = cplex.Cplex()
        varnames = ["x" + str(j) for j in range(nx)]
        # 1) Variables Announcement
        if vtypes == None:
            prob.variables.add(obj=c, lb=xmin, ub=xmax, names=varnames)
        else:
            var_types = [prob.variables.type.continuous] * nx

            for i in range(nx):
                if vtypes[i] == "b" or vtypes[i] == "B":
                    var_types[i] = prob.variables.type.binary

                elif vtypes[i] == "d" or vtypes[i] == "D":
                    var_types[i] = prob.variables.type.integer
            prob.variables.add(obj=c, lb=xmin, ub=xmax, types=var_types, names=varnames)

        # 2) Linear constraints
        rhs = beq + b
        sense = ['E'] * neq + ["L"] * nineq

        try:
            rows = zeros(0)
            cols = zeros(0)
            vals = zeros(0)

            if neq != 0:
                (rows, cols) = Aeq.nonzero()
                vals = Aeq[rows, cols].toarray()[0]

            rows_A = zeros(0)
            cols_A = zeros(0)
            vals_A = zeros(0)
            if nineq != 0:
                (rows_A, cols_A) = A.nonzero()
                vals_A = A[rows_A, cols_A].toarray()[0]

            rows = concatenate((rows, neq + rows_A)).astype(int).tolist()
            cols = concatenate((cols, cols_A)).astype(int).tolist()
            vals = concatenate([vals, vals_A])

        except:
            rows = zeros(0)
            cols = zeros(0)
            vals = zeros(0)
            if neq != 0:
                [rows, cols] = nonzero(Aeq)
                vals = Aeq[rows, cols]

            rows_A = zeros(0)
            cols_A = zeros(0)
            vals_A = zeros(0)
            if nineq != 0:
                [rows_A, cols_A] = nonzero(A)
                vals_A = A[rows_A, cols_A]

            rows = concatenate((rows, neq + rows_A)).astype(int).tolist()
            cols = concatenate((cols, cols_A)).astype(int).tolist()
            vals = concatenate((vals, vals_A)).tolist()

        if len(rows) != 0:
            prob.linear_constraints.add(rhs=rhs,
                                        senses=sense)
            prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
        # 3) Quadratic constraints
        if nqc != 0:
            for i in range(nqc):
                prob.quadratic_constraints.add(quad_expr=cplex.SparseTriple(Qc[i][0], Qc[i][1], Qc[i][2]), rhs=rc[i])

        # 4) Objective values
        qmat = [0] * nx
        for i in range(nx):
            qmat[i] = [[i], [q[i]]]
        # prob.objective.set_quadratic(qmat)

        if objsense is not None:
            if objsense == "max":
                prob.objective.set_sense(prob.objective.sense.maximize)
        else:
            prob.objective.set_sense(prob.objective.sense.minimize)

        prob.set_log_stream(None)
        prob.set_error_stream(None)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)
        # prob.timelimit = 100
        prob.parameters.preprocessing.presolve.set(0)
        prob.parameters.timelimit.set(100000)
        prob.parameters.mip.tolerances.mipgap.set(10 ** -2)
        prob.parameters.emphasis.mip.set(2)
        # prob.parameters.dettimelimit = 100

        prob.solve()

        obj = prob.solution.get_objective_value()
        x = prob.solution.get_values()
        success = 1

    except CplexError:
        x = 0
        obj = 0
        success = 0
        print(CplexError)

    except AttributeError:
        print('Encountered an attribute error')
        x = 0
        obj = 0
        success = 0

    # elapse_time = time.time() - t0
    # print(elapse_time)
    return x, obj, success


if __name__ == "__main__":
    # A test problem from Gurobi
    #  maximize
    #        x +   y + 2 z
    #  subject to
    #        x + 2 y + 3 z <= 4
    #        x +   y       >= 1
    #  x, y, z binary

    from numpy import array, zeros

    # c = array([4, 5, 6])
    #
    # A = array([[-1, -1, 0],
    #            [1, -1, 0],
    #            [-7, -12, 0]])
    #
    # b = array([-11, 5, -35])
    #
    # Aeq = array([[-1, -1, 1], ])
    # beq = array([0])
    #
    # lb = zeros((3, 1))
    #
    # vtypes = ["c"] * 3
    # solution = mixed_integer_linear_programming(c, A=A, b=b, Aeq=Aeq, beq=beq, xmin=lb, vtypes=vtypes)

    c = array([1, 1])
    A = array([[1, 1]])
    b = array([11])
    lb = array([6, 6])
    solution = mixed_integer_quadratic_constrained_programming(c, A=A, b=b, xmin=lb)

    print(solution)
