"""
Scenario reduction algorithm for two-stage stochastic programmings
The fast forward selection algorithm is used.

References:
    [1]https://edoc.hu-berlin.de/bitstream/handle/18452/3285/8.pdf?sequence=1
    [2]http://ftp.gamsworld.org/presentations/present_IEEE03.pdf
Considering the second stage optimization problem is linear programming, the distance function is refined to
c(ξ, ˜ξ) := max{1, kξkp−1, k˜ξkp−1}kξ − ˜ξk (p = 2, which is sufficient for right hand side uncertainties)

"""

from numpy import array, zeros, argmin, random, arange, linalg, ones, inf, delete, where, append


class ScenarioReduction():
    def __init__(self):
        self.name = "Scenario reduction"

    def run(self, scenario, weight, n_reduced, power):
        """

        :param scenario: A fan scenario tree, when more stage are considered, some merge operation can be implemented
        :param weight: Weight of each scenario
        :param n_reduced: Number of scenarios needs to be reduced
        :param power: The power in the distance calculation
        :return:
        """
        n_scenario = scenario.shape[0]  # number of original scenarios
        c = zeros((n_scenario, n_scenario))
        # Calculate the c matrix
        for i in range(n_scenario):
            for j in range(i+1):
                c[i, j] = linalg.norm((scenario[i, :] - scenario[j, :]), 2)
                c[i, j] = max([1, linalg.norm(scenario[i, :], power - 1), linalg.norm(scenario[j, :], power - 1)]) * \
                          c[i, j]
        c += c.transpose()

        J = arange(n_scenario)  # The original index range
        J_reduced = array([])
        # Implement the iteration
        for n in range(n_reduced):  # find the minimal distance
            c_n = inf * ones(n_scenario)
            c_n[J] = 0
            for u in J:
                # Delete the i-th distance
                J_temp = delete(J, where(J == u))
                for k in J_temp:
                    c_k_j = delete(c[int(k)], J_temp)
                    c_n[int(u)] += weight[int(k)] * min(c_k_j)
            u_i = argmin(c_n)
            J_reduced = append(J_reduced, u_i)
            J = delete(J, where(J == u_i))
        # Optimal redistribution
        p_s = weight.copy()
        p_s[J_reduced.astype(int)] = 0

        for i in J_reduced:
            c_temp = c[int(i), :]
            c_temp[J_reduced.astype(int)] = inf
            index = argmin(c_temp)
            p_s[index] += weight[int(i)]

        scenario_reduced = scenario[J.astype(int), :]
        weight_reduced = p_s[J.astype(int)]

        return scenario_reduced, weight_reduced


if __name__ == "__main__":
    n_scenario = 200
    scenario = random.random((n_scenario, 10))
    weight = ones(n_scenario) / n_scenario
    n_reduced = int(n_scenario / 2)
    power = 2
    scenario_reduction = ScenarioReduction()

    (scenario_reduced, weight_reduced) = scenario_reduction.run(scenario=scenario, weight=weight, n_reduced=n_reduced,
                                                                power=power)

    print(scenario_reduced)
