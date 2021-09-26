"""
The bidding strategy strategy for energy hubs
This function is to provide
1) a deterministic day-ahead bidding strategy for hybrid AC/DC multiple micro-grids
2) a stochstic bidding strategy
3) start-up and shut-down of resources
4) decomposition algorithm, Benders decomposition
"""
import sys
sys.path.append('/home/matrix/PycharmProjects/Optimization/solvers')