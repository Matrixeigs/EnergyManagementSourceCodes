from gurobipy import *
model = read('/Users/tianyangzhao/Dropbox/Codes/Distributionally robust optimization/DRO.lp')
model.optimize()
