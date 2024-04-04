'''
This file contains the Gurobi solver for the NSP
'''

from NSPQUBO import *
import gurobipy as gp
from gurobipy import GRB
import time

def Gurobi_minimize(parameters):
    '''
    This function uses the Gurobi solver to minimize the QUBO
    '''
    Q = nurseSchedulingQUBO(parameters)
    n = len(Q)
    m = gp.Model("qp")
    m.setParam('OutputFlag', 0)
    x = m.addMVar(n, vtype=GRB.BINARY, name="x")
    obj = 0
    for i in range(n):
        for j in range(n):
            obj += Q[i][j]*x[i]*x[j]
    m.setObjective(obj, GRB.MINIMIZE)
    tok = time.perf_counter()
    m.optimize()
    tik = time.perf_counter()
    solution = x.X
    timeTaken = round(tik-tok,4)
    return solution, timeTaken