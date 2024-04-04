'''
This file contains all the quantum functions for the nurse scheduling problem
'''
import numpy as np 
import math
from qiskit import *
from sympy.utilities.iterables import multiset_permutations
from NSPQUBO import *
from processing import *

global parametersBayesian, nBayes, kBayes
parametersBayesian = {'alpha': 12, # Hyperparameter inside tanh
              'beta': 1/2, # Hyperparameter outside LossReg
              'vu': 112, # Estimate of final output value
              'shot': 600, # Shots used by quantum computer
              'N': 4, # Number of nurses
              'D': 4, # Number of days
              'a': 7/2, # Hard shift constraint weight
              'lambd': 1.3, # Hard nurse constraint weight
              'gamma': 0.3, # Soft nurse constraint weight
              'FixedK': 2}
parametersBayesian, nBayes, kBayes = inferredParameters(parametersBayesian)

def pauliVars(n,k,qubits):
    '''
    This function returns the Pauli variables for a given n and k
    '''
    ones = [1]*k
    zeros = [0]*(n-k)
    base = ones + zeros
    perms = list(multiset_permutations(base))
    Xs, Ys, Zs = [], [], []
    for perm in perms:
        Xs.append(list(perm))
        Ys.append([2*p for p in perm])
        Zs.append([3*p for p in perm])
    lis = Zs + Xs + Ys
    return lis[0:qubits]

def pauliVarsZonly(n,k,qubits):
    '''
    This function returns the Pauli variables for a given n and k, only Z matrices
    '''
    ones = [1]*k
    zeros = [0]*(n-k)
    base = ones + zeros
    perms = list(multiset_permutations(base))
    Zs = []
    for perm in perms:
        Zs.append([3*p for p in perm])
    lis = Zs
    return lis[0:qubits]

def NparaZonly(n):
    '''
    Calculates the number of parameters for only Z matrices
    '''
    return n**2

def para(circ,paras,n):
    '''
    Parameterizes the quantum circuit for a fully general ansatz
    '''
    K, npara = Npara(n)
    for block in range(K):
        parasTemp = paras[2*block*n:(2*(block+1)*n)]
        for qubit in range(n):
            circ.p(parasTemp[2*qubit],qubit)
            circ.ry(parasTemp[2*qubit+1],qubit)
            if qubit != n-1:
                circ.cx(qubit,qubit+1)
    return circ

def paraZonly(circ,paras,n):
    '''
    Parameterizes the quantum circuit for a just Z matrices
    '''
    n = int(math.sqrt(len(paras)))
    for u in range(n):
        for v in range(n):
            thetaj = paras[u+v]
            circ.ry(thetaj,v)
        for c in range(n-1):
            circ.cx(c,c+1)
    return circ

def LossTanh(Q,alpha,EVs):
    '''
    Calculates the loss function based on the spin matrix Q, the alpha value and the expected values of the Pauli variables
    '''
    cost = 0
    for i in range(0,Q.shape[0]):
        for j in range(0,Q.shape[0]):
            cost += Q[i,j]*math.tanh(alpha*EVs[i])*math.tanh(alpha*EVs[j])
    return cost

def LossSign(Q,alpha,EVs):
    '''
    Calculates the loss function based on the spin matrix Q, the alpha value and the expected values of the Pauli variables
    '''
    cost = 0
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            signi = math.copysign(1,EVs[i])
            signj = math.copysign(1,EVs[j])
            cost += Q[i,j]*signi*signj
    return cost

def LossRegTanh(alpha,beta,vu,m,EVs):
    '''
    Calculates the loss function with the regularisation term
    '''
    cost = 0
    for i in range(m):
        cost += math.tanh(alpha*EVs[i])**2
    cost *= 1/m
    costSq = cost**2

    return beta*vu*costSq

def LossRegSign(alpha,beta,vu,m,EVs):
    '''
    Calculates the loss function with the regularisation term
    '''
    cost = 0
    for i in range(m):
        signi = math.copysign(1,EVs[i])
        cost += signi**2
    cost *= 1/m
    costSq = cost**2

    return beta*vu*costSq

def sim(QC, shot):
    '''
    Simulates the quantum circuit and returns the counts
    '''
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(QC, backend_sim), shots=shot)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(QC)
    return counts

def measure(circ,spin,qubit):
    '''
    Measures the qubit in the correct basis
    '''
    if spin == 1:
        circ.h(qubit)
    elif spin == 2:
        circ.sdg(qubit)
        circ.h(qubit)
    circ.measure(qubit,qubit)
    return circ

def expectedValue(counts,shot):
    '''
    Calculates the expected value based on a set of counts
    '''
    keys = list(counts.keys())
    signList = {}
    for key in keys:
        ke = [int(key[k]) for k in range(len(key))]
        numberOnes = sum(ke)
        if (numberOnes%2) == 0:
            signList[key] = 1
        else:
            signList[key] = -1
    expecVal = 0
    for key in keys:
        expecVal += signList[key]*counts[key]/shot
    return expecVal

def vqe(paras,varStore,shot):
    '''
    The variational quantum eigensolver for reduced qubit QUBO
    '''
    keys = list(varStore.keys())
    EVs = {}
    n = len(varStore[0])
    for key in keys:
        matrix = varStore[key]
        q = QuantumRegister(n)
        c = ClassicalRegister(n)
        circ = QuantumCircuit(q,c)
        para(circ,paras,n)
        for k in range(n):
            if matrix[k] != 0:
                measure(circ,matrix[k],k)
        counts = sim(circ,shot)
        EVs[key] = expectedValue(counts,shot)
    return EVs

def vqeZonly(paras,varStore,shot):
    '''
    The variational quantum eigensolver for reduced qubit QUBO
    '''
    keys = list(varStore.keys())
    EVs = {}
    n = len(varStore[0])
    for key in keys:
        matrix = varStore[key]
        q = QuantumRegister(n)
        c = ClassicalRegister(n)
        circ = QuantumCircuit(q,c)
        para(circ,paras,n)
        for k in range(n):
            if matrix[k] != 0:
                measure(circ,3,k)
        counts = sim(circ,shot)
        EVs[key] = expectedValue(counts,shot)
    return EVs

def Cost(paras,parameters):
    '''
    Calculates the cost function
    '''
    alpha = parameters['alpha']
    beta = parameters['beta']
    vu = parameters['vu']
    m = parameters['m']
    shot = parameters['shot']
    Q = parameters['Q']
    n = parameters['n']
    k = parameters['k']
    pauliVarZ = pauliVars(n,k,parameters['N']*parameters['D'])
    varStore = {}
    for i in range(len(pauliVarZ)):
        varStore[i] = pauliVarZ[i]
    EVs = vqe(paras,varStore,shot)
    cost = LossTanh(Q,alpha,EVs) + LossRegTanh(alpha,beta,vu,m,EVs)
    return cost

def CostBayesian(paras):
    '''
    Calculates the cost function for the bayesian optimization
    '''
    alpha = parametersBayesian['alpha']
    beta = parametersBayesian['beta']
    vu = parametersBayesian['vu']
    m = parametersBayesian['m']
    shot = parametersBayesian['shot']
    Q = parametersBayesian['Q']
    n = parametersBayesian['n']
    k = parametersBayesian['k']
    pauliVarZ = pauliVars(nBayes,kBayes,parametersBayesian['N']*parametersBayesian['D'])
    varStore = {}
    for i in range(len(pauliVarZ)):
        varStore[i] = pauliVarZ[i]
    EVs = vqe(paras,varStore,shot)
    cost = LossTanh(Q,alpha,EVs) + LossRegTanh(alpha,beta,vu,m,EVs)
    return cost    

def CostBayesianZonly(paras):
    '''
    Calculates the cost function for the bayesian optimization
    '''
    alpha = parameters['alpha']
    beta = parameters['beta']
    vu = parameters['vu']
    m = parameters['m']
    shot = parameters['shot']
    Q = parameters['Q']
    n = parameters['n']
    k = parameters['k']
    pauliVarZ = pauliVarsZonly(n,k,parameters['N']*parameters['D'])
    varStore = {}
    for i in range(len(pauliVarZ)):
        varStore[i] = pauliVarZ[i]
    EVs = vqeZonly(paras,varStore,shot)
    cost = LossTanh(Q,alpha,EVs) + LossRegTanh(alpha,beta,vu,m,EVs)
    return cost

def CostZonly(paras,parameters):
    '''
    Calculates the cost function
    '''
    alpha = parameters['alpha']
    beta = parameters['beta']
    vu = parameters['vu']
    m = parameters['m']
    shot = parameters['shot']
    Q = parameters['Q']
    n = parameters['n']
    k = parameters['k']
    pauliVarZ = pauliVarsZonly(n,k,parameters['N']*parameters['D'])
    varStore = {}
    for i in range(len(pauliVarZ)):
        varStore[i] = pauliVarZ[i]
    EVs = vqeZonly(paras,varStore,shot)
    cost = LossTanh(Q,alpha,EVs) + LossRegTanh(alpha,beta,vu,m,EVs)
    return cost

def CostZeroMode(paras,parameters,Q):
    '''
    Calculates the cost function
    '''
    m = parameters['m']
    shot = parameters['shot']
    n = parameters['n']
    k = parameters['k']
    pauliVarZ = pauliVars(n,k,parameters['N']*parameters['D'])
    varStore = {}
    for i in range(len(pauliVarZ)):
        varStore[i] = pauliVarZ[i]
    EVs = vqe(paras,varStore,shot)
    oneZero = {}
    for key in EVs.keys():
        if EVs[key] > 0:
            oneZero[key] = 1
        else:
            oneZero[key] = 0
    cost = 0
    for u in range(Q.shape[0]):
        for v in range(Q.shape[0]):
            cost += Q[u,v]*oneZero[u]*oneZero[v]
    return cost

def parasToSolution(paras,parameters):
    '''
    Converts the parameters to a solution
    '''
    shot = parameters['shot']
    n = parameters['n']
    k = parameters['k']
    qubits = parameters['N']*parameters['D']

    pauliVarZ = pauliVars(n,k,qubits)
    varStore = {}
    for i in range(len(pauliVarZ)):
        varStore[i] = pauliVarZ[i]
    EVs = vqe(paras,varStore,shot)
    finalCost = Cost(paras,parameters)
    keys = list(EVs.keys())
    solution = []
    for key in keys:
        if EVs[key] > 0:
            solution.append(1)
        else:
            solution.append(0)
    return solution

def parasToSolutionZonly(paras,parameters):
    '''
    Converts the parameters to a solution
    '''
    shot = parameters['shot']
    n = parameters['n']
    k = parameters['k']
    qubits = parameters['N']*parameters['D']

    pauliVarZ = pauliVarsZonly(n,k,qubits)
    varStore = {}
    for i in range(len(pauliVarZ)):
        varStore[i] = pauliVarZ[i]
    EVs = vqeZonly(paras,varStore,shot)
    finalCost = CostZonly(paras,parameters)
    print("Final cost: "+str(finalCost))
    keys = list(EVs.keys())
    solution = []
    for key in keys:
        if EVs[key] > 0:
            solution.append(1)
        else:
            solution.append(0)
    return solution

def parasToSolutionZeroMode(paras,parameters):
    '''
    Converts the parameters to a solution
    '''
    shot = parameters['shot']
    n = parameters['n']
    k = parameters['k']
    qubits = parameters['N']*parameters['D']

    pauliVarZ = pauliVars(n,k,qubits)
    varStore = {}
    for i in range(len(pauliVarZ)):
        varStore[i] = pauliVarZ[i]
    EVs = vqe(paras,varStore,shot)
    Q = nurseSchedulingQUBO(parameters)
    finalCost = CostZeroMode(paras,parameters,Q)
    print("Final cost: "+str(finalCost))
    keys = list(EVs.keys())
    solution = []
    for key in keys:
        if EVs[key] > 0:
            solution.append(1)
        else:
            solution.append(0)
    return solution



