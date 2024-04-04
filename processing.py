'''
This file contains all the pre/post processing functions for the nurse scheduling problem
'''
import numpy as np
import math
from NSPvisualization import *
from NSPQUBO import *

def Npara(n):
    '''
    Calculates the number of parameters
    '''
    test = 1
    k = 0
    while test < (2**(n+1) + 2*n):
        k += 1
        test = 2*k*n
    npara = 2*k*n
    return k, npara

def exhaustiveSearchSpin(Q):
    '''
    This returns the minimum value solution vector of the Ising problem
    '''
    scale = len(Q)
    minVal = math.inf
    solution = None
    c = 0
    for i in range(2**scale):
        binary = bin(i)[2:]
        binary = '0'*(scale-len(binary)) + binary
        binary = np.array([int(x) for x in binary])
        binary = 2*binary - 1
        val = np.matmul(binary,np.matmul(Q,binary))
        if val < minVal:
            minVal = val
            solution = binary
            c += 1
    print("Min spin value: "+str(minVal))
    return solution

def bestChoice(ourValue):
    '''
    This returns the best choice of n and k for the problem
    '''
    MAX = ourValue+1
    store = np.zeros((MAX,MAX))
    for n in range(MAX):
        for k in range(n+1):
            nCk = math.comb(n,k)
            store[n,k] = 3*nCk
    bestChoice = 0
    for nk in range(MAX):
        for kk in range(nk+1):
            if store[nk,kk] >= ourValue:
                bestChoice = [nk,kk]
                break
        if bestChoice != 0:
            break
    return bestChoice

def bestChoiceFixedk(ourValue,k):
    '''
    This returns the best choice of n for the problem
    '''    

    MAX = ourValue+1
    store = np.zeros((MAX))
    for n in range(MAX):
        nCk = math.comb(n,k)
        store[n] = 3*nCk
    bestChoice = 0
    for nk in range(MAX):
        if store[nk] >= ourValue:
            bestChoice = nk
            break
    return [bestChoice,k]

def bestChoiceFixedkZonly(ourValue,k):
    '''
    This returns the best choice of n for the problem with noyl Z matrices
    '''
    MAX = ourValue+1
    store = np.zeros((MAX))
    for n in range(MAX):
        nCk = math.comb(n,k)
        store[n] = nCk
    bestChoice = 0
    for nk in range(MAX):
        if store[nk] >= ourValue:
            bestChoice = nk
            break
    return [bestChoice,k]

def searchAndPlot(ExhaustiveSearch,ShouldPlot,parameters,solution):
    '''
    This searches for the optimal solution and plots the solution
    '''
    if ExhaustiveSearch == True:
        print("Finding the optimal solution")
        trueSolution = exhaustiveSearchSpin(parameters['Q'])
        trueSolution = (trueSolution + 1)/2
        print("Optimal solution:")
        print(solutionToGrid(trueSolution,parameters))
    
    if ShouldPlot == True:
        print("Plotting the solution")
        if ExhaustiveSearch == True:
            gridPlotter(trueSolution, True,parameters)
        gridPlotter(solution, True,parameters)
    return trueSolution

def inferredParameters(parameters):
    '''
    This function infers the secondary parameters from the initial parameters
    '''
    FIXEDK = parameters['FixedK']   

    Qbin = nurseSchedulingQUBO(parameters)
    choices = bestChoiceFixedk(parameters['D']*parameters['N'],FIXEDK)
    n = choices[0]
    k = choices[1]
    K, npara = Npara(n)
    parameters['Q'] = quboToIsing(Qbin)
    #parameters['Q'] = Qbin
    parameters['npara'] = npara
    parameters['n'] = n
    parameters['k'] = k
    parameters['m'] = parameters['N']*parameters['D']
    return parameters, n, k

def inferredParametersZonly(parameters):
    '''
    This function infers the secondary parameters from the initial parameters
    '''
    FIXEDK = parameters['FixedK']  

    Qbin = nurseSchedulingQUBO(parameters)
    choices = bestChoiceFixedkZonly(parameters['D']*parameters['N'],FIXEDK)
    n = choices[0]
    k = choices[1]
    _, npara = Npara(n)
    parameters['Q'] = quboToIsing(Qbin)
    parameters['npara'] = npara
    parameters['n'] = n
    parameters['k'] = k
    parameters['m'] = parameters['N']*parameters['D']
    return parameters, n, k