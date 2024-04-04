'''
//                                                                     //
// This code has been written by James Hancock unless otherwise stated //
//                                                                     //
//                       Contact information:                          //
//                   james.hancock@plymouth.ac.uk                      //
//                                                                     //
//                           Instiuiton:                               //
//                      University of Plymouth                         //
//                                                                     //
// This code was started on: 24/01/2024                                //
// This code was last edited on: 15/02/2024                            //
//                                                                     //
'''
'''
Parameters
'''
Solver = 'new' # The solver to use

global parameters
inputParameters = {
    'alpha': 4, # Hyperparameter inside tanh
    'beta': 1/2, # Hyperparameter outside LossReg
    'vu': 460, # Estimate of final output value
    'ClassicalOptimizationRoutine': 'COBYLA' # How many spin matrices appear in the terms
}
bayesianParameters = {'lengthscale': 1, 
                      'variance': 1}

adamParameters = {'alphaAdam': 0.001,
                  'b1': 0.9,
                  'b2': 0.999,
                  'eps': 1e-8,
                  'num_iters': 40}
setParameters = {
    'a': 7/2, # Hard shift constraint weight
    'N': 8, # Number of nurses
    'D': 8, # Number of days
    'lambd': 1.3, # Hard nurse constraint weight
    'gamma': 0.3, # Soft nurse constraint weight
    'FixedK': 2, # How many spin matrices are chosen to form mapping
    'shot': 600 # Shots used by quantum computer
} # The solver to use
SamplesRequired = 64
GurobiTest = False
'''
Packages
'''
import math
import numpy as np
from scipy.optimize import minimize
import time
import multiprocessing as mp

'''
Modules
'''
from gurobi import *
from Adam import *
#from Bayesian import *
from NSPQUBO import *
from NSPvisualization import *
from processing import *
from quantumPart import *
from vqeBasedSolver import *
  
'''
Functions
'''
def minimization(parameters,method):
    '''
    This minimizes the cost function
    '''
    parameters, n, k = inferredParameters(parameters)
    nvars = 3*math.comb(n,k)

    #print("We are using an n value of: "+str(n))
    #print("We are using an k value of: "+str(k))
    #print("This problem needs "+str(parameters['D']*parameters['N'])+" binary variables")
    #print("We have "+str(nvars)+" binary parameters available, unused will not be included in calculations")
    #print("The number of variational parameters needed is: "+str(parameters['npara']))
    
    npara = parameters['npara']
    guess = np.array([2*math.pi*np.random.uniform(0,1) for pn in range(npara)]) # Random guess
    tik = time.perf_counter()
    mini = minimize(Cost, guess, args = (parameters), method=method, options={'disp': False})
    tok = time.perf_counter()
    parameters['shot'] *= 3
    solution = parasToSolution(mini.x,parameters)
    parameters['shot'] /= 3
    parameters['shot'] = int(parameters['shot'])
    timeTaken = round(tok-tik,4)
   
    return solution, timeTaken

def solverFunc(Solver,parameters,trueMax):
    '''
    This function runs the solver a number of times and stores the results
    '''
    global TotalErrors
    global TotalTimeTaken
    prinT = False
    
    if Solver == 'VQE':
        solution, timeTaken = CVaR_time_minimize(parameters,parameters['ClassicalOptimizationRoutine'])
    elif Solver == 'Gurobi':
        solution, timeTaken = Gurobi_minimize(parameters)
    elif Solver == 'Adam':
        solution, timeTaken = Adam_minimize(parameters)
    elif Solver == 'Bayesian':
        bayesianParameters = {'lengthscale': 1, 'variance': 1}
        solution, timeTaken = Bayesian_minimize(parameters)
    else:
        solution, timeTaken = minimization(parameters,parameters['ClassicalOptimizationRoutine'])
    optimal = False
    
    approxMax = np.matmul(solution,np.matmul(nurseSchedulingQUBO(parameters),solution))
    approxMax = round(approxMax,4)
    
    error = approxMax
    error = round(error,4)
    #TotalErrors.append(error)
    #TotalTimeTaken.append(timeTaken) # Would be nice to get this working
    if error == 0.0:
        optimal = True
    if prinT == True:
        print("Found solution: ")
        print(solutionToGrid(solution,parameters))
        print("The approximate minimum value is: "+str(approxMax))
        print("The relative error is: "+str(error))  
        print("This took "+str(timeTaken)+" seconds")
    return error, timeTaken, optimal, solution

def printFunction(Errors, TimeTaken, optimals, bestSolution):
    print("=======================================")
    print("The average error is: "+str(round(np.mean(Errors),4)))
    print("The standard deviation of the error is: "+str(round(np.std(Errors),4)))
    print("$"+str(round(np.mean(Errors),4))+" \pm "+str(round(np.std(Errors),4))+"$")
    print("Its best performance was a relative error of "+str(min(Errors))+" and its worst performance was a relative error of "+str(max(Errors))+".")
    print("The best solution found was: "+str(bestSolution))
    print("The number of times it found the optimal solution was: "+str(optimals)+" = "+str((optimals/TestLoops)*100)+"%")
    return None

def loopFunc(Solver,parameters,trueMax,TestLoops):
    '''
    This function runs the solver a number of times and stores the results
    '''
    bestSolution = []
    Errors = []
    optimals = 0
    TimeTaken = []
    for katy in range(TestLoops):
        error, timeTaken, optimal, solution = solverFunc(Solver,parameters,trueMax)
        Errors.append(error)
        TimeTaken.append(timeTaken)
        if optimal == True:
            optimals += 1
    print("Errors: ")
    print(Errors)
    print("Time taken: ")
    print(TimeTaken)
    return Errors, TimeTaken, optimals, bestSolution

def findingMin(parameters):
    '''
    This function finds the minimum value of the cost function
    '''
    ExhaustiveSearch = True
    ShouldPlot = False
    parameters['Q'] = quboToIsing(nurseSchedulingQUBO(parameters))
    trueSolution = searchAndPlot(ExhaustiveSearch,ShouldPlot,parameters,[0])
    print(trueSolution)
    trueMax = np.matmul(trueSolution,np.matmul(nurseSchedulingQUBO(parameters),trueSolution))
    trueMax = round(trueMax,4)
    print("The true minimum value is: "+str(trueMax))
    return trueMax


if __name__ == '__main__':
    Ncores = mp.cpu_count()
    print("We have "+str(Ncores)+" cores available")
    parameters = {**inputParameters, **setParameters,**bayesianParameters,**adamParameters}
    #trueMax = findingMin(parameters)
    trueMax = 0
    TestLoops = math.ceil(SamplesRequired/Ncores)
    for counter in range(1):
        processes = []
        for i in range(Ncores):
            p = mp.Process(target=loopFunc, args=(Solver,parameters,1,TestLoops))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    if GurobiTest == True:
        loopFunc('Gurobi',parameters,trueMax,TestLoops)
    



