'''
This file contains the functions for Bayesian optimization
'''

from quantumPart import *
import time
import numpy as np
from processing import *
import GPyOpt
import GPy

def Bayesian_minimize(parameters):
    '''
    This function uses Bayesian optimization to minimize the QUBO - provided by GitHub Copilot
    '''
    parameters, n, k = inferredParameters(parameters)
    k, npara = Npara(parameters['N'])
    bounds = [{'name': 'x'+str(i), 'type': 'continuous', 'domain': (0, 1)} for i in range(npara)]
    tik = time.perf_counter()
    optimized_params = Bayesian_optimizer(parameters, bounds)
    tok = time.perf_counter()
    solution = parasToSolution(optimized_params,parameters)
    timeTaken = round(tok-tik,4)
    return solution, timeTaken


def Bayesian_optimizer(parameters, bounds, num_iters=300):
    '''
    Bayesian optimization - provided by GitHub Copilot
    '''
    # Define the kernel
    kernel = GPy.kern.RBF(input_dim=len(bounds), variance=parameters['variance'], lengthscale=parameters['lengthscale'])

    optimizer = GPyOpt.methods.BayesianOptimization(f=lambda x: Cost(x.flatten(), parameters), 
                                                    domain=bounds, 
                                                    model_type='GP', 
                                                    kernel=kernel)
    optimizer.run_optimization(max_iter=num_iters)
    return optimizer.x_opt
