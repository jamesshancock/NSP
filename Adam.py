'''
This file contains the functions for Adam optimization
'''

from quantumPart import *
import time
import numpy as np

def Adam_minimize(parameters):
    '''
    This function uses Adam optimization to minimize the QUBO - provided by GitHub Copilot
    '''
    parameters, n, k = inferredParameters(parameters)
    k, npara = Npara(parameters['N'])
    init_params = np.random.rand(npara)
    grad = grad_cost
    tik = time.perf_counter()
    optimized_params = adam_optimizer(grad, init_params, parameters)
    tok = time.perf_counter()
    solution = parasToSolution(optimized_params,parameters)
    timeTaken = round(tok-tik,4)
    return solution, timeTaken
     
def adam_optimizer(grad, init_params, parameters):
    '''
    Adam optimization - provided by GitHub Copilot
    '''
    alpha = parameters['alphaAdam']
    b1 = parameters['b1']
    b2 = parameters['b2']
    eps = parameters['eps']
    num_iters = parameters['num_iters']
    m = np.zeros_like(init_params)
    v = np.zeros_like(init_params)
    for i in range(num_iters):
        g = grad(init_params, parameters)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * np.square(g)
        m_hat = m / (1 - np.power(b1, i + 1))
        v_hat = v / (1 - np.power(b2, i + 1))
        init_params -= alpha * m_hat / (np.sqrt(v_hat) + eps)
    return init_params

def grad_cost(paras, parameters):
    '''
    Calculates the gradient of the cost function using the parameter shift rule - provided by GitHub Copilot
    '''
    grad = np.zeros_like(paras)
    shift = np.pi / 2
    for i in range(len(paras)):
        shifted_paras = paras.copy()

        # Shift parameter up
        shifted_paras[i] += shift
        cost_up = Cost(shifted_paras, parameters)

        # Shift parameter down
        shifted_paras[i] -= 2 * shift
        cost_down = Cost(shifted_paras, parameters)

        # Compute gradient
        grad[i] = (cost_up - cost_down) / 2

    return grad