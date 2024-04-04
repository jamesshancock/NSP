'''
This file contains all the functions used to create the QUBO for the nurse scheduling problem
'''
import numpy as np

def W(d):
    '''
    This returns the required workdforce for a given day
    '''
    return 1

def E(n):
    '''
    This returns the effort for a given nurse
    '''
    return 1

def F(n,N,D):
    '''
    This is the number of work days each nurse would like to work
    '''
    return np.floor(D/N)

def h1(n,d):
    '''
    This gives the availability of a nurse on a given day, larger values imply they are busy
    '''
    return 1

def h2(d):
    '''
    This changes if there are some unpreferrable days, larger values imply they are unpreferrable
    '''
    return 1

def G(n,d):
    '''
    This returns the preference for a given nurse on a given day
    '''
    hOne = h1(n,d)
    hTwo = h2(d)

    return hOne*hTwo

def labelling(N,D):
    '''
    This returns the variable labelling for the QUBO
    '''

    variables = []
    for i in range(N):
        for j in range(D):
            variables.append((i,j))
    labels = {}
    for i in range(len(variables)):
        labels[variables[i]] = i
    return labels

def HS(N,D):
    '''
    This returns the hard shift constraint matrix, J
    '''
    labels = labelling(N,D)
    scale = N*D

    J = np.zeros((scale,scale))
    for n in range(N):
        for d in range(D):
            if d < D-1:
                J[labels[(n,d)],labels[(n,d+1)]] = 1
    return J

def HN(N,D):
    '''
    This returns the hard nurse constraint matrix, L
    '''
    labels = labelling(N,D)
    scale = N*D
    L = np.zeros((scale,scale))
    for d in range(D):
        for n in range(N):
            for nprime in range(N):
                if n == nprime:
                    L[labels[(n,d)],labels[(nprime,d)]] = E(n)**2 - 2*N*W(d)*E(n)
                else:
                    L[labels[(n,d)],labels[(nprime,d)]] = 2*E(n)*E(nprime)

    return L

def sn(N,D):
    '''
    This returns the soft nurse constraint matrix, S
    '''
    labels = labelling(N,D)
    scale = N*D
    S = np.zeros((scale,scale))
    for n in range(N):
        for d in range(D):
            for dprime in range(D):
                if d == dprime:
                    S[labels[(n,d)],labels[(n,dprime)]] = G(n,d)**2 - 2*D*F(n,N,D)*G(n,d)
                else:
                    S[labels[(n,d)],labels[(n,dprime)]] = 2*G(n,d)
    return S

def nurseSchedulingQUBO(parameters):
    '''
    This returns the QUBO for the nurse scheduling problem
    '''
    N = parameters['N']
    D = parameters['D']
    a = parameters['a']
    lambd = parameters['lambd']
    gamma = parameters['gamma']

    labels = labelling(N,D)
    scale = N*D

    J = a*HS(N,D)
    L = lambd*HN(N,D)
    S = gamma*sn(N,D)

    Q = J + L + S
    
    # This ensures that the Q matrix is symmetric
    Q = 1/2*(Q + Q.T)
    
    return Q

def quboToIsing(qubo_matrix):
    # Get the size of the QUBO matrix
    n = len(qubo_matrix)

    # Initialize the Ising Hamiltonian matrix
    ising_matrix = np.zeros((n, n))

    # Fill in the Ising Hamiltonian matrix based on QUBO matrix
    for i in range(n):
        for j in range(i, n):
            # Diagonal elements (local fields)
            if i == j:
                ising_matrix[i, j] = qubo_matrix[i, i]
            else:
                # Off-diagonal elements (coupling strengths)
                ising_matrix[i, j] = 2 * qubo_matrix[i, j]
                ising_matrix[j, i] = 2 * qubo_matrix[i, j]
    return ising_matrix



