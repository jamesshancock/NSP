'''
This is the old vqe file from the WFLO problem which has been updated to work with the NSP
'''
import math
import numpy as np
import random
import scipy
from scipy.optimize import minimize
from qiskit import *
from qiskit import Aer
import qiskit.opflow
import time
from NSPQUBO import *
from sympy import *
import sympy as sp

def z_list(nb):
    b = ['{:0{}b}'.format(i, nb) for i in range(nb*nb-1)]
    if (nb%2) == 0:
        x = ''
        for k in range(nb):
            x = x + '1'
        b.append(x)
    z = []
    for bli in b:
        b0 = list(bli)
        for bk in range(len(b0)):
            if b0[bk] == '1':
                b0[bk] = '3'
        bli = "".join(b0)
        z.append(bli)
    return z

def QtoVQE(Q):
    Q = Matrix(Q)
    Shape = shape(Q)
    n = Shape[0]
    s = '0'*n
    s = list(s)
    s[1] = '2'
    s = "".join(s)
    I = sp.symbols('0'*n, real = True)
    par = []
    X = zeros(n,n)
    for i in range(n):
        for j in range(n):
            s0 = '0'*n
            s0 = list(s0)
            s0[i] = '3'
            s0 = "".join(s0)
            s1 = '0'*n
            s1 = list(s1)
            s1[j] = '3'
            s1 = "".join(s1)
            s2 = '0'*n
            s2 = list(s2)
            s2[i] = '3'
            s2[j] = '3'
            s2 = "".join(s2)    

            t0 = sp.symbols(s0, real = True)
            t1 = sp.symbols(s1, real = True)
            if i != j:
                t2 = sp.symbols(s2, real = True)
            else:
                t2 = I
            X[i,j] = I + t0 + t1 + t2
    C = []
    term = 0
    for i in range(n):
        for k in range(n):
            term = term + Q[i,k]*X[k,i]
    zlist = z_list(n)
    zterms = []
    for z in range(len(zlist)):
        exec(f'zt{z} = sp.symbols(zlist[z], real = True)')
        exec(f'zterms.append(zt{z})')

    h = []
    for k in range(len(zterms)):
        coef = term.coeff(zterms[k])
        h.append(str(0.25*coef))
        h.append(zlist[k])
    h_len = int(len(h)/2)
    dellist = []
    for hk in range(h_len):
        if h[2*hk] == '0':
            dellist.append(2*hk)
            dellist.append(2*hk+1)
    h0 = []
    for k in range(len(h)):
        if (k in dellist) == False:
            h0.append(h[k])
    return h0

def coeffH(h):
    K = int(len(h)/2)
    Coef = []
    matri = []
    for co in range(K):
        Coef.append(float(h[2*co]))
        matri.append([int(h[2*co+1][j]) for j in range(n)])
    return Coef, matri

def binary_list(nb):
    b = ['{:0{}b}'.format(i, nb) for i in range(nb*nb-1)]
    if (nb%2) == 0:
        x = ''
        for k in range(nb):
            x = x + '1'
        b.append(x)
    return b

def ryBlock(paray,QC):
    for ky in range(len(paray)):
        QC.ry(paray[ky],ky)
    return QC

def pBlock(varp,QC):
    for kp in range(len(varp)):
        QC.p(varp[kp],kp)
    return QC

def cnotBlock(QC):
    for kc in range(n-1):
        QC.cx(kc,kc+1)
    return QC

def measure_0(qubit,QC): #function used to measure in the sigma_1 basis
    return QC
        
def measure_1(qubit,QC): #function used to measure in the sigma_1 basis
    QC.h(qubit)
    QC.measure(qubit,qubit)
    return QC
    
def measure_2(qubit,QC): #function used to measure in the sigma_2 basis
    QC.sdg(qubit)
    QC.h(qubit)
    QC.measure(qubit,qubit)
    return QC
    
def measure_3(qubit,QC): #function used to measure in the sigma_3 basis
    QC.measure(qubit,qubit)
    return QC

def measure(case,qubit,QC):
    if case == 1:
        measure_1(qubit,QC)
    elif case == 2:
        measure_2(qubit,QC)
    elif case == 0:
        measure_0(qubit, QC)
    else:
        measure_3(qubit,QC)
    return QC

def S(sn):
    S = sn #number of layers of gates
    L = []
    for kl in range(S):
        if (kl % 2) == 0:
            L.append(0)
        else:
            L.append(1)
    return S, L

def para(var,QC):
    for kPara in range(S1):
        paras = var[kPara*n:(kPara+1)*n]
        ryBlock(paras,QC)
        if (kPara%2) == 0:
            cnotBlock(QC)
    S2 = S1 + 1
    paras = var[S1*n:(S2)*n]
    ryBlock(paras,QC)
    return QC
               
def EV(counts,shot,M):
    b = binary_list(n)
    signlist = {}
    for x1 in b:
        c1 = x1.count('1')
        if (c1%2) == 0:
            signlist[x1] = 1
        else:
            signlist[x1] = -1
    counts_store = []
    for x in b:
        if x in counts:
            temp = counts[x]/shot
            counts_store.append(signlist[x]*temp)
    EV = sum(counts_store)
    
    return EV
          
def sim(QC, shot):
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(QC, backend_sim), shots=shot)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(QC)
    return counts

def CVaR(Var,h,shot,alpha):  
    var_store.append(Var)
    Coef, matri = coeffH(h)
    cvar = 0
    n = len(h[1])
    q = QuantumRegister(n)
    c = ClassicalRegister(n)
    EV_store = []
    K = math.ceil(len(Coef)*alpha)
    for k_coef in range(len(Coef)):
        if matri[k_coef] == [0]*n:
            EV1 = 1
        else:
            EV1 = 0
            circ = QuantumCircuit(q,c)
            para(Var,circ)
            for g in range(n):
                M = matri[k_coef][g]
                measure(M,g,circ)
            numbers = sim(circ,shot)
            
            EV1 = EV(numbers,shot,matri[k_coef])
            
        EV_store.append(Coef[k_coef]*EV1)
    EV_store.sort()
    for k in range(K):
        cvar = cvar + EV_store[k]
    return cvar

def CVaR_time(Var,h,shot,alpha):
    Coef, matri = coeffH(h)
    cvar = 0
    q = QuantumRegister(n)
    c = ClassicalRegister(n)
    EV_store = []
    K = math.ceil(len(Coef)*alpha)
    for k_coef in range(len(Coef)):
        if matri[k_coef] == [0]*n:
            EV1 = 1
        else:
            EV1 = 0
            circ = QuantumCircuit(q,c)
            para(Var,circ)
            for g in range(n):
                M = matri[k_coef][g]
                measure(M,g,circ)
            numbers = sim(circ,shot)
            EV1 = EV(numbers,shot,matri[k_coef])
            
        EV_store.append(Coef[k_coef]*EV1)
    EV_store.sort()
    for k in range(K):
        cvar = cvar + EV_store[k]
    return cvar

def CVaR_minimize(parameters,method):
    Q = nurseSchedulingQUBO(parameters)
    h = QtoVQE(Q)
    argsli = (h,parameters['shot'],1)
    global n
    global S1
    global var_store
    var_store = []
    n = len(h[1])
    S1, L = S(n)
    npara = int(S1*n)
    guess = [2*math.pi*random.uniform(0,1) for k in range(npara)]
    CVaR_min = minimize(CVaR, guess, args = argsli, method=method,
                           options={'disp': False, 'return_all': True, 'ftol': 1e-6})
    return CVaR_min, var_store

def CVaR_time_minimize(parameters,method):
    Q = nurseSchedulingQUBO(parameters)
    h = QtoVQE(Q)
    argsli = (h,parameters['shot'],1)
    global n
    global S1
    n = len(h[1])
    S1, L = S(n)
    npara = int(S1*n)
    guess = [2*math.pi*random.uniform(0,1) for k in range(npara)]
    tik = time.perf_counter()
    CVaR_min = minimize(CVaR_time, guess, args = argsli, method=method,
                           options={'disp': False})
    tok = time.perf_counter()
    timeTaken = round(tok-tik,4)
    numbers = VQEtoQ(CVaR_min.x,n)
    best = 0
    for key in list(numbers.keys()):
        if numbers[key] > best:
            bestkey = key
            best = numbers[bestkey]
    solution = [int(bestkey[k]) for k in range(len(bestkey))]
    return solution, timeTaken

def VQEtoQ(Var,N):
    global n
    global S1
    n = N
    S1, L = S(n)
    q = QuantumRegister(n)
    c = ClassicalRegister(n)
    circ = QuantumCircuit(q,c)
    para(Var,circ)
    for g in range(n):
        circ.x(g)
        measure(3,g,circ)
    numbers = sim(circ,10000)
    return numbers
