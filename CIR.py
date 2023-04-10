import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def CIR(alpha, b, sigma, T, k, S_0):
    dt = T/k
    S = np.zeros(k+1)
    S[0] = S_0
    for i in range(1, k+1):
        dS = alpha * (b - S[i-1]) * dt + sigma * np.sqrt(S[i-1] * dt) * np.random.normal(scale=np.sqrt(dt))
        S[i] = S[i-1] + dS
        if S[i] < 0 : S[i] = "Erreur"
    return S


def multiCIR(alpha, b, sigma, T, k, S_0, nb_samples): 
    multiCIR = []
    for i in range(nb_samples): 
        multiCIR.append(CIR(alpha, b, sigma, T, k, S_0))
    return multiCIR

def CIR_ML(alpha, b, sigma, T, k, S_0, L):
    delta = np.zeros(L+1)
    for l in range(0,L):
        delta[l] = 2 ** (-l)
    S = np.zeros(k+1)
    S[0] = S_0
    for l in range(0,L):
        for i in range(1, k+1):
            S[i] = delta[l] * alpha * (b - S[i-1]) + sigma * np.sqrt(S[i-1]) * np.random.normal(scale=delta[l]) + S[i-1]
            if S[i] < 0 : S[i] = "Erreur"
    return S

def multiCIR_ML(alpha, b, sigma, T, k, S_0, nb_samples, L): 
    multiCIR = []
    for i in range(nb_samples): 
        multiCIR.append(CIR_ML(alpha, b, sigma, T, k, S_0,L))
    return multiCIR