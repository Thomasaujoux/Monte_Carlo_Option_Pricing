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
