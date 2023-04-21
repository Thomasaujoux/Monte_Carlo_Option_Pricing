import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import chaospy as ch
from scipy import stats




def CIR(alpha, b, sigma, T, k, S_0):
    dt = T/k
    S = np.zeros(k+1)
    S[0] = S_0
    for i in range(1, k+1):
        dS = alpha * (b - S[i-1]) * dt + sigma * np.sqrt(S[i-1] ) * np.random.normal(scale=np.sqrt(dt))
        S[i] = S[i-1] + dS
        if S[i] < 0 : S[i] = "Erreur"
    return S

def multiCIR(alpha, b, sigma, T, k, S_0, nb_samples): 
    multiCIR = []
    for i in range(nb_samples): 
        multiCIR.append(CIR(alpha, b, sigma, T, k, S_0))
    return multiCIR


    
def CIR_ML(alpha, b, sigma, L, T, S_0): 
    delta = 2**(-L)
    k = int(np.ceil(T/delta))
    S = np.zeros(k+1)
    t = np.zeros(k+1)
    S[0] = S_0
    for i in range(1, k+1):
        dS = alpha * (b - S[i-1]) * delta + sigma * np.sqrt(S[i-1] ) * np.random.normal(scale=np.sqrt(delta))
        S[i] = S[i-1] + dS
        if S[i] < 0 : S[i] = "Erreur"
        t[i] = i*delta
    return t, S


def ML_principle(alpha, b, sigma, L, T, S_0): 
    ML_CIR = []
    M = CIR_ML(alpha, b, sigma, L, T, S_0)
    ML_CIR.append(M)
    A = ([ML_CIR[0][0][i] for i in range(0,len(ML_CIR[0][0]),2)],[ML_CIR[0][1][i] for i in range(0,len(ML_CIR[0][0]),2)])
    ML_CIR.append(A)
           
    return ML_CIR


def multiCIR_ML(alpha, b, sigma, L, T, S_0, nb_samples): 
    multiCIR = []
    for i in range(nb_samples): 
        multiCIR.append(ML_principle(alpha, b, sigma, L, T, S_0))
        print(multiCIR)
    return multiCIR






def sobol_generator(nb_samples, k):
    """
    Generates m samples (each for one path) each having n numbers in Sobol sequence.
    
    INPUT:
        m (int): number of samples
        n (int): number of Sobol sequence numbers in each sample
        
    OUTPUT:
        (numpy.ndarray): A two-dimensional array of Sobol sequence numbers for conducting QMC simulation
    """
    sob_array = np.empty((0, k))
    for i in range(nb_samples // 39 + 1):
        sob = ch.create_sobol_samples(k, 39, i)
        sob_array = np.append(sob_array, sob, axis=0)
    return sob_array

def sobol_generator_random(nb_samples, k):
    """
    Generates m samples (each for one path) each having n numbers in Sobol sequence.
    
    INPUT:
        m (int): number of samples
        n (int): number of Sobol sequence numbers in each sample
        
    OUTPUT:
        (numpy.ndarray): A two-dimensional array of Sobol sequence numbers for conducting QMC simulation
    """
    sob_array = np.empty((0, k))
    for i in range(nb_samples // 39 + 1):
        sob = ch.create_sobol_samples(k, 39, i)
        sob_array = np.append(sob_array, sob, axis=0)
    sob_array = (sob_array + np.random.uniform(size = (len(sob_array),k))) % 1
    
    return sob_array

def multiCIR_QMC(alpha, b, sigma, T, k, S_0, nb_samples):
    
    dt = T/k
    multiCIR = []
    
    epsilon_s = stats.norm.ppf(sobol_generator(nb_samples, k+1))
    
    for j in range(nb_samples):
        S = np.zeros(k+1)
        S[0] = S_0
            
        for i in range(1, k+1):
            dS = alpha * (b - S[i-1]) * dt + sigma * np.sqrt(S[i-1] ) * epsilon_s[j, i]
            S[i] = S[i-1]+ dS
                
        multiCIR.append(S)
    return multiCIR

def multiCIR_QMC_random(alpha, b, sigma, T, k, S_0, nb_samples):
    
    dt = T/k
    multiCIR = []
    
    epsilon_s = stats.norm.ppf(sobol_generator_random(nb_samples, k+1))
    
    for j in range(nb_samples):
        S = np.zeros(k+1)
        S[0] = S_0
            
        for i in range(1, k+1):
            dS = alpha * (b - S[i-1]) * dt + sigma * np.sqrt(S[i-1]) * epsilon_s[j, i]
            S[i] = S[i-1]+ dS
                
        multiCIR.append(S)
    return multiCIR