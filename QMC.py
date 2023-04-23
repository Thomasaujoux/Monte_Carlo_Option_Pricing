import numpy as np
import chaospy as ch
from scipy import stats
from scipy.stats import qmc

import calcul


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
    sob_array = []
    for i in range(nb_samples):
        sampler = qmc.Sobol(d=1, scramble=True) 
        sob = sampler.random_base2(m=5)
        sob = sob.reshape(1,32)
        sob_array.append(sob[0][0:21])
    return sob_array


def multiCIR_QMC(alpha, b, sigma, T, k, S_0, nb_samples):
    
    dt = T/k
    multiCIR = []
    
    epsilon_s = stats.norm.ppf(sobol_generator(nb_samples, k+1), scale=np.sqrt(dt))
    
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
    
    epsilon_s = stats.norm.ppf(sobol_generator_random(nb_samples, k+1), scale=np.sqrt(dt))
    
    for j in range(nb_samples):
        S = np.zeros(k+1)
        S[0] = S_0
            
        for i in range(1, k+1):
            dS = alpha * (b - S[i-1]) * dt + sigma * np.sqrt(S[i-1]) * epsilon_s[j, i]
            S[i] = S[i-1]+ dS
                
        multiCIR.append(S)
    return multiCIR


def QMC_mc_sim(nb_samples, k, S_0, T, r, sigma, K, alpha, b):
    """
    Conducts MC simulation,
    
    INPUT:
        nb_samples (int): Number of samples in simulation
        k (int): Number of price step we aim to simulate in each path
        S_0 (float): Underlying asset price at time zero
        T (float): Time period of option contract
        r (float): Risk-netural interest rate
        sigma (float): Volatility in the environment
        K (float): Exercise price of the option
        alpha (float): taux de convergence
        b (float): taux de convergence
        
    OUTPUT:
        (Numpy.ndarray): A one-dimensional array of present value of simulated payoffs
    """
    present_payoffs = np.zeros(nb_samples)
    multiCIR_QMC2 = multiCIR_QMC(alpha, b, sigma, T, k, S_0, nb_samples)
    
    for i in range(nb_samples):
        present_payoffs[i] = calcul.pv_calc(calcul.payoff_calc(multiCIR_QMC2[i], K), r, T)
    return(present_payoffs)

def QMC_mc_sim_random(nb_samples, k, S_0, T, r, sigma, K, alpha, b):
    """
    Conducts MC simulation,
    
    INPUT:
        nb_samples (int): Number of samples in simulation
        k (int): Number of price step we aim to simulate in each path
        S_0 (float): Underlying asset price at time zero
        T (float): Time period of option contract
        r (float): Risk-netural interest rate
        sigma (float): Volatility in the environment
        K (float): Exercise price of the option
        alpha (float): taux de convergence
        b (float): taux de convergence
        
    OUTPUT:
        (Numpy.ndarray): A one-dimensional array of present value of simulated payoffs
    """
    present_payoffs = np.zeros(nb_samples)
    multiCIR_QMC_random2 = multiCIR_QMC_random(alpha, b, sigma, T, k, S_0, nb_samples)
    
    for i in range(nb_samples):
        present_payoffs[i] = calcul.pv_calc(calcul.payoff_calc(multiCIR_QMC_random2[i], K), r, T)
    return(present_payoffs)

