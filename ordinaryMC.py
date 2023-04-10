import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import CIR

def payoff_calc(S_ti, K):
    """
    This function calculates future payoff of the asian option based on arithmetic average of the price path
    
    INPUT:
        S_ti (numpy.ndarray): A one-dimensional array of stock final prices
        K (float): Exercise price of the option
    
    OUTPUT:
        (numpy.ndarray): A one dimensional array of payoffs for different prices
    """
    
    payoff = np.maximum(0, np.mean(S_ti) - K)
    return payoff

def pv_calc(payoff, r, T):
    """
    Calculates present value of an amount of money in future.
    
    INPUT:
        payoff (float): Future value of money
        r (float): Risk neutral interest rate
        T (float): Period of time
    
    OUTPUT:
        (float): Present value of FV
    """
    
    return payoff * np.exp(-r * T)

def ordinary_mc_sim(nb_samples, k, S_0, T, r, sigma, K, alpha, b):
    """
    Conducts MC simulation,
    
    INPUT:
        no_of_paths (int): Number of samples in simulation
        n_steps (int): Number of price step we aim to simulate in each path
        S_0 (float): Underlying asset price at time zero
        T (float): Time period of option contract
        r (float): Risk-netural interest rate
        sigma (float): Volatility in the environment
        x_price (float): Exercise price of the option
        
    OUTPUT:
        (Numpy.ndarray): A one-dimensional array of present value of simulated payoffs
    """
    present_payoffs = np.zeros(nb_samples)
    multiCIR = CIR.multiCIR(alpha, b, sigma, T, k, S_0, nb_samples)
    
    for i in range(nb_samples):
        present_payoffs[i] = pv_calc(payoff_calc(multiCIR[i], K), r, T)
    return(present_payoffs)

def sim_iterator(max_sample, k, S_0, T, r, sigma, K, alpha, b):
    """
    Iterates simulation with different sample sizes (form 10 to a maximum size with steps of 10)
    
    INPUT:
        max_sample (int): Maximum sample size for the iteration of simulations
        n_steps (int): Number of price step we aim to simulate in each path
        S_0 (float): Underlying asset price at time zero
        T (float): Time period of option contract
        r (float): Risk-netural interest rate
        sigma (float): Volatility in the environment
        x_price (float): Exercise price of the option
        method (string): 'sobol', 'halton' or 'ordinary'
    
    OUTPUT:
        (numpy.ndarray): confidence intervals of the simulations
        (numpy.ndarray): price estimations of the simulations
    """
    
    mean_pv_payoffs = np.zeros(int(max_sample / 10))
    
    for nb_samples in range(10, max_sample + 1, 10):
        present_payoffs = ordinary_mc_sim(nb_samples,k, S_0, T, r, sigma, K, alpha, b)
        mean_pv_payoffs[int(nb_samples/10) - 1] = np.mean(present_payoffs)

    
    return mean_pv_payoffs