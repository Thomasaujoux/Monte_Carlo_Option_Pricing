import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import CIR
import chaospy as ch


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

def sobol_generator(m, n):
    """
    Generates m samples (each for one path) each having n numbers in Sobol sequence.
    
    INPUT:
        m (int): number of samples
        n (int): number of Sobol sequence numbers in each sample
        
    OUTPUT:
        (numpy.ndarray): A two-dimensional array of Sobol sequence numbers for conducting QMC simulation
    """
    sob_array = np.empty((0, n))
    for i in range(m // 39 + 1):
        sob = ch.create_sobol_samples(n, 39, i)
        sob_array = np.append(sob_array, sob, axis=0)
    return sob_array



def sobol_seq_sim2(no_of_paths, k, S_0, T, r, sigma, K, alpha, b):
    """
    Conducts QMC simulation using Sobol sequence.
    
    INPUT:
        no_of_paths (int): Number of samples in simulation
        k (int): Number of price step we aim to simulate in each path
        S_0 (float): Underlying asset price at time zero
        T (float): Time period of option contract
        r (float): Risk-netural interest rate
        sigma (float): Volatility in the environment
        K (float): Exercise price of the option
        
    OUTPUT:
        (Numpy.ndarray): A one-dimensional array of present value of simulated payoffs
    """
    dt = T / k
    present_payoffs = np.zeros(no_of_paths)
    epsilon_s = stats.norm.ppf(sobol_generator(no_of_paths, k))
    
    for k in range(no_of_paths):
        S = np.zeros(k)
        S[0] = S_0
        for i in range(1, k):
            S[i] = alpha * (b - S[i-1]) * dt + sigma * np.sqrt(S[i-1] * dt) * epsilon_s[k, i]
        present_payoffs[k] = pv_calc(payoff_calc(S, K), r, T)
    return present_payoffs


def halton_generator(m, n):
    """
    Generates m samples each having n numbers in Halton sequence.
    
    INPUT:
        m (int): The order of Halton sequence. Defines the number of samples
        n (int): Dimension of each sequence.Defines the number of Halton sequence numbers in each sample.
        
    OUTPUT:
        (numpy.ndarray): A two-dimensional array of Sobol sequence numbers for conducting QMC simulation
    """
    hal_seq = ch.create_halton_samples(m, n)
    return hal_seq

def halton_seq_sim(no_of_paths, n_steps, S_0, T, r, sigma, x_price):
    """
    Conducts QMC simulation using Halton sequence.
    
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
    dt = T / n_steps
    present_payoffs = np.zeros(no_of_paths)
    epsilon_h = stats.norm.ppf(halton_generator(no_of_paths, n_steps))
    
    for k in range(no_of_paths):
        price_steps = np.zeros(n_steps)
        price_steps[0] = S_0
        for i in range(1, n_steps):
            price_steps[i] = price_steps[i-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * epsilon_h[i, k] * np.sqrt(dt))
        present_payoffs[k] = pv_calc(payoff_calc(price_steps, x_price), r, T)
    return(present_payoffs)

