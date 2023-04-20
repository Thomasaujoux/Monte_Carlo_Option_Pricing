import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import CIR
import ordinaryMC
import comparaison



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
    multiCIR_QMC = CIR.multiCIR_QMC(alpha, b, sigma, T, k, S_0, nb_samples)
    
    for i in range(nb_samples):
        present_payoffs[i] = ordinaryMC.pv_calc(ordinaryMC.payoff_calc(multiCIR_QMC[i], K), r, T)
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
    multiCIR_QMC = CIR.multiCIR_QMC_random(alpha, b, sigma, T, k, S_0, nb_samples)
    
    for i in range(nb_samples):
        present_payoffs[i] = ordinaryMC.pv_calc(ordinaryMC.payoff_calc(multiCIR_QMC[i], K), r, T)
    return(present_payoffs)


def sim_iterator_QMC(max_sample, k, S_0, T, r, sigma, K, alpha, b):
    """
    Iterates simulation with different sample sizes (form 10 to a maximum size with steps of 10)
    
    INPUT:
        max_sample (int): Maximum sample size for the iteration of simulations
        k (int): Number of price step we aim to simulate in each path
        S_0 (float): Underlying asset price at time zero
        T (float): Time period of option contract
        r (float): Risk-netural interest rate
        sigma (float): Volatility in the environment
        x_price (float): Exercise price of the option
        K (float): Exercise price of the option
        alpha (float): taux de convergence
        b (float): taux de convergence
    
    OUTPUT:
        (numpy.ndarray): confidence intervals of the simulations #pas encore intégré
        (numpy.ndarray): price estimations of the simulations
    """
    
    mean_pv_payoffs = np.zeros(int(max_sample / 10))
    confidence_intervals = np.array([None, None])


    for nb_samples in range(10, max_sample + 1, 10):
        present_payoffs = QMC_mc_sim(nb_samples,k, S_0, T, r, sigma, K, alpha, b)
        mean_pv_payoffs[int(nb_samples/10) - 1] = np.mean(present_payoffs)
        confidence_intervals = np.row_stack((confidence_intervals, comparaison.CI_calc(present_payoffs)))

    return(mean_pv_payoffs, confidence_intervals)


def sim_iterator_QMC_random(max_sample, k, S_0, T, r, sigma, K, alpha, b):
    """
    Iterates simulation with different sample sizes (form 10 to a maximum size with steps of 10)
    
    INPUT:
        max_sample (int): Maximum sample size for the iteration of simulations
        k (int): Number of price step we aim to simulate in each path
        S_0 (float): Underlying asset price at time zero
        T (float): Time period of option contract
        r (float): Risk-netural interest rate
        sigma (float): Volatility in the environment
        x_price (float): Exercise price of the option
        K (float): Exercise price of the option
        alpha (float): taux de convergence
        b (float): taux de convergence
    
    OUTPUT:
        (numpy.ndarray): confidence intervals of the simulations #pas encore intégré
        (numpy.ndarray): price estimations of the simulations
    """
    
    mean_pv_payoffs = np.zeros(int(max_sample / 10))
    confidence_intervals = np.array([None, None])


    for nb_samples in range(10, max_sample + 1, 10):
        present_payoffs = QMC_mc_sim_random(nb_samples,k, S_0, T, r, sigma, K, alpha, b)
        mean_pv_payoffs[int(nb_samples/10) - 1] = np.mean(present_payoffs)
        confidence_intervals = np.row_stack((confidence_intervals, comparaison.CI_calc(present_payoffs)))

    return(mean_pv_payoffs, confidence_intervals)