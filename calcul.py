import numpy as np

import ordinaryMC
import QMC
import MLMC
import comparaison



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


def pv_calc(FV, r, T):
    """
    Calculates present value of an amount of money in future.
    
    INPUT:
        FV (float): Future value of money
        r (float): Risk neutral interest rate
        T (float): Period of time
    
    OUTPUT:
        (float): Present value of FV
    """
    
    return FV * np.exp(-r * T)


def sim_iterator(max_sample, k, S_0, T, r, sigma, K, alpha, b, *,method):
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
    
    assert(method in ['ordinary', 'QMC', 'QMC_random','MLMC'])

    mean_pv_payoffs = np.zeros(int(max_sample / 10))
    confidence_intervals = np.array([None, None])

    if method == 'ordinary':
        for nb_samples in range(10, max_sample + 1, 10):
            present_payoffs = ordinaryMC.ordinary_mc_sim(nb_samples,k, S_0, T, r, sigma, K, alpha, b)
            mean_pv_payoffs[int(nb_samples/10 - 1)] = np.mean(present_payoffs)
            confidence_intervals = np.row_stack((confidence_intervals, comparaison.CI_calc(present_payoffs)))

    elif method == 'QMC':
        for nb_samples in range(10, max_sample + 1, 10):
            present_payoffs = QMC.QMC_mc_sim(nb_samples, k, S_0, T, r, sigma, K, alpha, b)
            mean_pv_payoffs[int(nb_samples/10 - 1)] = np.mean(present_payoffs)
            confidence_intervals = np.row_stack((confidence_intervals, comparaison.CI_calc(present_payoffs)))

    elif method == 'QMC_random':
        for nb_samples in range(10, max_sample + 1, 10):
            present_payoffs = QMC.QMC_mc_sim_random(nb_samples, k, S_0, T, r, sigma, K, alpha, b)
            mean_pv_payoffs[int(nb_samples/10 - 1)] = np.mean(present_payoffs)
            confidence_intervals = np.row_stack((confidence_intervals, comparaison.CI_calc(present_payoffs)))


    return(mean_pv_payoffs, confidence_intervals)
