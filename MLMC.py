import CIR
import ordinaryMC
import numpy as np

def MLMC(L, N, M, alpha, beta, gamma, f):
    """
    Implements the multi-level Monte Carlo algorithm for function f.
    
    L: number of levels
    N: number of samples at each level
    M: number of simulations
    alpha: convergence rate of weak error
    beta: convergence rate of variance
    gamma: bias reduction factor
    f: function to be estimated
    """
    
    # Compute the values of delta and epsilon for each level
    delta = np.sqrt(beta / (2 * L))
    eps = np.sqrt(alpha / (2 * L))
    
    # Initialize arrays for sample means and variances
    means = np.zeros(L)
    variances = np.zeros(L)
    
    for l in range(L):
        # Compute the number of samples needed at each level
        n_l = int(np.ceil((delta / eps) ** 2 * (2 ** l)))
        
        # Perform M simulations at each level with N samples
        samples = np.zeros((M, n_l))
        for m in range(M):
            samples[m] = np.random.normal(size=n_l)
        values = np.array([f(sample) for sample in samples])
        
        # Compute the sample mean and variance at each level
        means[l] = np.mean(values)
        variances[l] = np.var(values)
    
    # Compute the MLMC estimate
    summands = np.array([(means[l] - means[l-1]) / (eps * np.sqrt(variances[l-1])) for l in range(1, L)])
    mlmc_est = means[0] + gamma * np.sum(summands)
    
    return mlmc_est

def optimalL(k, S_0, T, r, sigma, K, alpha, b):
    L = 0 
    N_L = 10**4
    V_L = 

def MLMC(nb_samples, k, S_0, T, r, sigma, K, alpha, b, L):
    """
    Conducts MLMC method,
    
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
    multiCIR_ML = CIR.multiCIR_ML(alpha, b, sigma, T, k, S_0, nb_samples, L)
    
    for i in range(nb_samples):
        present_payoffs[i] = ordinaryMC.pv_calc(ordinaryMC.payoff_calc(multiCIR[i], K), r, T)
    return(present_payoffs)



def ml_mc_sim(nb_samples, k, S_0, T, r, sigma, K, alpha, b):
    L = 0
    sim = []
    while 'condition de convergence pas respectée'
        L = L+1
        sim[l] = MLMC(nb_samples, k, S_0, T, r, sigma, K, alpha, b, L)
        values = np.array([present_payoffs for sample in samples])
    
    
