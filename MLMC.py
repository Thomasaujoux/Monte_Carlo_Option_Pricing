import CIR
import ordinaryMC
import numpy as np


def h_l(k,T,l):
    """
    Calcul  de h_l
    
    k: number of simulations
    T (float): Time period of option contract
    l (int): indice of level
    
    output : h_l (float)
    """
    h_l = 4**(-l)*T
    return h_l

def N_l(variance, k, T,l, L, epsilon):
    
    
    
    sum_vh = 0 
    for l in range(0,L):
        sum_vh = sum_vh + np.sqrt(variance[l]/h_l(k,T,l))
    print(sum_vh)
    print(2*epsilon**(-2)*np.sqrt(variance[l]*h_l(k,T,l))*sum_vh)
    N_l = int(np.ceil(2*epsilon**(-2)*np.sqrt(variance[l]*h_l(k,T,l))*sum_vh))
                      
    return N_l

def level_mc_sim(nb_samples, S_0, T, r, sigma, K, alpha, b, L):
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
    multiCIR = CIR.multiCIR_ML(alpha, b, sigma, L, T, S_0, nb_samples)
    
    for i in range(nb_samples):
        present_payoffs[i] = ordinaryMC.pv_calc(ordinaryMC.payoff_calc(multiCIR[i], K), r, T)
    return(np.mean(present_payoffs))

def sim_MLMC(k, S_0, T, r, sigma, K, alpha, b):
    
    #1) start with L=0
    L = 0 
    N = []
    epsilon = np.exp(-1) #fixé de cette façon dans le papier
    convergence = False
    
    while convergence==False or L < 2:
        # Initialize arrays for sample means and variances
        means = np.zeros(L+1)
        variances = np.zeros(L+1)
        for l in range(L+1):
            
            N.append(100)
            multiCIR_ML = CIR.multiCIR_ML(alpha, b, sigma, L, T, S_0, N[l])
            values = np.array([mc_telescopic_sum(alpha, b, sigma, L, T, S_0, N[l], r, K) for sample in multiCIR_ML])
        
            # Compute the sample mean and variance at each level
            means[l] = np.mean(values)
            variances[l] = np.var(values) #2) estimate VL using an initial N_L = 10**4 samples
            print(variances)
            
            #3) define optimal N_l, l = 0,...,L using Eqn. (12)
            New = N_l(variances, k, T, l, L, epsilon)
            
        #4)evaluate extra samples at each level as needed for new N_l
            if New > N[l]:
                values = np.concatenate(([mc_telescopic_sum(alpha, b, sigma, L, T, S_0, N[l], r, K) for sample in multiCIR_ML],[mc_telescopic_sum(alpha, b, sigma, L, T, S_0, New - N[l], r, K) for sample in Bis_multiCIR_ML]))
                N[l] = New
                
                
            if np.abs(means[L] - k**(-1)*means[L-1]) < 1/np.sqrt(2)*((k**2)-1)**epsilon:
                convergence=True
                
            else :
                L = L+1
                print(L)
           
    
    return values, L, N


def sim_MLMC_Lfixe( S_0, T, r, sigma, K, alpha, b, N,L):
    means = np.zeros(L)
    variances = np.zeros(L)
    list_payoff=[]
    for l in range(L):
        multicir=CIR.multiCIR_ML(alpha, b, sigma, l, T, S_0, N)
        list_payoff_level=[]
        payoff_for_var=[]
        
        for i in range(N):
            a=ordinaryMC.pv_calc(ordinaryMC.payoff_calc(multicir[i][0][1],K),r,T)
            b=ordinaryMC.pv_calc(ordinaryMC.payoff_calc(multicir[i][1][1],K),r,T)
            payoff_for_var.append(a)
            list_payoff_level.append(a-b)
        
        list_payoff.append(list_payoff_level)
        means[l] = np.mean(list_payoff_level)
        variances[l]=np.var(payoff_for_var)
    return means,variances