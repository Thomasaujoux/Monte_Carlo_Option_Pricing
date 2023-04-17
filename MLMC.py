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
    h_l = k**(-l)*T
    return h_l

def N_l(variance, k, T,l, L, epsilon):
    
    
    
    sum_vh = 0 
    for l in range(0,L):
        sum_vh = sum_vh + np.sqrt(variance[l]/h_l(k,T,l))
    print(sum_vh)
    print(2*epsilon**(-2)*np.sqrt(variance[l]*h_l(k,T,l))*sum_vh)
    N_l = int(np.ceil(2*epsilon**(-2)*np.sqrt(variance[l]*h_l(k,T,l))*sum_vh))
                      
    return N_l

def mc_telescopic_sum(L,multiCIR_ML, K, r, T, S_0,sigma,alpha, b): 
    pv_calc0 = ordinary_mc_sim(nb_samples, int(np.ceil(T), S_0, T, r, sigma, K, alpha, b)
    pv_calc_sum= 0
    for l in range(1,L):
        pv_calc_sum = pv_calc + ordinary_mc_sim(nb_samples, int(np.ceil(T/2**(-l), S_0, T, r, sigma, K, alpha, b)

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
            
            N.append(10**4)
            multiCIR_ML = CIR.multiCIR_ML(alpha, b, sigma, T, k, S_0, N[l], L+1)
            values = np.array([mc_telescopic_sum(ordinaryMC.pv_calc(ordinaryMC.payoff_calc(sample, K), r, T)) for sample in multiCIR_ML])
        
            # Compute the sample mean and variance at each level
            means[l] = np.mean(values)
            variances[l] = np.var(values) #2) estimate VL using an initial N_L = 10**4 samples
            
            #3) define optimal N_l, l = 0,...,L using Eqn. (12)
            New = N_l(variances, k, T, l, L, epsilon)
            
        #4)evaluate extra samples at each level as needed for new N_l
            if New > N[l]:
                Bis_multiCIR_ML = CIR.multiCIR_ML(alpha, b, sigma, T, k, S_0, New-N[l], L)
                values = np.concatenate((multiCIR_ML,Bis_multiCIR_ML))
                N[l] = New
                
                
            if np.abs(means[L] - k**(-1)*means[L-1]) < 1/np.sqrt(2)*((k**2)-1)**epsilon:
                convergence=True
                
            else :
                L = L+1
           
    
    return values, L, N
