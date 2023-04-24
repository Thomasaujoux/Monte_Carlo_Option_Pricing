from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np

import calcul



def CIR_ML(alpha, b, sigma, L, T, S_0): 
    '''
    Création du CIR multi level
    '''

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
    '''
    Cette fonction permet de ne prendre qu'un point sur deux entre le level l-1 et le level L
    '''

    ML_CIR = []
    M = CIR_ML(alpha, b, sigma, L, T, S_0)
    ML_CIR.append(M)
    A = ([ML_CIR[0][0][i] for i in range(0,len(ML_CIR[0][0]),2)],[ML_CIR[0][1][i] for i in range(0,len(ML_CIR[0][0]),2)])
    ML_CIR.append(A)
           
    return ML_CIR



def multiCIR_ML(alpha, b, sigma, L, T, S_0, nb_samples): 

    '''
    Construction du CIR pour tous les échantillons pour un level fixé
    '''

    multiCIR = []
    
    if L==0 :
        for i in range(nb_samples): 
            multiCIR.append(CIR_ML(alpha, b, sigma, L, T, S_0))

    else :
        for i in range(nb_samples): 
            multiCIR.append(ML_principle(alpha, b, sigma, L, T, S_0))
    
    return multiCIR



def h_l(T,l):
    """
    Calcul  de h_l
    
    k: number of simulations
    T (float): Time period of optsion contract
    l (int): indice of level
    
    output : h_l (float)
    """
    h_l = 2**(-l)*T
    return h_l



def N_l(variance, k, T,l, L, epsilon):

    '''
    Calcul de N_l

    '''
    
    sum_vh = 0 
    for level in range(0,L+1):
        sum_vh = sum_vh + np.sqrt(variance[level]/h_l(T,l))
    
    N_l = int(np.ceil(2*epsilon**(-2)*np.sqrt(variance[l]*h_l(T,l))*sum_vh))
                      
    return N_l



def level_mc_sim(nb_samples, S_0, T, r, sigma, K, alpha, b, L):
    """
    Simulation du multi level Monte Carlo pour calculer l'option pricing
    
    INPUT:
        nb_samples (int): Nombre d'échantillon
        k (int): Nombre de price step 
        S_0 (float): Asset price au temps 0
        T (float): option contract
        r (float): Risk-netural interest rate
        sigma (float): Volatility in the environment
        K (float): Exercise price of the option
        alpha (float): taux de convergence
        b (float): taux de convergence
        
    OUTPUT:
        (Numpy.ndarray): tableau des present values des payoffs simulés
    """
    present_payoffs = np.zeros(nb_samples)
    multiCIR = multiCIR_ML(alpha, b, sigma, L, T, S_0, nb_samples)
    
    for i in range(nb_samples):
        present_payoffs[i] = calcul.pv_calc(calcul.payoff_calc(multiCIR[i], K), r, T)
    return(np.mean(present_payoffs))



def ML_mc_sim(k, S_0, T, r, sigma, K, alpha, b):

    '''
    Création de l'algorithme inspiré du papier Giles pour retourner les N_l et le L optimal

    OUTPUT:
        y_chap : estimateur optimal de l'option pricing
        L : nombre de level optimal
        N : tableau des échantillons optimaux pour chaque niveau
        variances : variances à chaqeu level
        mean : moyenne à chaque level

    '''
    
    #1) start with L=0
    L = 0
    N = []
    epsilon = 10**(-3) #fixé de cette façon dans le papier
    convergence = False
    y_chap=0
    values=[]
    # Initialize arrays for sample means and variances
    
    while convergence==False or L < 2:

        N.append(100)
            
        # Compute the sample mean and variance at each level
        values.append(sim_MLMC_Lfixe( S_0, T, r, sigma, K, alpha, b, N[L],L))
        means = np.zeros(L+1)
        variances = np.zeros(L+1)
        
        for l in range(L+1):
            
            means[l]=np.mean(values[l])
            variances[l] = np.var(values[l]) #2) estimate VL using an initial N_L = 10**4 samples
            
            
            #3) define optimal N_l, l = 0,...,L using Eqn. (12)
            New = N_l(variances, k, T, l, L, epsilon)
      
        #4)evaluate extra samples at each level as needed for new N_l
            if New > N[l]:
                values[l] = values[l]+sim_MLMC_Lfixe( S_0, T, r, sigma, K, alpha, b, New - N[l],l)
                N[l] = New

        if L>=2:
            if np.abs(means[L] - 2**(-1)*means[L-1]) < 1/np.sqrt(2)*((2**2)-1)*epsilon:
                convergence=True
                
            else:
                L = L+1
                
        else: 
            L = L+1
          
    y_chap = np.sum(means)
    return  y_chap, L, N, variances, means



def sim_MLMC_Lfixe( S_0, T, r, sigma, K, alpha, b, N,L):
    
    '''
    Simulation du multi level Monte Carlo pour un level fixé
    '''


    if L==0 :
        payoff_level=[]
        multicir=multiCIR_ML(alpha, b, sigma, L, T, S_0, N)
        for i in range(N):
            a=calcul.pv_calc(calcul.payoff_calc(multicir[i][1],K),r,T)
            payoff_level.append(a)
       
    else:
          
        multicir=multiCIR_ML(alpha, b, sigma, L, T, S_0, N)

        payoff_level=[]
            
            
        for i in range(N):
            a=calcul.pv_calc(calcul.payoff_calc(multicir[i][0][1],K),r,T)
            b=calcul.pv_calc(calcul.payoff_calc(multicir[i][1][1],K),r,T)
            payoff_level.append(a-b)

    return payoff_level

               
