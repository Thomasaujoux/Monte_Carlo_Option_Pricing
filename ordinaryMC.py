import numpy as np

import calcul



def CIR(alpha, b, sigma, T, k, S_0):
    '''
    Fonction qui simule le CIR classique
    '''

    dt = T/k
    S = np.zeros(k+1)
    S[0] = S_0
    for i in range(1, k+1):
        dS = alpha * (b - S[i-1]) * dt + sigma * np.sqrt(S[i-1] ) * np.random.normal(scale=np.sqrt(dt))
        S[i] = S[i-1] + dS
        if S[i] < 0 : 
            S[i] = "Erreur"
    return S



def multiCIR(alpha, b, sigma, T, k, S_0, nb_samples): 
    '''
    Fonction qui renvoie un tableau de simulation CIR pour tous les échantillons
    '''
    multiCIR = []
    for i in range(nb_samples): 
        multiCIR.append(CIR(alpha, b, sigma, T, k, S_0))
    return multiCIR




def ordinary_mc_sim(nb_samples, k, S_0, T, r, sigma, K, alpha, b):
    """
    Simulation du Monte Carlo pour calculer l'option pricing
    
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
    multiCIR2 = multiCIR(alpha, b, sigma, T, k, S_0, nb_samples)
    
    for i in range(nb_samples):
        present_payoffs[i] = calcul.pv_calc(calcul.payoff_calc(multiCIR2[i], K), r, T)
    return(present_payoffs)


