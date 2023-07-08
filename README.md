# "Option Pricing" project as part of the "Simulation and Monte Carlo methods" course 
## ENSAE 2022/2023

The aim of this project is to calculate the price of an Asian option between t=0 and t=1. To do this, we will use three methods: standard Monte Carlo, Quasi Monte Carlo and Multi-Level Monte Carlo.   

Implementation of the research article (https://artowen.su.domains/courses/362/readings/GilesMultilevel.pdf) for the Multi-LevelMonte Carlo method.

We will then evaluate which method is the best by calculating MSEs and CPU times.

We were awarded 16.5/20 for this project.

## Setup


The code can simply be run on Google Colab.

    
## Structure

- Fichier **data_preparation.R** :
  - *transform_csv* : fonction pour transformer le csv de Yahoo finance en dataframe avec les rendements et sans les variables qui ne nous intéresse pas
  - *plot_series_temp* : réalise les graphiques des prix, rendements et rendements au carré en fonction du temps
  - *autocorrelations* : graphiques autocorrélations des rendements et rendements au carré
  - *transform_csv_with_discount*(data,r) : transform_csv mais qui discount les prix au taux quotidient r
  
- Fichier **white_noise_laws.R** : On implémente des lois normalisées pour la loi des eta, pour qu'on puisse mettre autre chose que "rnorm(n)". Toutes les fonctions renvoient un vecteur de taille n.
  - runif_normalisee(n) : loi uniforme normalisée
  - rt_8_normalisee(n) : loi de Student à 8 degrés de liberté normalisée
  - normalised_student(n) : loi de Student à 5 degrés de liberté normalisée (le degré ici a vocation à être modifiée si nécessaire)

- Fichier **condition_stationnarite.R** :
  - *condition_stationnarite* : à partir de la loi des etas (bruit blanc), renvoie un dataframe de couples (alpha,beta) qui indiquent la condition de stationnarité
  - *superposition_3graphiques_condi_statio* : à partir de 3 dataframes (comme ceux de *condition_stationnarite*), renvoie le graphs des trois courbes de la condition de stationnarité

- **Main** is the notebook summarising our work. It uses the following related Python files :
    - For the standard Monte Carlo algorithm  : *ordinaryMC*
    - For the Quasi Monte Carlo algorithm : *QMC*
    - For the Multi-Level Monte Carlo algorithm : *MLMC*
    - For payoff calculations, present values : *calcul*
    - To calculate the price for different simulations : *calcul*
    - To compare the different methods : *comparaison*
    - Testing the Van Der Corput method (in progress) : *bonusVanDerCorput*

- **ordinaryMC** :
  - *CIR* : Function simulating the standard CIR.
  - *multiCIR* : Function that returns a CIR simulation table for all samples.
  - *ordinary_mc_sim* : Monte Carlo simulation to calculate option pricing.

- **QMC** :
  - *sobol_generator* : Generates m samples (each for one path) each having n numbers in Sobol sequence.
  - *sobol_generator_random* : Generates m samples (each for one path) each having n numbers in Sobol sequence.
  - *multiCIR_QMC* : Function that returns a QMC CIR simulation table for all samples.
  - *multiCIR_QMC_random* : Function that returns a random QMC CIR simulation table for all samples.
  - *QMC_mc_sim* : Conducts MC simulation.
  - *QMC_mc_sim_random* : Conducts random MC simulation.

- **MLMC** :
  - *CIR_ML* : Creation of the multi-level CIR.
  - *ML_principle* : This function takes only one point out of two between level l-1 and level L.
  - *multiCIR_ML* : Construction of the CIR for all samples for a fixed level.
  - *h_l* : Calculating h_l.
  - *N_l* : Calculation of N_l.
  - *level_mc_sim* : Simulation du multi level Monte Carlo pour calculer l'option pricing.
  - *ML_mc_sim* : Creation of the algorithm inspired by the Giles paper to return the N_l and the optimal L.
  - *sim_MLMC_Lfixe* : Multi-level Monte Carlo simulation for a fixed level.

- **Calcul** :
  - *payoff_calc* : This function calculates future payoff of the asian option based on arithmetic average of the price path.
  - *pv_calc* : Calculates present value of an amount of money in future.
  - *sim_iterator* : Iterates simulation with different sample sizes (form 10 to a maximum size with steps of 10).

- **comparaison** :
  - *CI_calc* : Calculates 95% confidence interval for the estimation of expected value of a random variable, given a sample.
  - *CI_length_calc* : Calculates length of a confidence interval.
  - *threshold_finder* : In an array of confidence intervals in the order of descending lengths, returns the index of fist interval shorter than a threshold.
  - *CI_length_calc* : Calculates length of a confidence interval.
  - *CPU* : Calculate CPU Times
  - *mse_time* : Calculate MSE Times
  - *mse_comparaison* : Compare different MSE

- **bonusVanDerCorput** :
  - *van_der_corput_base_2* : This function generates a van der Corput sequence of length n with the given base.


## Authors

- [@Thomasaujoux](https://github.com/Thomasaujoux)
- [@Suzie14](https://github.com/Suzie14)
- [@elenalmg](https://github.com/elenalmg)