# Projet "Option Pricing" dans le cadre du cours "Simulation and Monte Carlo methods" 
## ENSAE 2022/2023

Le but de ce projet est de calculer le prix d'une option asiatique entre t=0 et t=1. Pour cela, nous allons utiliser trois méthodes : Monte Carlo standard, Quasi Monte Carlo et Multi-Level Monte Carlo.   

Nous allons ensuite évaluer quelle est la meilleure méthode en calculant les MSE et les CPU time.

Structure des fichiers : 
- *Main* est le notebook résumant notre travail. Il fait appel aux fichiers Python annexes suivants: 
    - Pour l'algorithme de Monte Carlo standard  : *ordinaryMC*
    - Pour l'algorithme de quasi Monte Carlo : *QMC*
    - Pour l'algorithme de Multi Level Monte Carlo : *MLMC*
    - Pour les calculs de payoff, present values : *calcul*
    - Pour le calcul du prix pour différentes simulations : *calcul*
  

