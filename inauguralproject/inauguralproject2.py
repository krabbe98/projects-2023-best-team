from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        #par.nu = 0.001
        par.nu_M = 0.01
        par.nu_F = 0.001
        #par.epsilon = 1.0
        par.epsilon_M = 0.5
        par.epsilon_F = 1.5
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if np.allclose(par.sigma, 0): # if par.sigma==0: can't compare float with equality
            H=np.min([HM,HF])
        elif np.allclose(par.sigma, 1): # elif par.sigma==1: can't compare float with equality
            H = HM**(1-par.alpha) * HF**par.alpha
        else:
            H = ((1 - par.alpha) * HM**((par.sigma - 1) / par.sigma) + par.alpha * HF**((par.sigma - 1) / par.sigma))**(par.sigma / (par.sigma - 1))

        
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_M = 1+1/par.epsilon_M
        epsilon_F = 1+1/par.epsilon_F
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu_M*(TM**epsilon_M/epsilon_M)+par.nu_F*(TF**epsilon_F/epsilon_F)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    

 #Continuous:   
    def solve(self):
              
        def obj(x):
            u = self.calc_utility(x[0], x[1], x[2], x[3])
            return - u
    
        bounds = [(0, 24)]*4
        guess = [6.0]*4
# call the numerical minimizer
        solution = optimize.minimize(obj, x0 = guess, bounds=bounds) #options={'xatol': 1e-4})

       
        return solution.x

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol
        WF_list = [0.8, 0.9, 1., 1.1, 1.2]
        for it, alpha in enumerate(WF_list):
            par.wF = alpha
            out = self.solve()
            sol.LM_vec[it] = out[0]
            sol.HM_vec[it] = out[1]      
            sol.LF_vec[it] = out[2]
            sol.HF_vec[it] = out[3]      
        
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        def obj(x):
            par = self.par
            sol = self.sol
            par.nu_M = x[0]
            par.nu_F = x[1]            
            par.sigma = x[2]
            par.epsilon_M = x[3]
            par.epsilon_F = x[4]            
            self.solve_wF_vec()
            self.run_regression()
            fun = (par.beta0_target - sol.beta0)**2.0 + (par.beta1_target - sol.beta1)**2.0

            return fun

        bounds = [(0., 24.), (0., 24.),  (0., 24.),  (0., 24.) (0., 24.)] #[(min_alpha, max_alpha), (min_sigma, max_sigma)]
        guess = [.001, .001, 1, 1, 1] #[alpha, sigma]
        solution = optimize.minimize(obj, x0 = guess, bounds=bounds, method = "nelder-mead") #options={'xatol': 1e-4})

        return solution.x

        pass    

        

