import numpy as np
from .bivariate_copula import BivariateCopula


class Independent(BivariateCopula):
    # independent copula

    def __init__(self):
        super().__init__(model_name = "Independent", family_name = np.nan, 
                         initial_param_guess = [],
                         param_bounds = [], param_names = [], 
                         params = [])
        
    def _logpdf(self, u1, u2):
        return np.zeros_like(u1)
    

    def _cdf(self, u1, u2):
        return u1 * u2
    

    def _params_to_tau(self, *params):
        return 0
    

    def _params_to_rho(self, *params):
        return 0
    

    def _conditional_ppf(self, u1, q):
        return q
    

    def _conditional_cdf(self, u1, u2):
        return u2