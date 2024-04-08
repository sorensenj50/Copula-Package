import numpy as np
from .bivariate_copula import BivariateCopula

from type_definitions import Vectorizable


class Independent(BivariateCopula):
    # independent copula

    def __init__(self):
        super().__init__(model_name = "Independent", family_name = np.nan, 
                         initial_param_guess = [],
                         param_bounds = [], param_names = [], 
                         params = [])
        
    def _logpdf(self, u1: Vectorizable, u2: Vectorizable) -> Vectorizable:
        return np.zeros_like(u1)
    

    def _cdf(self, u1: Vectorizable, u2: Vectorizable) -> Vectorizable:
        return u1 * u2
    

    def _params_to_tau(self, *params: float) -> float:
        return 0
    

    def _params_to_rho(self, *params: float) -> float:
        return 0
    

    def _conditional_ppf(self, u1: Vectorizable, q: Vectorizable) -> Vectorizable:
        return q
    

    def _conditional_cdf(self, u1: Vectorizable, u2: Vectorizable) -> Vectorizable:
        return u2