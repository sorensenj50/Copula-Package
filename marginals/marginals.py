import numpy as np
from datetime import datetime
from scipy.stats import rv_continuous

import utils
import base

from typing import Union, Tuple
from type_definitions import Vectorizable



class Marginal(base.Base):
    def __init__(self, rv_obj: Union[rv_continuous, None], *args, **kwargs):

        self.rv_obj = rv_obj
        self.summary_title = "Marginal Distribution"
        super().__init__(*args, **kwargs)


    def _handle_input(self, x_or_u: Vectorizable, is_x: bool = True, adj: float = 1e-4) -> Vectorizable:
        if not (utils.is_arraylike(x_or_u) or utils.is_number(x_or_u)):
            raise SyntaxError

        if is_x:
            return x_or_u
        
        return utils.clip_u_input(x_or_u, adj = adj)
    

    def cdf(self, x: Vectorizable) -> Vectorizable:
        # input validation
        return self._cdf(self._handle_input(x), *self.params)
    

    def _cdf(self, x: Vectorizable, *params: float) -> Vectorizable:
        return self.rv_obj.cdf(x, *params)
    

    def pdf(self, x: Vectorizable) -> Vectorizable:
        # error handling
        return self._pdf(self._handle_input(x), *self.params) 
    

    def _pdf(self, x: Vectorizable, *params: float) -> Vectorizable:
        return self.rv_obj.pdf(x, *params)


    def ppf(self, q: Vectorizable, adj: float = 1e-6) -> Vectorizable:
        # input validation
        return self._ppf(self._handle_input(q, is_x = False, adj = adj), *self.params)


    def _ppf(self, q: Vectorizable, *params: float) -> Vectorizable:
        return self.rv_obj.ppf(q, *params)
    

    def logpdf(self, x: Vectorizable) -> Vectorizable:
        # input validation
        return self._logpdf(self._handle_input(x), *self.params)
    

    def _logpdf(self, x: Vectorizable, *params: float) -> Vectorizable:
        return self.rv_obj.logpdf(x, *params)

    
    def _log_likelihood(self, x: Vectorizable, *params: float) -> float:
        # this will be called by the joint probability disitribution when getting total LL
        # not needed for fitting
        return np.sum(self._logpdf(x, *params))
    

    def fit(self, x, robust_cov = True) -> None:
        # input validation
        valid_x = self._handle_input(x)

        # relying on scipy implementation of fit
        opt_params = self.rv_obj.fit(x)
        self._post_process_fit(valid_x, np.array([*opt_params]), 
                               self._get_obj_func(valid_x), robust_cov = robust_cov)

    
    def simulate(self, n: int = 1000, seed: Union[int, None] = None, adj: float = 1e-6) -> np.ndarray:
        # rely on Scipy
        rng = np.random.default_rng(seed = seed)
        u = rng.uniform(size = n)
        return self.ppf(u, adj = 1e-6)
    

    def _get_top_summary_table(self) -> Tuple[list, list]:
        now = datetime.now()
        top_left = [
            ("Model Name:", self.model_name), ("Model Family:", self.family_name),
            ("Esimation Method:", self._get_estimation_method()),("Num. Params:", self.k), ("Num. Obs:", self.n),
            ("Date:", now.strftime("%a, %b %d %Y")),("Time:", now.strftime("%H:%M:%S")), ("", ""), ("", ""),
        ]

        top_right = [
            ("Log-Likelihood:", utils.format_func(self.LL, 10)), ("AIC:", utils.format_func(self.aic, 10)),
            ("BIC:", utils.format_func(self.bic, 10)), ("Skewness:", utils.format_func(self.skewness, 10)), 
            ("Excess Kurtosis:", utils.format_func(self.kurtosis, 10)), ("95% VaR:", utils.format_func(self.var, 10)),
            ("95% CVaR:", utils.format_func(self.cvar, 10)), ("", ""), ("", ""),
        ]

        return top_left, top_right
    
    @property
    def skewness(self) -> float:
        return self._params_to_skewness(*self.params)
    

    @property
    def kurtosis(self) -> float:
        return self._params_to_kurtosis(*self.params)
    

    @property
    def var(self) -> float:
        return self._params_to_var(*self.params)
    

    @property
    def cvar(self) -> float:
        return self._params_to_cvar(*self.params)
    
    
    def _params_to_skewness(self, *params: float) -> float:
        raise NotImplementedError
    
    
    def _params_to_kurtosis(self, *params: float) -> float:
        raise NotImplementedError
    
    
    def _params_to_var(self, *params: float, alpha: float = 0.95) -> float:
        # simply accessing the defined PPF of the distribution
        return self._ppf(1 - alpha, *params)
    

    def _params_to_cvar(self, *params: float, alpha: float = 0.95) -> float:
        raise NotImplementedError