import numpy as np
from datetime import datetime

import utils
import base



class Marginal(base.Base):
    def __init__(self, rv_obj, *args, **kwargs):

        self.rv_obj = rv_obj
        self.summary_title = "Marginal Distribution"
        super().__init__(*args, **kwargs)


    def _handle_input(self, x_or_u, is_x = True, adj = 1e-4):
        if not (utils.is_arraylike(x_or_u) or utils.is_number(x_or_u)):
            raise SyntaxError

        if is_x:
            return x_or_u
        
        return utils.clip_u_input(x_or_u, adj = adj)
    

    def cdf(self, x):
        # input validation
        return self._cdf(self._handle_input(x), *self.params)
    

    def _cdf(self, x, *params):
        return self.rv_obj.cdf(x, *params)
    

    def pdf(self, x):
        # error handling
        return self._pdf(self._handle_input(x), *self.params) 
    

    def _pdf(self, x, *params):
        return self.rv_obj.pdf(x, *params)


    def ppf(self, q, adj = 1e-4):
        # input validation
        return self._ppf(self._handle_input(q, is_x = False, adj = adj), *self.params)


    def _ppf(self, u, *params):
        return self.rv_obj.ppf(u, *params)
    

    def logpdf(self, x):
        # input validation
        return self._logpdf(self._handle_input(x), *self.params)
    

    def _logpdf(self, x, *params):
        return self.rv_obj.logpdf(x, *params)

    
    def _log_likelihood(self, x, *params):
        # this will be called by the joint probability disitribution when getting total LL
        # not needed for fitting
        return np.sum(self._logpdf(x, *params))
    

    def fit(self, x, robust_cov = True):
        # input validation
        valid_x = self._handle_input(x)

        # relying on scipy implementation of fit
        opt_params = self.rv_obj.fit(x)
        self._post_process_fit(valid_x, np.array([*opt_params]), 
                               self._get_obj_func(valid_x), robust_cov = robust_cov)

    
    def simulate(self, n = 1000, seed = None):
        # rely on Scipy
        rng = np.random.default_rng(seed = seed)
        u = rng.uniform(size = n)
        return self.ppf(u)
    

    def _get_top_summary_table(self):
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
    def skewness(self):
        return self._params_to_skewness(*self.params)
    

    @property
    def kurtosis(self):
        return self._params_to_kurtosis(*self.params)
    

    @property
    def var(self):
        return self._params_to_var(*self.params)
    

    @property
    def cvar(self):
        return self._params_to_cvar(*self.params)
    
    
    def _params_to_skewness(self, *params):
        raise NotImplementedError
    
    
    def _params_to_kurtosis(self, *params):
        raise NotImplementedError
    
    
    def _params_to_var(self, *params, alpha = 0.95):
        # simply accessing the defined PPF of the distribution
        return self._ppf(1 - alpha, *params)
    

    def _params_to_cvar(self, *params, alpha = 0.95):
        raise NotImplementedError