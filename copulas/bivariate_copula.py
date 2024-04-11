import utils
import base

import numpy as np
from datetime import datetime

from typing import Union, Callable, Tuple
from type_definitions import Vectorizable


class BivariateCopula(base.Base):
    def __init__(self, *args, **kwargs):
        self.summary_title = "Bivariate Copula"
        self.estimation_method_str = "CMLE"
        super().__init__(*args, **kwargs)
    

    def _handle_u_input(self, u: Vectorizable, adj: float = 1e-5):
        if not (utils.is_arraylike(u) or utils.is_number(u)):
            # must be number or array

            raise TypeError

        return utils.clip_u_input(u, adj)


    def _handle_uu_input(self, u1: Vectorizable, u2: Vectorizable, adj: float) -> Vectorizable:
        # fix this logic

        # u1 and u2 are both valid individually
        valid_u1 = self._handle_u_input(u1, adj)
        valid_u2 = self._handle_u_input(u2, adj)

        if utils.is_number(valid_u1) and utils.is_number(valid_u2):
            return valid_u1, valid_u2
        elif utils.is_arraylike(valid_u1) and utils.is_arraylike(valid_u2):
            if valid_u1.shape == valid_u2.shape:
                return valid_u1, valid_u2
            
            # must have the same shape
            raise TypeError

        # must have the same type
        raise TypeError
    
    
    def _reshape_wrapper(self, u1: Vectorizable, u2: Vectorizable, f: Callable, *params: float) -> Vectorizable:
        # if u1 and u2 are arrays, ensures flat input and reshapes after applying function
        # because of earlier validation, if 1 is array, then both are

        if utils.is_arraylike(u1):
            out_shape = u1.shape
            return f(u1.flatten(), u2.flatten(), *params).reshape(out_shape)

        return f(u1, u2, *params)
    

    @property
    def tau(self) -> float:
        return self._params_to_tau(*self.params)


    def _tau_to_params(self, tau: float) -> Tuple[float, ...]:
        # return a tuple
        return (np.nan,)
    

    def _params_to_tau(self, *params: float) -> float:
        # Default is brute force numerical integration
        # using the formulation from Joe 2.42
        # this assumes copula is symmetric (i.e., that u1 and u2 can be swapped to the same conditional_cdf)
        
        n = 200
        dudv = 1 / (n ** 2)
        u1, u2 = utils.get_u_grid(range_num = n, adj = 1e-6)

        return 1 - 4 * np.sum(self._conditional_cdf(u1, u2, *params) * self._conditional_cdf(u2, u1, *params) * dudv)

    
    @property
    def rho(self) -> float:
        return self._params_to_rho(*self.params)


    def _params_to_rho(self, *params: float) -> float:
        # Default is brute force numerical integration
        # using the formulation from Joe 2.47

        num = 200
        dudv = 1 / (num ** 2)
        u1, u2 = utils.get_u_grid(range_num = num, adj = 1e-6)
        return 12 * np.sum(self._cdf(u1.flatten(), u2.flatten(), *params) * dudv) - 3
    

    def _log_likelihood(self, u1: Vectorizable, u2: Vectorizable, *params: float) -> float:
        return np.sum(self._logpdf(u1, u2, *params))
    
    
    def fit(self, u1: Vectorizable, u2: Vectorizable, method: str = "MLE", optimizer: str = "Powell", 
            initial_param_guesses: Union[list, None] = None, robust_cov: bool = True, adj: float = 1e-5) -> None:

        # input validation
        u1_valid, u2_valid = self._handle_uu_input(u1, u2, adj = adj)
        

        # check length of initial param guesses
        if (initial_param_guesses is not None) and (len(initial_param_guesses) != len(self.params)):
            raise SyntaxError
    
        initial_guess = initial_param_guesses if initial_param_guesses is not None else self.initial_param_guess
        objective_func = self._get_obj_func(u1_valid, u2_valid)

        opt_results = self._fit(objective_func, initial_guess, self.param_bounds, optimizer = optimizer)

        self._post_process_fit(utils.flatten_concatenate(u1_valid, u2_valid), opt_results.x, 
                               objective_func, robust_cov = robust_cov)
        

    # abstract this function to fit
    def fit_mm(self, u1: float, u2: float, robust_cov: bool = True, adj: float = 1e-5) -> None:

        u1_valid, u2_valid = self._handle_uu_input(u1, u2, adj = adj)
        tau = utils.empirical_kendall_tau(u1, u2)
        opt_params = self._tau_to_params(tau)

        # still want to pass params through post processing to obtain standard errors
        objective_func = self._get_obj_func(u1_valid, u2_valid)

        self._post_process_fit(utils.flatten_concatenate(u1_valid, u2_valid), 
                               opt_params, objective_func=objective_func, robust_cov = robust_cov)
    

    def _get_top_summary_table(self) -> Tuple[list, list]:

        now = datetime.now()
        top_left = [
            ("Model Name:", self.model_name), ("Model Family:", self.family_name), 
            ("Estimation Method:", self.estimation_method_str),("Num. Params:", self.k), ("Num. Obs:", self.n),
            ("Date:", now.strftime("%a, %b %d %Y")),("Time:", now.strftime("%H:%M:%S")), ("", ""),
        ]

        top_right = [
            ("Log-Likelihood:", utils.format_func(self.LL, 10, 4)), ("AIC:", utils.format_func(self.aic, 10, 4)),
            ("BIC:", utils.format_func(self.bic, 10, 4)), ("Kendall's Tau:", utils.format_func(self.tau, 10, 4)), 
            ("Spearman's Rho:", utils.format_func(self.rho, 10, 4)), ("Upper Tail Depend.:", utils.format_func(self.upper_tail, 10, 4)),
            ("Lower Tail Depend.:", utils.format_func(self.lower_tail, 10, 4)),
            ("", ""),
        ]

        return top_left, top_right
    

    def logpdf(self, u1: Vectorizable, u2: Vectorizable, adj: float = 1e-5) -> Vectorizable:
        valid_u1, valid_u2 = self._handle_uu_input(u1, u2, adj = adj)
        return self._logpdf(valid_u1, valid_u2, *self.params)
    

    def _logpdf(self, u1: Vectorizable, u2: Vectorizable, *params: float) -> Vectorizable:
        raise NotImplementedError


    def pdf(self, u1: Vectorizable, u2: Vectorizable, adj: float = 1e-5) -> Vectorizable: 
        valid_u1, valid_u2 = self._handle_uu_input(u1, u2, adj = adj)
        return self._pdf(valid_u1, valid_u2, *self.params)
    

    def _pdf(self, u1: Vectorizable, u2: Vectorizable, *params: float) -> Vectorizable:
        #exp of log pdf
        return np.exp(self._logpdf(u1, u2, *params))
    

    def cdf(self, u1: Vectorizable, u2: Vectorizable, adj: float = 1e-5) -> Vectorizable:
        valid_u1, valid_u2 = self._handle_uu_input(u1, u2, adj = adj)
        return self._reshape_wrapper(valid_u1, valid_u2, self._cdf, *self.params)
    

    def _cdf(self, u1: Vectorizable, u2: Vectorizable, *params: float, adj = 1e-5) -> Vectorizable:
        raise NotImplementedError
    

    def conditional_cdf(self, u1: Vectorizable, u2: Vectorizable, adj: float = 1e-5) -> Vectorizable:
        valid_u1, valid_u2 = self._handle_uu_input(u1, u2, adj = adj)
        return self._conditional_cdf(valid_u1, valid_u2, *self.params)
    

    def _conditional_cdf(self, u1: Vectorizable, u2: Vectorizable, *params: float, adj: float = 1e-5) -> Vectorizable:
        raise NotImplementedError
    

    def conditional_ppf(self, u1: Vectorizable, q: Vectorizable, adj: float = 1e-5) -> Vectorizable:
        # what is the q-th quantile of u2 given u1?
        # i.e., what is the median given u1 = 0.01?
        # adj only used 

        u1, q = self._handle_uu_input(u1, q, adj = adj)
        return self._conditional_ppf(u1, q, *self.params, adj = adj)
    
    
    def _conditional_ppf(self, u1: Vectorizable, q: Vectorizable, *params: float, adj: float = 1e-5) -> Vectorizable:

        # default implementation of the conditional quantile function uses numerical optimization method
        # on condtional cdf to find inverse
        # this code runs into problems if the data is < adj or > 1 - adj

        return utils.solve_for_conditional_ppf(self._conditional_cdf, u1, q, *params, adj = adj)

    
    @property
    def lower_tail(self) -> float:
        return self._lower_tail_dependance(*self.params)


    def _lower_tail_dependance(self, *params: float) -> float:
        # the limit of "quantile dependance" when q approaches 0
        # adjustment factor is used for q
        raise NotImplementedError
    

    @property
    def upper_tail(self) -> float:
        return self._upper_taiL_dependance(*self.params)


    def _upper_taiL_dependance(self, *params: float) -> float:
        # the limit of "quantile dependance" when q approaches 1
        # adjustment factor is used for q
        # too numerically unstable to use numerical methods
        raise NotImplementedError

 
    def quantile_dependance(self, q: Vectorizable, adj: float = 1e-5) -> Vectorizable:
        # this doesn't work for scalar inputs
        valid_q = self._handle_u_input(q, adj = adj)
        return self._quantile_dependance(valid_q, *self.params)
    

    def _quantile_dependance(self, q: Vectorizable, *params: float) -> Vectorizable:
        # if q > 0.5: probability of u2 > q given u1 > q
        # if q < 0.5: probability of u2 < q given u1 < q
        # this can be thought of geometrically using the CDF

        qq_point = self._cdf(q, q, *params)
        return np.where(q > 0.5, (1 - 2 * q + qq_point) / (1 - q), qq_point / q)
    

    def simulate(self, n: int = 1000, seed: Union[int, None] = None, adj: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:

        rng = np.random.default_rng(seed = seed)

        u1 = rng.uniform(size = n)
        q = rng.uniform(size = n)
        u2 = self.conditional_ppf(u1, q, adj = adj)
        
        return u1, u2