import utils
from base import Base

import numpy as np
from scipy import stats
from scipy.special import gamma
from datetime import datetime

class Marginal(Base):
    def __init__(self, rv_obj, *args, **kwargs):

        self.rv_obj = rv_obj
        self.summary_title = "Marginal Distribution"
        super().__init__(*args, **kwargs)


    def _handle_input(self, x_or_u, is_x = True, adj = 1e-4):
        if not utils.is_arraylike(x_or_u):
            raise SyntaxError
        
        if is_x:
            return x_or_u
        
        return utils.clip_u_input(x_or_u, adj = adj)


    # WRAPPER / EXTERNAL functions

    def cdf(self, x):
        return self._cdf(self._handle_input(x), *self.params)
    
    def _cdf(self, x, *params):
        return self.rv_obj.cdf(x, *params)
    
    def pdf(self, x):
        return self._pdf(self._handle_input(x), *self.params) 
    
    def _pdf(self, x, *params):
        return self.rv_obj.pdf(x, *params)

    def ppf(self, u):
        return self._ppf(self._handle_input(u, is_x = False), *self.params)

    def _ppf(self, x, *params):
        return self.rv_obj.ppf(x, *params)
    
    def logpdf(self, x):
        return self._logpdf(self._handle_input(x), *self.params)
    
    def _logpdf(self, x, *params):
        return self.rv_obj.logpdf(x, *params)

    
    def _log_likelihood(self, x, *params):
        # this will be called by the joint probability disitribution when getting total LL
        # not needed for fitting 
        return np.sum(self._logpdf(x, *params))
    

    def fit(self, x):
        valid_x = self._handle_input(x)

        # relying on scipy implementation of fit
        opt_params = self.rv_obj.fit(x)
        self._post_process_fit(np.array([*opt_params]), self._get_objective_func(x), len(valid_x))
        
    
    def simulate(self, n = 1000, seed = None):
        gen = np.random.default_rng(seed = seed)
        u = gen.uniform(size = n)
        return self.ppf(u)
    

    def _get_top_summary_table(self):
        now = datetime.now()
        top_left = [
            ("Model Name:", self.model_name), 
            ("Method:", "MLE"),("Num. Params:", self.k), ("Num. Obs:", self.n),
            ("Date:", now.strftime("%a, %b %d %Y")),("Time:", now.strftime("%H:%M:%S")), ("", ""), ("", ""),
        ]

        top_right = [
            ("Log-Likelihood:", utils.format_func(self.LL, 10, 4)), ("AIC:", utils.format_func(self.aic, 10, 4)),
            ("BIC:", utils.format_func(self.bic, 10, 4)), (" ", " "), (" ", " "), (" ", " "),
            ("", ""), ("", ""),
        ]

        return top_left, top_right
    


class Normal(Marginal):
    def __init__(self, mu = 0, sigma = 1, adj = 1e-4):

        # the order of these params depends on SciPy
        super().__init__(stats.norm, model_name = "Normal", initial_param_guess = [0, 1], 
                         param_names = ("mu", "sigma"), param_bounds = [(-np.inf, np.inf), (adj, np.inf)],
                         params = (mu, sigma))



class StudentsT(Marginal):
    def __init__(self, df = 30, mu = 0, sigma = 1, adj = 1e-4):
       
        # the order of these params depends on SciPy
        super().__init__(stats.t, model_name = "StudentT", initial_param_guess = [30, 0, 1], 
                         param_names = ("df", "mean", "stdev"), param_bounds = [(1, np.inf), (-np.inf, np.inf), (adj, np.inf)],
                         params = (df, mu, sigma))
        
class StandardSkewedT(Marginal):
    # hansen 1994

    def __init__(self, eta = 30, lam = 0, adj = 1e-2):
        super().__init__(None, model_name = "SkewedStudentsT", 
                         initial_param_guess = [30, 0], param_names = ["mu", "sigma", "eta", "lam"],
                         param_bounds = [(2 + adj, np.inf), (-1 + adj, 1 - adj)],
                         params = (eta, lam))

        
    def _logpdf(self, x, eta, lam):

        # constants
        C = gamma((eta + 1) / 2) / (np.sqrt(np.pi * (eta - 2))  * gamma(eta / 2))
        A = 4 * lam * C * (eta - 2) / (eta - 1)
        B = np.sqrt(1 + 3 * (lam**2) - (A**2))

        # this introduces skewness
        denom = np.where(x < -A/B, 1 - lam, 1 + lam)
        inside_term = 1 + 1/(eta - 2) * np.square((B * x + A)/denom)
        return np.log(B) + np.log(C) - ((eta + 1) / 2) * np.log(inside_term)
    

    def _pdf(self, x, eta, lam):
        return np.exp(self._logpdf(x, eta, lam))



class Uniform(Marginal):
    def __init__(self, a = 0, b = 1):
        super().__init__(stats.uniform, model_name = "Uniform", initial_param_guess = [0, 1],
                         param_names = ("a", "b"), param_bounds = [(-np.inf, np.inf), (-np.inf)],
                         params = (a, b))
        
    def _to_scipy_params(self, a, b):
        # scipy parameterizes by loc and scale
        # scale = b - a
        return a, b - a
        

    def _cdf(self, x, a, b):
        return self.rv_obj.cdf(x, *self._to_scipy_params(a, b))
    
    def _pdf(self, x, a, b):
        return self.rv_obj.pdf(x, *self._to_scipy_params(a, b))
    
    def _ppf(self, x, a, b):
        return self.rv_obj.ppf(x, *self._to_scipy_params(a, b))
    
    def _logpdf(self, x, a, b):
        return self.rv_obj.logpdf(x, *self._to_scipy_params(a, b))

        




# skewed normal & skewed t
# standard t / skewed t to help pin down df



