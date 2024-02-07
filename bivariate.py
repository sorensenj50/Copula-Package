import utils
from base import Base

import numpy as np
from scipy import stats
from scipy.optimize import minimize, brentq
from scipy.special import gamma
from datetime import datetime



class BivariateCopula(Base):
    
    def _handle_u_input(self, u, adj):
        if not (utils.is_arraylike(u) or utils.is_number(u)):
            # must be number or array

            raise TypeError

        return utils.clip_u_input(u, adj)

    def _handle_uu_input(self, u1, u2, adj):


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
    
    
    def _reshape_wrapper(self, u1, u2, f, *params):
        # if u1 and u2 are arrays, ensures flat input and reshapes after applying function

        # because of earlier validation, if 1 is array, then both are
        if utils.is_arraylike(u1):
            out_shape = u1.shape
            return f(u1.flatten(), u2.flatten(), *params).reshape(out_shape)

        return f(u1, u2, *params)

    

class PureCopula(BivariateCopula):
    def __init__(self, *args, **kwargs):
        self.summary_title = "Bivariate Copula"
        super().__init__(*args, **kwargs)


    def _kendall_tau(self, *params):
        raise NotImplementedError
    

    def _spearman_rho(self, *params):
        raise NotImplementedError
    

    def _log_likelihood(self, u1, u2, *params):
        return np.sum(self._logpdf(u1, u2, *params))
    

    def fit(self, u1, u2, method = "Powell", initial_param_guesses = None, adj = 1e-4):

        # input validation
        u1_valid, u2_valid = self._handle_uu_input(u1, u2, adj = adj)
        

        # check length of initial param guesses
        if (initial_param_guesses is not None) and (len(initial_param_guesses) != len(self.params)):
            raise SyntaxError
    
        initial_guess = initial_param_guesses if initial_param_guesses is not None else self.initial_param_guess

        objective_func = self._get_objective_func(u1_valid, u2_valid)
        opt_results = self._fit(objective_func, initial_guess, self.param_bounds, method = method)
        self._post_process_fit(opt_results.x, objective_func, len(u1.flatten()))


    def _fit(self, f, initial_param_guess, param_bounds, method = "Powell"):
        
        # defualt mle optimization (aka canonical likelihood implementation)
        return minimize(f, initial_param_guess, bounds = param_bounds, method = method)


    def _post_process_fit(self, opt_params, objective_func, n):
        
        super()._post_process_fit(opt_params, objective_func, n)
        self.tau = self._kendall_tau(*self.params)
        self.rho = self._spearman_rho(*self.params)


    def _get_top_summary_table(self):
        now = datetime.now()
        top_left = [
            ("Model Name:", self.model_name), ("Model Family:", self.family_name), 
            ("Method:", "CMLE"),("Num. Params:", self.k), ("Num. Obs:", self.n),
            ("Date:", now.strftime("%a, %b %d %Y")),("Time:", now.strftime("%H:%M:%S")), ("", ""), ("", ""),
        ]

        top_right = [
            ("Log-Likelihood:", utils.format_func(self.LL, 10, 4)), ("AIC:", utils.format_func(self.aic, 10, 4)),
            ("BIC:", utils.format_func(self.bic, 10, 4)), ("Kendall's Tau:", utils.format_func(self.tau, 10, 4)), 
            ("Spearman's Rho:", utils.format_func(self.rho, 10, 4)), ("Upper Tail Depend.:", "NA"),("Lower Tail Depend.:", "NA"),
            ("", ""), ("", ""),
        ]

        return top_left, top_right

        
    def conditional_quantile(self, u1, q, adj = 1e-4):
        # what is the q-th quantile of u2 given u1?
        # i.e., what is the median given u1 = 0.01?

        u1, q = self._handle_uu_input(u1, q, adj = adj)
        return self._conditional_quantile(u1, q, *self.params)
    

    def _conditional_quantile(self, u1, q, *params, adj = 1e-4):
        # default implementation of the conditional quantile function uses numerical optimization method
        # on condtional cdf

        def F(u1, q):
            f = lambda u2: self._conditional_cdf(u1, u2, *params) - q
            return brentq(f, a = adj, b = 1 - adj)

        # both should be the same type
        # handling scalar input
        if utils.is_number(u1):
            return F(u1, q)
        
        # array input

        # flattening to handle any shape
        u1_flat = np.array(u1).flatten(); q_flat = np.array(q).flatten()

        n = len(u1_flat)
        u2_flat = np.empty(shape = n)

        for i in range(n):
            u2_flat[i] = F(u1_flat[i], q_flat[i])
        
        # reshaping
        return u2_flat.reshape(u1.shape)
    

    def logpdf(self, u1, u2, adj = 1e-4):
        valid_u1, valid_u2 = self._handle_uu_input(u1, u2, adj = adj)
        return self._logpdf(valid_u1, valid_u2, *self.params)
    

    def _logpdf(self, u1, u2, *params):
        raise NotImplementedError
        

    def simulate(self, n = 1000, seed = None):
        gen = np.random.default_rng(seed = seed)

        u1 = gen.uniform(size = n)
        q = gen.uniform(size = n)
        u2 = self._conditional_quantile(u1, q, *self.params)
        
        return u1, u2


    def pdf(self, u1, u2, adj = 1e-5): 
        valid_u1, valid_u2 = self._handle_uu_input(u1, u2, adj = adj)
        return self._pdf(valid_u1, valid_u2, *self.params)
    

    def _pdf(self, u1, u2, *params):
        #exp of log pdf
        return np.exp(self._logpdf(u1, u2, *params))
    

    def cdf(self, u1, u2, adj = 1e-5):
        valid_u1, valid_u2 = self._handle_uu_input(u1, u2, adj = adj)
        return self._reshape_wrapper(valid_u1, valid_u2, self._cdf, *self.params)
    

    def _cdf(self, u1, u2, *params):
        raise NotImplementedError
    

    def quantile_dependance(self, q, adj = 1e-4):
        valid_q = self._handle_u_input(q, adj = adj)
        return self._quantile_dependance(valid_q, adj, *self.params)
    

    def _quantile_dependance(self, q, adj, *params):
        # if q > 0.5: probability of u2 > q given u1 > q
        # if q < 0.5: probability of u2 < q given u1 < q
        # this can be thought of geometrically using the CDF

        one = (np.ones(shape = q.shape) if utils.is_arraylike(q) else 1) - adj

        A = self._cdf(q, one, *params)
        B = self._cdf(one, one, *params)
        C = self._cdf(one, q, *params)
        D = self._cdf(q, q, *params)

        return np.where(q > 0.5, (B - A + D - C) / (B - A), D / A)

    

        

class Elliptical(PureCopula):
    def __init__(self, *args, **kwargs):
        self.family_name = "Elliptical"
        super().__init__(*args, **kwargs)


    def _cov_det(self, Q):
        return 1 - Q ** 2
    

    def _scale_factor(self, Q):
        return np.sqrt(self._cov_det(Q))
    

    def _distance(self, z1, z2, Q):
        # distance between points, used in the copula density
        raise NotImplementedError



class Normal(Elliptical):
    def __init__(self, Q = 0, adj = 1e-4):
        super().__init__(model_name = "Normal", initial_param_guess = [1], 
                         param_bounds = [(-1 + adj, 1 - adj)], param_names = ("Q",), 
                         params = (Q,))
        
    
    def _distance(self, z1, z2, Q):
        # modified mahalonobis distance
        return ((Q * z1) ** 2 - (2 * Q * z1 * z2) + (Q * z2) ** 2) / self._cov_det(Q)
    

    def _cdf(self, u1, u2, Q):
        z1 = stats.norm.ppf(u1); z2 = stats.norm.ppf(u2)
        z = np.stack([z1, z2], axis = 1)
        return stats.multivariate_normal.cdf(z, cov = np.array([[1, Q],[Q, 1]]))
    
    
    def _logpdf(self, u1, u2, Q):
        z1 = stats.norm.ppf(u1); z2 = stats.norm.ppf(u2)
        return -np.log(self._scale_factor(Q)) - 1/2 * self._distance(z1, z2, Q)


    def _conditional_quantile(self, u1, q, Q):

        # z of conditioning variable
        z1 = stats.norm.ppf(u1)

        # z of the input quantile, given the conditioning z and their correlation
        z2 = stats.norm.ppf(q, loc = Q * z1, scale = self._scale_factor(Q))

        # back to quantile space
        return stats.norm.cdf(z2)
    

    def _kendall_tau(self, Q):
        return 2 * np.arcsin(Q) / np.pi
    

    def _spearman_rho(self, Q):
        return 6 * np.arcsin(Q / 2) / np.pi
    


    
    
class StudentsT(Elliptical):
    def __init__(self, df = 30, Q = 0, adj = 1e-4, df_upper_bound = 100):
        super().__init__(model_name = "StudentT", initial_param_guess = [30, 0], 
                         param_bounds = [(1, df_upper_bound), (-1 + adj, 1 - adj)], 
                         param_names = ("df", "Q"), params = (df, Q))


    def _distance(self, z1, z2, Q):
        return ((z1 ** 2) - (2 * Q * z1 * z2) + (z2 ** 2)) / self._cov_det(Q)
    

    def _cdf(self, u1, u2, df, Q):
        z1 = stats.t.ppf(u1, df); z2 = stats.t.ppf(u2, df)
        z = np.stack([z1, z2], axis = 1)
    
        return stats.multivariate_t.cdf(z, df = df, shape = np.array([[1, Q],[Q, 1]]))
    

    def _logpdf(self, u1, u2, df, Q):
        n = 2

        # to t variables
        z1 = stats.t.ppf(u1, df); z2 = stats.t.ppf(u2, df)

        log_K = np.log(gamma((df + n) / 2)) + (n - 1) * np.log(gamma(df / 2)) + -n * np.log(gamma((df + 1) / 2))
        log_scale = np.log(self._scale_factor(Q))
        log_numerator = (-(df + n)/2) * np.log(1 + self._distance(z1, z2, Q) / df)
        log_denom = (-(df + 1)/2) * np.log((1 + (z1 ** 2)/df) * (1 + (z2 ** 2)/df))

        return (log_K - log_scale) + (log_numerator - log_denom)
    
    
    def _conditional_quantile(self, u1, q, df, Q):
        # df + 1 on both distributions??

        # standardized univariate t with df deg of freedom
        z1 = stats.t.ppf(u1, df)
        
        # conditional standardized t distribution
        z2 = stats.t.ppf(q, df + 1, loc = z1 * Q, scale = (self._scale_factor(Q) * np.sqrt(df + z1 **2)) / np.sqrt(df + 1))

        return stats.t.cdf(z2, df + 1)
    

    def _kendall_tau(self, df, Q):
        return 2 * np.arcsin(Q) / np.pi
    

    def _spearman_rho(self, df, Q):
        return np.nan
    


class Archimedean(PureCopula):
    def __init__(self, rotate_u1, rotate_u2, *args, **kwargs):

        self.rotate_u1 = rotate_u1
        self.rotate_u2 = rotate_u2

        self.u1_rotation_func = (lambda u: 1 - u) if self.rotate_u1 else (lambda u: u)
        self.u2_rotation_func = (lambda u: 1 - u) if self.rotate_u2 else (lambda u: u)

        self.family_name = "Archimedean"

        super().__init__(*args, **kwargs)
    
    

# rotation?
class Clayton(Archimedean):
    def __init__(self, alpha = 1e-4, rotate_u1 = False, rotate_u2 = False, adj = 1e-4):
        super().__init__(rotate_u1, rotate_u2, model_name = "Clayton", initial_param_guess = [adj],
                         param_bounds = [(adj, np.inf)], param_names = ("alpha",),
                         params = (alpha,))

    
    def _cdf(self, u1, u2, alpha):
        return np.power((np.power(u1, -alpha) + np.power(u2, -alpha) - 1), -1/alpha)


    def _logpdf(self, u1, u2, alpha):

        log_1 = np.log(alpha + 1)
        log_2 = (-2 - 1/alpha) * np.log(np.power(u1, -alpha) + np.power(u2, -alpha) - 1)
        log_3 = (-alpha - 1) * (np.log(u1) + np.log(u2))

        return log_1 + log_2 + log_3
    

    def _conditional_quantile(self, u1, q, alpha):
        return np.power((1 + np.power(u1, -alpha) * (np.power(q, -alpha/(1+alpha)) -1)), -1/alpha)
    
    
    def _kendall_tau(self, alpha):
        return alpha / (alpha + 2)
    

    

# rotation?
class Gumbel(Archimedean):
    def __init__(self, delta = 1):
        super().__init__(model_name = "Gumbel", initial_param_guess = [1], 
                         param_bounds = [(1, np.inf)], param_names = ("delta",),
                         params = (delta,))


    def _A(self, u1, u2, delta):
        return np.power(np.power(-np.log(u1), delta) + np.power(-np.log(u2), delta), 1/delta)


    def _cdf(self, u1, u2, delta):
        return np.exp(-self._A(u1, u2, delta))
    

    def _conditional_cdf(self, u1, u2, delta):
        # conditional of u2 given u1

        prod1 = 1/u1
        prod2 = np.power(-np.log(u1), delta - 1)
        prod3 = np.power(np.power(-np.log(u1), delta) + np.power(-np.log(u2), delta), (1 - delta)/delta)
        A = self._A(u1, u2, delta)

        return prod1 * prod2 * prod3 * np.exp(-A)


    def _logpdf(self, u1, u2, delta):
        A = self._A(u1, u2, delta)

        log_1 = np.log(A + delta - 1)
        log_2 = (1-2*delta) * np.log(A)
        log_3 = -A - (np.log(u1) + np.log(u2))
        log_4 = (delta - 1) * (np.log(-np.log(u1)) + np.log(-np.log(u2)))

        return log_1 + log_2 + log_3 + log_4
    
    def _kendall_tau(self, delta):
        return 1 - 1 / delta
    

    
    




    





    


