import utils
import base

import numpy as np
from scipy import stats
from scipy.optimize import minimize, brentq
from scipy.special import gamma
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor


class BivariateCopula(base.Base):
    def __init__(self, *args, **kwargs):
        self.summary_title = "Bivariate Copula"
        super().__init__(*args, **kwargs)

    @property
    def tau(self):
        return self._params_to_tau(*self.params)
    
    @property
    def rho(self):
        return self._params_to_rho(*self.params)
    
    @property
    def lower_tail(self):
        return self._lower_tail_dependance(*self.params)
    
    @property
    def upper_tail(self):
        return self._upper_taiL_dependance(*self.params)

    

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
    

    def _tau_to_params(self, tau):
        # return a tuple
        return (np.nan,)
    

    def _params_to_tau(self, *params):
        return np.nan
    

    def _rho_to_params(self, rho):
        return (np.nan,)
    

    def _params_to_rho(self, *params):
        return np.nan
    

    def _log_likelihood(self, u1, u2, *params):
        return np.sum(self._logpdf(u1, u2, *params))
    
    
    def fit(self, u1, u2, method = "MLE", optimizer = "Powell", initial_param_guesses = None, robust_cov = True, adj = 1e-4):

        # input validation
        u1_valid, u2_valid = self._handle_uu_input(u1, u2, adj = adj)
        

        # check length of initial param guesses
        if (initial_param_guesses is not None) and (len(initial_param_guesses) != len(self.params)):
            raise SyntaxError
    
        initial_guess = initial_param_guesses if initial_param_guesses is not None else self.initial_param_guess
        objective_func = self._get_objective_func(u1_valid, u2_valid)

        opt_results = self._fit(objective_func, initial_guess, self.param_bounds, optimizer = optimizer)

        self._post_process_fit(utils.flatten_concatenate(u1_valid, u2_valid), opt_results.x, 
                               objective_func, robust_cov = robust_cov)
        

    # abstract this function to fit
    def fit_mm(self, u1, u2, robust_cov = True, adj = 1e-4):

        u1_valid, u2_valid = self._handle_uu_input(u1, u2, adj = adj)
        tau = utils.empirical_kendall_tau(u1, u2)
        opt_params = self._tau_to_params(tau)

        # still want to pass params through post processing to obtain standard errors
        objective_func = self._get_objective_func(u1_valid, u2_valid)

        self._post_process_fit(utils.flatten_concatenate(u1_valid, u2_valid), 
                               opt_params, objective_func=objective_func, robust_cov = robust_cov)
    

    def _get_top_summary_table(self):
        now = datetime.now()
        top_left = [
            ("Model Name:", self.model_name), ("Model Family:", self.family_name), 
            ("Method:", "MLE"),("Num. Params:", self.k), ("Num. Obs:", self.n),
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

        
    def conditional_quantile(self, u1, q, adj = 1e-4):
        # what is the q-th quantile of u2 given u1?
        # i.e., what is the median given u1 = 0.01?

        u1, q = self._handle_uu_input(u1, q, adj = adj)
        return self._conditional_quantile(u1, q, *self.params)
    

    
    def _conditional_quantile(self, u1, q, *params, adj = 1e-6):
        # re write to use reshape wrapper
        # re write to throw error if f(a) or f(b) error

        # default implementation of the conditional quantile function uses numerical optimization method
        # on condtional cdf to find inverse
        # this code runs into problems if the data is < adj or > 1 - adj


        def F(u1, q, *params, adj = 1e-6):
            f = lambda u2: self._conditional_cdf(u1, u2, *params) - q
            return brentq(f, a = adj, b = 1 - adj)

        if utils.is_number(u1) and utils.is_number(q):
            return F(u1, q, *params)


        # flattening to handle any shape
        # handling case of non np array input (list, etc)
        out_shape = u1.shape
        u1_flat = np.array(u1).flatten(); q_flat = np.array(q).flatten()
        u2 = [F(u1, q, *params, adj = adj) for u1, q in zip(u1_flat, q_flat)]
        
        # reshaping
        return np.array(u2).reshape(out_shape)
    

    def logpdf(self, u1, u2, adj = 1e-4):
        valid_u1, valid_u2 = self._handle_uu_input(u1, u2, adj = adj)
        return self._logpdf(valid_u1, valid_u2, *self.params)
    

    def _logpdf(self, u1, u2, *params):
        raise NotImplementedError
        

    def simulate(self, n = 1000, seed = None, adj = 1e-6):
        rng = np.random.default_rng(seed = seed)

        u1 = rng.uniform(size = n)
        q = rng.uniform(size = n)
        u2 = self._conditional_quantile(u1, q, *self.params, adj = adj)
        
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
    

    def _lower_tail_dependance(self, *params):
        # the limit of "quantile dependance" when q approaches 0
        # adjustment factor is used for q
        raise NotImplementedError
    

    def _upper_taiL_dependance(self, *params):
        # the limit of "quantile dependance" when q approaches 1
        # adjustment factor is used for q
        # too numerically unstable to use numerical methods

        raise NotImplementedError

 
    def quantile_dependance(self, q, adj = 1e-4):
        # this doesn't work for scalar inputs
        valid_q = self._handle_u_input(q, adj = adj)
        return self._quantile_dependance(valid_q, *self.params)
    

    def _quantile_dependance(self, q, *params):
        # if q > 0.5: probability of u2 > q given u1 > q
        # if q < 0.5: probability of u2 < q given u1 < q
        # this can be thought of geometrically using the CDF
        qq_point = self._cdf(q, q, *params)
        return np.where(q > 0.5, (1 - 2 * q + qq_point) / (1 - q), qq_point / q)
    


class Independent(BivariateCopula):
    def __init__(self):
        super().__init__(model_name = "Independent", initial_param_guess = [],
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
    
    def _conditional_quantile(self, u1, q):
        return q




class Elliptical(BivariateCopula):
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
        super().__init__(model_name = "Normal", initial_param_guess = [0], 
                         param_bounds = [(-1 + adj, 1 - adj)], param_names = ("Q",), 
                         params = (Q,))
        
    
    def _distance(self, z1, z2, Q):
        # modified mahalonobis distance
        return ((Q * z1) ** 2 - (2 * Q * z1 * z2) + (Q * z2) ** 2) / self._cov_det(Q)
    

    def _cdf(self, u1, u2, Q):
        z1 = stats.norm.ppf(u1); z2 = stats.norm.ppf(u2)
        z = np.stack([np.atleast_1d(z1), np.atleast_1d(z2)], axis = 1)
        return stats.multivariate_normal.cdf(z, cov = np.array([[1, Q],[Q, 1]]))
    
    
    def _logpdf(self, u1, u2, Q):
        z1 = stats.norm.ppf(u1); z2 = stats.norm.ppf(u2)
        return -np.log(self._scale_factor(Q)) - 1/2 * self._distance(z1, z2, Q)


    def _conditional_quantile(self, u1, q, Q, adj = 1e-4):
        # adj unused but here for consistency

        # z of conditioning variable
        z1 = stats.norm.ppf(u1)

        # z of the input quantile, given the conditioning z and their correlation
        z2 = stats.norm.ppf(q, loc = Q * z1, scale = self._scale_factor(Q))

        # back to quantile space
        return stats.norm.cdf(z2)
    

    def _params_to_tau(self, Q):
        return 2 * np.arcsin(Q) / np.pi
    
    
    def _tau_to_params(self, tau):
        return tuple(2 * np.sin((np.pi / 6) * tau))


    def _params_to_rho(self, Q):
        return 6 * np.arcsin(Q / 2) / np.pi
    

    def _lower_tail_dependance(self, *params):
        # McNeil 2005
        return 0
    
    def _upper_taiL_dependance(self, *params):
        # McNeil 2005
        return 0
    


    
class StudentsT(Elliptical):
    def __init__(self, df = 30, Q = 0, adj = 1e-4, df_upper_bound = 100):
        super().__init__(model_name = "StudentsT", initial_param_guess = [30, 0], 
                         param_bounds = [(1, df_upper_bound), (-1 + adj, 1 - adj)], 
                         param_names = ("df", "Q"), params = (df, Q))


    def _distance(self, z1, z2, Q):
        return ((z1 ** 2) - (2 * Q * z1 * z2) + (z2 ** 2)) / self._cov_det(Q)
    

    def _cdf(self, u1, u2, df, Q):
        z1 = stats.t.ppf(u1, df); z2 = stats.t.ppf(u2, df)
        z = np.stack([np.atleast_1d(z1), np.atleast_1d(z2)], axis = 1)
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
    
    
    def _conditional_quantile(self, u1, q, df, Q, adj = 1e-4):
        # adj unused but here for consistency
        # df + 1 on both distributions??

        # standardized univariate t with df deg of freedom
        z1 = stats.t.ppf(u1, df)
        
        # conditional standardized t distribution
        z2 = stats.t.ppf(q, df + 1, loc = z1 * Q, scale = (self._scale_factor(Q) * np.sqrt(df + z1 **2)) / np.sqrt(df + 1))

        return stats.t.cdf(z2, df + 1)
    
    # Lindskog et al (2003) for tau and t 
    #def _kendall_t(self, df, Q):
    #
    

    def _tail_dependance(self, df, Q):
        # McNeil 2005
        return 2 * stats.t.cdf(-np.sqrt((df + 1) * (1 - Q) / (1 + Q)), df + 1)
    

    def _upper_taiL_dependance(self, df, Q):
        return self._tail_dependance(df, Q)
    

    def _lower_tail_dependance(self, df, Q):
        return self._tail_dependance(df, Q)
    

    


class Archimedean(BivariateCopula):
    def __init__(self, rotation, model_name, *args, **kwargs):
        self.family_name = "Archimedean"

        # model name has to be set before rotation
        self.model_name = model_name
        self._set_rotation(rotation)
        super().__init__(model_name, *args, **kwargs)

        # setting rotation again because parent class init overrides model_name
        self._set_rotation(rotation)


    def _set_rotation(self, rotation):
        self._check_rotation(rotation)
        self.rotation = rotation

        if rotation > 0:
            self.model_name += " (Rot. {})".format(self.rotation)
        
        # assigning rotation transformation function
            
        if rotation == 0:
            self._pdf_rot_func = lambda u1, u2: (u1, u2)
            self._cdf_rot_func = lambda u1, u2, C: C
            self._quantile_rot_func1 = lambda u1, q: (u1, q)
            self._quantile_rot_func2 = lambda u2: u2
            self._tau_rot_func = lambda x: x
            self._upper_tail_rot = self._unrotated_upper_tail_dependance
            self._lower_tail_rot = self._unrotated_lower_tail_dependance
        
        elif rotation == 90:
            self._pdf_rot_func = lambda u1, u2: (1 - u2, u1)
            self._cdf_rot_func = lambda u1, u2, C: u1 - C
            self._quantile_rot_func1 = lambda u1, q : (u1, 1 - q)
            self._quantile_rot_func2 = lambda u2: 1 - u2
            self._tau_rot_func = lambda x: -x
            self._upper_tail_rot = self._unrotated_lower_upper_dependance
            self._lower_tail_rot = self._unrotated_upper_lower_dependance

        elif rotation == 180:
            self._pdf_rot_func = lambda u1, u2: (1 - u1, 1 - u2)
            self._cdf_rot_func = lambda u1, u2, C: u1 + u2 -1 + C
            self._quantile_rot_func1 = lambda u1, q: (1 - u1, 1 - q)
            self._quantile_rot_func2 = lambda u2: 1 - u2
            self._tau_rot_func = lambda x: x
            self._upper_tail_rot = self._unrotated_lower_tail_dependance
            self._lower_tail_rot = self._unrotated_upper_tail_dependance

        elif rotation == 270:
            self._pdf_rot_func = lambda u1, u2: (u2, 1 - u1)
            self._cdf_rot_func = lambda u1, u2, C: u2 - C
            self._quantile_rot_func1 = lambda u1, q: (1 - u1, q)
            self._quantile_rot_func2 = lambda u2: u2
            self._tau_rot_func = lambda x: -x
            self._upper_tail_rot = self._unrotated_upper_lower_dependance
            self._lower_tail_rot = self._unrotated_lower_upper_dependance
        

    def _check_rotation(self, rotation):
        if rotation not in [0, 90, 180, 270]:
            # input error
            raise SyntaxError
        
    # default is zero, implementation can be overriden
    def _unrotated_upper_tail_dependance(self, *params):
        return 0
    
    
    def _unrotated_lower_tail_dependance(self, *params):
        return 0
    
   
    def _unrotated_upper_lower_dependance(self, *params):
        return 0
    

    def _unrotated_lower_upper_dependance(self, *params):
        return 0
    

    def _upper_taiL_dependance(self, theta):
        return self._upper_tail_rot(theta)
    

    def _lower_tail_dependance(self, theta):
        return self._lower_tail_rot(theta)
    

    
class Clayton(Archimedean):
    def __init__(self, theta = 1e-4, rotation = 0, adj = 1e-4):
        super().__init__(rotation = rotation, model_name = "Clayton", initial_param_guess = [adj],
                         param_bounds = [(adj, np.inf)], param_names = ("theta",), params = (theta,))
        
        

    
    def _cdf(self, u1, u2, theta):

        # rotation variables if necessary
        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)
        C = np.power((np.power(rot_u1, -theta) + np.power(rot_u2, -theta) - 1), -1/theta)

        # passng original variables for additional handling of rotation
        return self._cdf_rot_func(u1, u2, C)


    def _logpdf(self, u1, u2, theta):
        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)

        log_1 = np.log(theta + 1)
        log_2 = (-2 - 1/theta) * np.log(np.power(rot_u1, -theta) + np.power(rot_u2, -theta) - 1)
        log_3 = (-theta - 1) * (np.log(rot_u1) + np.log(rot_u2))

        return log_1 + log_2 + log_3
    

    # how to adjust for rotation?
    def _conditional_quantile(self, u1, q, theta, adj = 1e-4):
        rot_u1, rot_q = self._quantile_rot_func1(u1, q)
        return self._quantile_rot_func2(np.power((1 + np.power(rot_u1, -theta) * (np.power(rot_q, -theta/(1+theta)) -1)), -1/theta))
    
    
    def _params_to_tau(self, theta):
        return self._tau_rot_func(theta / (theta + 2))
    

    def _tau_to_params(self, tau):
        return (2 * tau * (1 / (1 - tau)),)
    

    def _unrotated_lower_tail_dependance(self, theta):
        return 2 ** (-1 / theta)


class Frank(Archimedean):
    def __init__(self, theta = 1e-4, rotation = 0, adj = 1e-4):
        super().__init__(rotation = rotation, model_name = "Frank", initial_param_guess = [adj],
                         param_bounds = [(adj, np.inf)], param_names = ("theta",),
                         params = (theta,))
        

    def _cdf(self, u1, u2, theta):
        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)

        num = (np.exp(-theta * rot_u1) - 1) * (np.exp(-theta * rot_u2) - 1)
        denom = np.exp(-theta) - 1
        C = -1/theta * np.log(1 + num / denom)

        return self._cdf_rot_func(u1, u2, C)
    
    def _params_to_tau(self, theta):
        return 
    


    

# rotation?
class Gumbel(Archimedean):
    def __init__(self, theta = 1, rotation = 0):
        super().__init__(rotation = rotation, model_name = "Gumbel", initial_param_guess = [1], 
                         param_bounds = [(1, np.inf)], param_names = ("theta",),
                         params = (theta,))


    def _A(self, u1, u2, theta):
        return np.power(np.power(-np.log(u1), theta) + np.power(-np.log(u2), theta), 1/theta)


    def _cdf(self, u1, u2, theta):
        # rotating inputs if necessary
        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)
        C = np.exp(-self._A(rot_u1, rot_u2, theta))

        # final transformation on cdf using original inputs
        return self._cdf_rot_func(u1, u2, C)
    

    def _conditional_cdf(self, u1, u2, theta):
        # conditional of u2 given u1
        prod1 = 1/u1
        prod2 = np.power(-np.log(u1), theta - 1)
        prod3 = np.power(np.power(-np.log(u1), theta) + np.power(-np.log(u2), theta), (1 - theta)/theta)
        return prod1 * prod2 * prod3 * np.exp(-self._A(u1, u2, theta))
    

    def _conditional_quantile(self, u1, q, *params, adj = 1e-4):
        # rotation transformation
        # then relying on brentq solver to get quantile given conditional_cdf

        rot_u1, rot_q = self._quantile_rot_func1(u1, q)
        return self._quantile_rot_func2(super()._conditional_quantile(rot_u1, rot_q, *params, adj=adj))
    

    def _logpdf(self, u1, u2, theta):

        # rotating inputs if necessary
        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)

        A = self._A(rot_u1, rot_u2, theta)

        log_1 = np.log(A + theta - 1)
        log_2 = (1-2*theta) * np.log(A)
        log_3 = -A - (np.log(rot_u1) + np.log(rot_u2))
        log_4 = (theta - 1) * (np.log(-np.log(rot_u1)) + np.log(-np.log(rot_u2)))

        return log_1 + log_2 + log_3 + log_4
    

    def _params_to_tau(self, theta):
        return self._tau_rot_func(1 - 1 / theta)
    

    def _tau_to_params(self, tau):
        return (1 / (1 - tau),)


    def _unrotated_upper_tail_dependance(self, theta):
        return 2 - (2 ** (1 / theta))
    


        
    



class NormalMixture(BivariateCopula):
    def __init__(self, p1 = 0.5, p2 = 0.5, Q1 = 0, Q2 = 0, adj = 1e-4):

        # case if lengths of p and Q disagree / with n_normals
        self.summary_title = "Bivariate Copula"
        self.family_name = "Elliptical"
        p1, p2 = self._normalize_p(p1, p2)
        self.base_model = Normal()

        super().__init__("Mixture", [],
                         [(adj, 1 - adj), (adj, 1 - adj), (-1 + adj, 1 - adj), (-1 + adj, 1 - adj)],
                         ["p1", "p2", "Q1", "Q2"], [p1, p2, Q1, Q2])
        

    def _get_weighted_obj_func(self, u1, u2, weights, copula):
        return lambda params: -np.sum(weights * copula._logpdf(u1, u2, *params))
    

    def _normalize_p(self, p1, p2):
        p_sum = p1 + p2
        return p1 / p_sum, p2 / p_sum 
    

    def _get_random_p(self, n, rng):
        return rng.dirichlet(np.ones(n), size = (n, 2))
    

    def _get_random_q(self, n, rng):
        return rng.uniform(-1, 1, size = (n, 2))    


    def fit(self, u1, u2, seed = None, n_init = 20, tol = 1e-4, max_iter = 100):

        # random initilization, 
       
        rng = np.random.default_rng(seed = seed)
        random_p = self._get_random_p(n_init, rng)
        random_Q = self._get_random_q(n_init, rng)

        params_arr = np.empty(shape = (n_init, 4)); LL_list = []

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._run_em_algo, u1, u2, random_p[i, 0], random_p[i, 1], random_Q[i, 0], random_Q[i, 1], 
                                             tol = tol, max_iter = max_iter) for i in range(n_init)]

        for i, future in enumerate(futures):
            *params, LL = future.result()
            params_arr[i] = list(params); LL_list.append(LL)

        best_index = np.argmin(LL_list)
        p1, p2, Q1, Q2 = params_arr[best_index]

        #return p1, p2, Q1, Q2, LL_list[best_index]



    def _run_em_algo(self, u1, u2, p1, p2, Q1, Q2, tol = 1e-4, max_iter = 100, m_method = "MLE"):

        # takes input data and iniitla values for parameters
        i = 0
        LL = 0

        while i < max_iter:
            gamma1, gamma2 = self._e_step(u1, u2, p1, p2, Q1, Q2) 
            new_p1, new_p2, new_Q1, new_Q2, new_LL = self._m_step(u1, u2, gamma1, gamma2, Q1, Q2)
        
            if np.abs(new_LL - LL) < tol:
                return new_p1, new_p2, new_Q1, new_Q2, new_LL
            
            # setting new variables
            p1, p2, Q1, Q2, LL = new_p1, new_p2, new_Q1, new_Q2, new_LL
            i += 1

        # hit max iterations
        return p1, p2, Q1, Q2, LL


    def _e_step(self, u1, u2, p1, p2, Q1, Q2):
        gamma1 = p1 * self.base_copula._pdf(u1, u2, Q1)
        gamma2 = p2 * self.base_copula._pdf(u1, u2, Q2)
        gamma_sum = gamma1 + gamma2

        return gamma1 / gamma_sum, gamma2 / gamma_sum
    

    def _m_step(self, u1, u2, gamma1, gamma2, Q1, Q2, optimizer = "Powell"):

        new_p1 = np.mean(gamma1); new_p2 = np.mean(gamma2)

        f1 = self._get_weighted_obj_func(u1, u2, gamma1, self.base_copula)
        f2 = self._get_weighted_obj_func(u1, u2, gamma2, self.base_copula)

        # use previous Q has initial guess?
        results1 = self.base_copula._fit(f1, [Q1], self.base_copula.param_bounds, 
                                             optimizer = optimizer)
        
        results2 = self.base_copula._fit(f2, [Q2], self.base_copula.param_bounds, 
                                             optimizer = optimizer)

        # returning new_p1, new_p2, new_Q1, new_Q2, and the total log likelihood
        return new_p1, new_p2, results1.x[0], results2.x[0], -1 * (results1.fun + results2.fun)
    

    def _pdf(self, u1, u2, p1, p2, Q1, Q2):
        return p1 * self.base_model._pdf(u1, u2, Q1) + p2 * self.base_model._pdf(u1, u2, Q2)
    

    def _logpdf(self, u1, u2, *params):
        return np.log(self._pdf(u1, u2, *params))


    def _cdf(self, u1, u2, p1, p2, Q1, Q2):
        return p1 * self.base_model._cdf(u1, u2, Q1) + p2 * self.base_model._cdf(u1, u2, Q2)
    

    def _conditional_cdf(self, u1, u2, p1, p2, Q1, Q2):
        # cdf of u2 conditioned on u1

        z1 = stats.norm.ppf(u1); z2 = stats.norm.ppf(u2)

        num1, num2 = z2 - Q1 * z1, z2 - Q2 * z1
        denom1, denom2 = self.base_model._scale_factor(Q1), self.base_model._scale_factor(Q2)

        val = p1 * stats.norm.cdf(num1 / denom1) + p2 * stats.norm.cdf(num2 / denom2)
        return val
    

    def simulate(self, n = 1000, seed = None, adj = 1e-4):

        p1, p2, Q1, Q2 = self.params

        rng = np.random.default_rng(seed = seed)
        param_draw = rng.choice([Q1, Q2], p = [p1, p2], replace = True, size = n)

        u1 = rng.uniform(size = n)
        u2 = np.empty(shape = n)
        q = rng.uniform(size = n)

        for i, Q in enumerate(param_draw):
            u2[i] = self.base_model._conditional_quantile(u1[i], q[i], Q, adj = adj)

        return u1, u2
    
    
    def _lower_tail_dependance(self, *params):
        # mixture of two normals has to have 0 right?
        return 0
    

    def _upper_taiL_dependance(self, *params):
        # mixture of two normals has to have zero right?
        return 0

    
    




    





    


