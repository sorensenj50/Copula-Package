import utils
from base import Base

import numpy as np
from scipy import stats
from scipy.special import gamma, beta
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from datetime import datetime

class Marginal(Base):
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
                         param_names = ["mu", "sigma"], param_bounds = [(-np.inf, np.inf), (adj, np.inf)],
                         params = [mu, sigma])



class StudentsT(Marginal):
    def __init__(self, df = 30, mu = 0, sigma = 1, adj = 1e-4):
       
        # the order of these params depends on SciPy
        super().__init__(stats.t, model_name = "StudentT", initial_param_guess = [30, 0, 1], 
                         param_names = ["df", "mean", "stdev"], param_bounds = [(1, np.inf), (-np.inf, np.inf), (adj, np.inf)],
                         params = [df, mu, sigma])
        

class Uniform(Marginal):
    def __init__(self, a = 0, b = 1):
        super().__init__(stats.uniform, model_name = "Uniform", initial_param_guess = [0, 1],
                         param_names = ["a", "b"], param_bounds = [(-np.inf, np.inf), (-np.inf)],
                         params = [a, b])
        
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


        
class StandardSkewedT(Marginal):
    # hansen 1994

    # need to define fit!!

    def __init__(self, eta = 30, lam = 0, adj = 1e-2):
        super().__init__(None, model_name = "SkewedStudentsT", 
                         initial_param_guess = [30, 0], param_names = ["mu", "sigma", "eta", "lam"],
                         param_bounds = [(2 + adj, np.inf), (-1 + adj, 1 - adj)],
                         params = [eta, lam])

    def _get_ABC(self, eta, lam):
        C = gamma((eta + 1) / 2) / (np.sqrt(np.pi * (eta - 2))  * gamma(eta / 2))
        A = 4 * lam * C * (eta - 2) / (eta - 1)
        B = np.sqrt(1 + 3 * (lam**2) - (A**2))

        return A, B, C
        
    def _logpdf(self, x, eta, lam):

        # constants
        A, B, C = self._get_ABC(eta, lam)

        # this introduces skewness
        denom = np.where(x < -A/B, 1 - lam, 1 + lam)
        inside_term = 1 + 1/(eta - 2) * np.square((B * x + A)/denom)
        return np.log(B) + np.log(C) - ((eta + 1) / 2) * np.log(inside_term)
    

    def _pdf(self, x, eta, lam):
        return np.exp(self._logpdf(x, eta, lam))


    def _ppf(self, u, eta, lam):
        # TODO: check PPF aligns with CDF

        # source: Tino Contino (DirtyQuant)

        # constants
        A, B, _ = self._get_ABC(eta, lam)
        eta_const = np.sqrt((eta - 2) / eta)

        # switching
        core = np.where(u < (1 - lam) / 2, 
                        (1 - lam) * stats.t.ppf(u / (1 - lam), eta), 
                        (1 + lam) * stats.t.ppf((u + lam) / (1 + lam), eta))
        
        return (1 / B) * (eta_const * core - A)
    

    def _cdf(self, x, eta, lam):
        # source: Tino Contino (DirtyQuant)

        # constants
        A, B, _ = self._get_ABC(eta, lam)
        numerator = np.sqrt(eta / (eta - 2)) * (B * x + A)

        return np.where(x < -A/B,
                    (1 - lam) * stats.t.cdf(numerator / (1 - lam), eta),
                    (1 + lam) * stats.t.cdf(numerator / (1 + lam), eta) - lam)
        

class UndefinedCDF(Marginal):


    def _set_cdf_ppf(self, loc, scale, *params, n = 1000):
        
        bounds = utils.find_x_bounds(loc, scale, self._pdf, *params)
        x_range = np.linspace(bounds[0], bounds[1], n)

        pdf_values = self._pdf(x_range, *params)
        
        # cumulative numerical integration
        cdf_values = cumulative_trapezoid(pdf_values, x_range, initial = 0)

        self.interp1d_cdf_func, self.interp1d_ppf_func = utils.build_cdf_interpolations(x_range, cdf_values)

    
    # redefining CDF and PPF methods to not only passing of parameters
    # CDF and PPF operate on set interpolations of numerical integration

    def cdf(self, x):
        # error handling
        return self._cdf(x)
    

    def _cdf(self, x):
        return self.interp1d_cdf_func(x)
    

    def ppf(self, u):
        # error handling
        return self._ppf(u)
    

    def _ppf(self, u):
        return self.interp1d_ppf_func(u)
    



class GaussianKDE(UndefinedCDF):
    def __init__(self, bw_method = None):
        # bw_method passed to scipy.stats.gaussian_kde
        # can be scalar, "scott", "silverman", or callable

        # explicitly defaulting to Scott if not provided--this SciPy's default too
        self.bw_method = bw_method if bw_method is not None else "scott"
        self.bw_method_desc = self.get_bw_method_desc(self.bw_method)

        super().__init__(None, model_name = "GaussianKDE", initial_param_guess = [], param_names = [],
                         param_bounds = [], params = [])
    
    def fit(self, x):
        # check that univariate
        self.kde = stats.gaussian_kde(x, bw_method = self.bw_method)

        # getting CDF and PPF interpolations
        
        self._prepare_cdf_ppf(np.min(x), np.max(x))

        # setting variables
        self.is_fit = True
        self.n = len(x)
        self.LL = self._log_likelihood(x)
        self.aic = np.nan
        self.bic = np.nan



    def _set_cdf_ppf(self, min_x, max_x):
        # overriding method defined in parent class

        x_range = np.linspace(min_x - 3 * self.kde.factor, max_x + 3 * self.kde.factor, 1000)    

        # cdf of Kernel Mixture is Mixture of component CDFs
        cdf_values = np.zeros_like(x_range)
        for xi in self.kde.dataset[0]:
            cdf_values += stats.norm.cdf(x_range, loc = xi, scale = self.kde.factor)

        cdf_values /= cdf_values[-1]
        self.interp1d_cdf_func, self.interp1d_ppf_func = utils.build_cdf_interpolations(x_range, cdf_values)


    def get_bw_method_desc(self, bw_method):
        if (bw_method == "scott") or (bw_method == "silverman"):
            return bw_method
        elif callable(bw_method):
            return "User Function"
        else:
            return "User Set"


    def _get_top_summary_table(self):
        now = datetime.now()
        top_left = [
            ("Model Name:", self.model_name), 
            ("Method:", self.bw_method_desc),("Num. Params:", 1), ("Num. Obs:", self.n),
            ("Date:", now.strftime("%a, %b %d %Y")),("Time:", now.strftime("%H:%M:%S")), ("", ""), ("", ""),
        ]

        top_right = [
            ("Bandwidth", utils.format_func(self.kde.factor, 10, 4)), ("Log-Likelihood:", utils.format_func(self.LL, 10, 4)), 
            ("AIC:", utils.format_func(self.aic, 10, 4)),
            ("BIC:", utils.format_func(self.bic, 10, 4)), (" ", " "), (" ", " "), (" ", " "),
            ("", ""),
        ]

        return top_left, top_right


    def _pdf(self, x):
        return self.kde(x)
    
    def _logpdf(self, x):
        return np.log(self._pdf(x))



class GeneralizedFamily(UndefinedCDF):

    def __init__(self, rv_obj, model_name, initial_param_guess, param_names, param_bounds, params):

        # preemptively fitting CDF and PPF interpolations
        # fitting with new params will override
        self._set_cdf_ppf(params[0], params[1], *params)
        super().__init__(rv_obj, model_name, initial_param_guess, param_names, param_bounds, params)

    def fit(self, x):
        # fit with MLE
        pass




class SkewedGeneralizedT(GeneralizedFamily):
    def __init__(mu = 0, sigma = 1, lam = 0, p = 2, q = 1e4, adj = 1e-4):
        super().__init__(None, model_name = "SkewedGeneralizedT", 
                         initial_param_guess = [0, 1, 0, 2, 1000], param_names = ["mu", "sigma", "lambda", "p", "q"],
                         param_bounds = [(-np.inf, np.inf), (adj, np.inf), (-1 + adj, 1 - adj), (adj, np.inf), (adj, np.inf)],
                         params = [mu, sigma, lam, p, q])


    def _pdf(self, x, mu, sigma, lam, p, q):

        # common terms
        beta_term1 = beta(1/p, q)
        beta_term2 = beta(2/q, q - 1/p)
    
        # v variable
        v1 = np.power(q, -1/p)
        v2 = 1 + 3 * np.power(lam, 2)
        v3 = beta(3/p, q - 2/p) / beta_term1
        v4 = (4 * np.power(lam, 2)) * np.power(beta_term2 / beta_term1, 2)
        v = v1 / np.sqrt(v2 * v3 - v4)
    
        # m variable
        sigma_term = 2 * sigma * v * np.power(q, 1/p)
        m = lam * sigma_term * beta_term2 / beta_term1
    
    
        # the final pdf
        x_term = x - mu + m
        pdf1 = sigma_term * beta_term1
        pdf2 = np.power(np.abs(x_term), p)
        pdf3 = q * np.power(v * sigma, p) * np.power(1 + lam * np.sign(x_term), p)
    
        return p / (pdf1 * np.power(1 + pdf2 / pdf3, 1/p + q))
    




# standard t / skewed t to help pin down df



