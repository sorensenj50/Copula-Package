import utils
import base
from mixture import Mixture

import numpy as np
from scipy import stats
from scipy import special
from datetime import datetime


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
            ("Esimation Method:", self.estimation_method),("Num. Params:", self.k), ("Num. Obs:", self.n),
            ("Date:", now.strftime("%a, %b %d %Y")),("Time:", now.strftime("%H:%M:%S")), ("", ""), ("", ""),
        ]

        top_right = [
            ("Log-Likelihood:", utils.format_func(self.LL, 10, 4)), ("AIC:", utils.format_func(self.aic, 10, 4)),
            ("BIC:", utils.format_func(self.bic, 10, 4)), ("Skewness:", utils.format_func(self.skewness, 10, 4)), 
            ("Excess Kurtosis:", utils.format_func(self.kurtosis, 10, 4)), ("95% VaR:", utils.format_func(self.var, 10, 4)),
            ("95% CVaR:", utils.format_func(self.cvar, 10, 4)), ("", ""), ("", ""),
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
        # default is Monte Carlo simulation, which can be fairly efficient
        # child classes can override this with analytic solutions

        seed = 0 # fixed seed for reproducibility
        n = 5000 # speed accuracy tradeoff
        return utils.monte_carlo_cvar(self, n = n, seed = seed, alpha = alpha)
    


class Normal(Marginal):
    def __init__(self, mu = 0, sigma = 1, adj = 1e-4):

        # the order of these params depends on SciPy
        super().__init__(stats.norm, model_name = "Normal", family_name = "Parametric", initial_param_guess = [0, 1], 
                        param_bounds = [(-np.inf, np.inf), (adj, np.inf)], param_names = ["mu", "sigma"],
                        params = [mu, sigma])
        

    def _params_to_skewness(self, mu, sigma):
        return 0
    

    def _params_to_kurtosis(self, mu, sigma):
        return 0
    

    def _params_to_cvar(self, mu, sigma, alpha = 0.95):
        # Matthew Norton et al 2019
        # not passing params to _pdf and _ppf: standard normal
        return mu - sigma * (self._pdf(self._ppf(alpha))) / (1 - alpha)
    

# private
class CenteredNormal(Marginal):
    def __init__(self, sigma = 1, adj = 1e-4):

        super().__init__(stats.norm, model_name = "CenteredNormal", family_name = "Parametric",
                         initial_param_guess = [1], param_bounds = [(adj, np.inf)],
                         param_names = ["sigma"], params = [sigma])
        
    # relying on core methods implemented generically by parent class
    # imposing mu = 0 constraint
    def _pdf(self, x, sigma):
        return super()._pdf(x, 0, sigma)
    

    def _logpdf(self, x, sigma):
        return super()._logpdf(x, 0, sigma)
    

    def _cdf(self, x, sigma):
        return super()._cdf(x, 0, sigma)
    

    def _ppf(self, q, sigma):
        return super()._ppf(q, 0, sigma)
    

    def fit(self, x, robust_cov = True):
        # input validation
        valid_x = self._handle_input(x)

        # relying on scipy implementation of fit
        opt_params = self.rv_obj.fit(x, floc = 0)
        self._post_process_fit(valid_x, np.array([opt_params[1]]), 
                               self._get_obj_func(valid_x), robust_cov = robust_cov)
    
    def _params_to_skewness(self, sigma):
        return 0
    

    def _params_to_kurtosis(self, sigma):
        return 0
    

    def _params_to_cvar(self, sigma, alpha = 0.95):
        # Mattew Norton et al 2019
        # _pdf and _ppf use standard normal

        return - sigma * (self._pdf(self._ppf(alpha, 1), 1)) / (1 - alpha)
        
    

class StudentsT(Marginal):
    def __init__(self, df = 30, mu = 0, sigma = 1, adj = 1e-4):
       
        # the order of these params depends on SciPy
        super().__init__(stats.t, model_name = "StudentsT", family_name = "Parametric", initial_param_guess = [30, 0, 1], 
                        param_bounds = [(1, np.inf), (-np.inf, np.inf), (adj, np.inf)], param_names = ["df", "mu", "sigma"],
                        params = [df, mu, sigma])
        

    def _params_to_skewness(self, df, mu, sigma):
        return 0 if df > 3 else np.nan
        

    def _params_to_kurtosis(self, df, mu, sigma):
        if df > 4:
            return 6 / (df - 4)
        elif df > 2 and df <= 4:
            return np.inf
        else:
            return np.nan
    
    
    def _params_to_cvar(self, df, mu, sigma, alpha = 0.95):
        # Nortan (2019) and Carol Alexander IV.2.88å
        
        term1 = (df + stats.t.ppf(alpha, df) ** 2) / ((df - 1) * (1 - alpha))
        term2 = stats.t.pdf(stats.t.ppf(alpha, df), df)

        return mu - sigma * term1 * term2



class NormalMixture(Marginal, Mixture):
    def __init__(self, p1 = 0.5, mu1 = 0, mu2 = 0, sigma1 = 1, sigma2 = 1, adj = 1e-4):
        p1 = self._normalize_p(p1)

        Marginal.__init__(self, None, model_name = "NormalMixture", family_name = "Parametric Mixture",
                         initial_param_guess = [0.5, 0, 0, 1, 1], 
                         param_bounds = [(0, 1), (-np.inf, np.inf), (-np.inf, np.inf), (adj, np.inf), (adj, np.inf)],
                         param_names = ["p1", "mu1", "mu2", "sigma1", "sigma2"], params = [p1, mu1, mu2, sigma1, sigma2])

        Mixture.__init__(self, Normal())


    def _get_random_params(self, n, rng, data):
        # simple bootstrap to get random mean and standard deviation
        # for initialization of EM algorithm

        num_obs = data.shape[0]

        # smaller than sample size to increase randomness
        bootstrap_size = np.ceil(np.sqrt(num_obs))
        random_indices = rng.integers(num_obs, size = (n, int(bootstrap_size))) 
        random_params = np.zeros(shape = (n, 2)) # to be filled

        for i in range(n):
            bootstrap_selection = data[random_indices[i]]
            random_params[i] = np.array([np.mean(bootstrap_selection), np.std(bootstrap_selection)])

        return random_params
    

    def fit(self, x, seed = None, n_init = 20, tol = 1e-4, max_iter = 100, optimizer = "Powell"):
        # input validation

        LL, p1, mu1, sigma1, mu2, sigma2 = self._run_em_algo_multi(x, seed = seed, n_init = n_init, tol = tol,
                                                                   max_iter = max_iter, optimizer = optimizer)
        # reordering parameters to order in init
        self._set_params((p1, mu1, mu2, sigma1, sigma2))
        self._mini_post_process_fit(LL, x.shape[0])


    def _pdf(self, x, p1, mu1, mu2, sigma1, sigma2):
        return self._mixture_pdf(p1, (mu1, sigma1), (mu2, sigma2), x)
    

    def _logpdf(self, x, p1, mu1, mu2, sigma1, sigma2):
        return np.log(self._pdf(x, p1, mu1, mu2, sigma1, sigma2))
    

    def _cdf(self, x, p1, mu1, mu2, sigma1, sigma2):
        return self._mixture_cdf(p1, (mu1, sigma1), (mu2, sigma2), x)
    
    
    def _ppf(self, q, *params):
        # Brent Q solver
        a, b = self._get_lower_upper_bound(*params)
        return utils.solve_for_ppf(self._cdf, q, a, b, *params)
    

    def _get_lower_upper_bound(self, p1, mu1, mu2, sigma1, sigma2, Z_factor = 5):
        # finds boundaries needed for Brent Q solver of ppf
        # p1 is unused

        lower_bound = min(mu1 - Z_factor * sigma1, mu2 - Z_factor * sigma2)
        upper_bound = max(mu1 + Z_factor * sigma1, mu2 + Z_factor * sigma2)
        return lower_bound, upper_bound
    

    def simulate(self, n = 1000, seed = None):
        # this is faster and simpler than default of mixture PPF

        p1, mu1, mu2, sigma1, sigma2 = self.params
        rng = np.random.default_rng(seed = seed)

        # drawing parameters using mixture probabilities
        param_draws = rng.choice([0, 1], p = [p1, 1 - p1], size = n)
        mu_params = np.where(param_draws == 0, mu1, mu2)
        sigma_params = np.where(param_draws == 0, sigma1, sigma2)

        return np.array([rng.normal(loc = mu_params[i], scale = sigma_params[i]) for i in range(n)])
    

    def _params_to_mu(self, p1, mu1, mu2, sigma1, sigma2):
        # linearity of expectation
        return p1 * mu1 + (1 - p1) * mu2 

    
    def _params_to_variance(self, p1, mu1, mu2, sigma1, sigma2):
        # Fruhwirth-Shnatter (2006) page 11

        mu = self._params_to_mu(p1, mu1, mu2, sigma1, sigma2)
        part_1 = np.power(sigma1, 2) + np.power(mu1, 2)
        part_2 = np.power(sigma2, 2) + np.power(mu2, 2)

        return p1 * part_1 + (1 - p1) * part_2 - np.power(mu, 2)

    
    def _params_to_skewness(self, p1, mu1, mu2, sigma1, sigma2):
        # Fruhwirth-Shnatter (2006) page 11

        mu = self._params_to_mu(p1, mu1, mu2, sigma1, sigma2)
        variance = self._params_to_variance(p1, mu1, mu2, sigma1, sigma2)
        
        part_1 = p1 * (np.power(mu1 - mu, 2) + 3 * np.power(sigma1, 2)) * (mu1 - mu)
        part_2 = (1 - p1) * (np.power(mu2 - mu, 2) + 3 * np.power(sigma2, 2)) * (mu2 - mu)
        return (part_1 + part_2) / np.power(variance, 3/2)
    

    def _params_to_kurtosis(self, p1, mu1, mu2, sigma1, sigma2):
        # Fruhwirth-Shnatter (2006) page 11

        mu = self._params_to_mu(p1, mu1, mu2, sigma1, sigma2)
        variance = self._params_to_variance(p1, mu1, mu2, sigma1, sigma2)

        part_1 = np.power(mu1 - mu, 4) + (6 * np.power(mu1 - mu, 2) * np.power(sigma1, 2)) + 3 * np.power(sigma1, 4)
        part_2 = np.power(mu2 - mu, 4) + (6 * np.power(mu2 - mu, 2) * np.power(sigma2, 2)) + 3 * np.power(sigma2, 4)
        fourth_central_moment = p1 * part_1 + (1 - p1) * part_2

        # 4th standard moment, excess kurtosis
        return fourth_central_moment / np.power(variance, 2) - 3
    
    
    def _params_to_cvar(self, p1, mu1, mu2, sigma1, sigma2, alpha = 0.95):
        # Broda and Paolella (2011) Section 2.3.2
        # See also "Estimation methods for expected shortfall" by University of Manchester
        
        # quantile level
        var = self._params_to_var(p1, mu1, mu2, sigma1, sigma2, alpha = alpha)

        # Z-Scores given the two components
        c1 = (var - mu1) / sigma1; c2 = (var - mu2) / sigma2

        part_1 = p1 * stats.norm.cdf(c1) / (1 - alpha) * (mu1 - sigma1 * stats.norm.pdf(c1) / stats.norm.cdf(c1))
        part_2 = (1 - p1) * stats.norm.cdf(c2) / (1 - alpha) * (mu2 - sigma2 * stats.norm.pdf(c2) / stats.norm.cdf(c2))
        
        return part_1 + part_2




class NormalVarianceMixture(Marginal, Mixture):
    def __init__(self, p1 = 0.5, sigma1 = 1, sigma2 = 1, adj = 1e-4):

        p1 = self._normalize_p(p1)

        Marginal.__init__(self, None, model_name = "NormalVarianceMixture", family_name = "Parametric Mixture",
                          initial_param_guess = [0.5, 1, 1], param_bounds = [(0, 1), (adj, np.inf), (adj, np.inf)],
                          param_names = ["p1", "sigma1", "sigma2"], params = [p1, sigma1, sigma2])
        
        Mixture.__init__(self, CenteredNormal())

    
    def _get_random_params(self, n, rng, data):
        # like NormalMixture, bootstrap to get random standard deviations
        # for initialization of EM algo

        num_obs = data.shape[0]
        bootstrap_size = np.ceil(np.sqrt(num_obs))
        random_indices = rng.integers(num_obs, size = (n, int(bootstrap_size)))
        random_params = np.zeros(shape = n)

        for i in range(n):
            bootstrap_selection = data[random_indices[i]]
            random_params[i] = np.std(bootstrap_selection)

        return random_params
    
    def fit(self, x, seed = None, n_init = 20, tol = 1e-4, max_iter = 100, optimizer = "Powell"):
        # input validation

        # running EM algo
        LL, p1, sigma1, sigma2 = self._run_em_algo_multi(x, seed = seed, n_init = n_init, tol = tol, 
                                                         max_iter = max_iter, optimizer = optimizer)
        
        self._set_params((p1, sigma1, sigma2))
        self._mini_post_process_fit(LL, x.shape[0])
    

    def _pdf(self, x, p1, sigma1, sigma2):
        # linear mix
        return self._mixture_pdf(p1, (sigma1,), (sigma2,), x)
    

    def _cdf(self, x, p1, sigma1, sigma2):
        # linear mix
        return self._mixture_cdf(p1, (sigma1,), (sigma2,), x)
    

    def _ppf(self, q, *params):
        # Brent Q solver
        a, b = self._get_lower_upper_bound(*params)
        return utils.solve_for_ppf(self._cdf, q, a, b, *params)
    

    def _get_lower_upper_bound(self, p1, sigma1, sigma2, Z_factor = 5):
        # finds boundaries needed for Brent Q solving of inverse cdf
        # p1 is unused

        biggest_sigma = max(sigma1, sigma2)
        return Z_factor * biggest_sigma, -Z_factor * biggest_sigma
    

    def simulate(self, n = 1000, seed = None):
        # potentially faster than relying on solver for large n
        # we can sidestep the ppf by
        
        p1, sigma1, sigma2 = self.params
        rng = np.random.default_rng(seed = seed)

        param_draws = rng.choice([0, 1], p = [p1, 1 - p1], size = n)
        sigmas = np.where(param_draws == 0, sigma1, sigma2)

        return np.array([rng.normal(0, sigmas[i]) for i in range(n)])
    

    def _params_to_variance(self, p1, sigma1, sigma2):
        # Carol Alexander I.3.45
        # linear mix
        return p1 * np.power(sigma1, 2) + (1 - p1) * np.power(sigma2, 2)
    

    def _params_to_skewness(self, *params):
        # always zero for variance mixture
        # can only be non-zero in normal mixture by shifting mean
        return 0
    

    def _params_to_kurtosis(self, p1, sigma1, sigma2):
        # Carol Alexander I.3.46

        variance = self._params_to_variance(p1, sigma1, sigma2)
        numerator = p1 * np.power(sigma1, 4) + (1 - p1) * np.power(sigma2, 4)
        
        # -3 to get in excess kurtosis
        return 3 * numerator / np.power(variance, 2) - 3
    

    def _params_to_cvar(self, p1, sigma1, sigma2, alpha = 0.95):
        # Carol Alexander IV.2.89
        # I couldn't get her formula for the general NormalMixture to work, but this works for the VarianceMixture case

        var = self._params_to_var(p1, sigma1, sigma2)
        term_1 = p1 * sigma1 * stats.norm.pdf(var / sigma1)
        term_2 = (1 - p1) * sigma2 * stats.norm.pdf(var / sigma2)

        return -(term_1 + term_2) / (1 - alpha)
        



class StandardSkewedT(Marginal):
    # Hansen 1994

    def __init__(self, eta = 30, lam = 0, adj = 1e-4):
        super().__init__(None, model_name = "StandardSkewedT", family_name = "Parametric",
                         initial_param_guess = [30, 0], param_names = ["eta", "lam"],
                         param_bounds = [(2 + adj, np.inf), (-1 + adj, 1 - adj)],
                         params = [eta, lam])
        

    def _get_ABC(self, eta, lam):
        C = special.gamma((eta + 1) / 2) / (np.sqrt(np.pi * (eta - 2))  * special.gamma(eta / 2))
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


    def _ppf(self, q, eta, lam):

        # source: Tino Contino (DirtyQuant)

        # constants
        A, B, _ = self._get_ABC(eta, lam)
        eta_const = np.sqrt((eta - 2) / eta)

        # switching
        core = np.where(q < (1 - lam) / 2, 
                        (1 - lam) * stats.t.ppf(q / (1 - lam), eta), 
                        (1 + lam) * stats.t.ppf((q + lam) / (1 + lam), eta))
        
        return (1 / B) * (eta_const * core - A)
    

    def _cdf(self, x, eta, lam):
        # source: Tino Contino (DirtyQuant)

        # constants
        A, B, _ = self._get_ABC(eta, lam)
        numerator = np.sqrt(eta / (eta - 2)) * (B * x + A)

        return np.where(x < -A/B,
                    (1 - lam) * stats.t.cdf(numerator / (1 - lam), eta),
                    (1 + lam) * stats.t.cdf(numerator / (1 + lam), eta) - lam)
    

    
    def fit(self, x, optimizer = "Powell", robust_cov = True):
        # error handling
        valid_x = self._handle_input(x)

        f = self._get_objective_func(valid_x)
        opt_results = self._fit(f, self.initial_param_guess, self.param_bounds, optimizer = optimizer)
        self._post_process_fit(valid_x, opt_results.x, self._get_objective_func(x), robust_cov = robust_cov)


    def _params_to_skewness(self, eta, lam):
        return np.nan
    

    def _params_to_kurtosis(self, eta, lam):
        return np.nan




class GaussianKDE(Marginal):
    def __init__(self, bw_method = None, Z_factor = 5, interpolation_n = 2000, monte_carlo_n = 10_000, monte_carlo_seed = None):
        # bw_method passed to scipy.stats.gaussian_kde
        # can be scalar, "scott", "silverman", or callable

        super().__init__(None, model_name = "GaussianKDE", family_name = "Non-Parametric", 
                         initial_param_guess = [], param_names = [],
                         param_bounds = [], params = [])

        # explicitly defaulting to Scott if not provided--this SciPy's default too
        self.bw_method = bw_method if bw_method is not None else "scott"
        self.kde_factor = np.nan
        self.estimation_method = self.get_bw_method_desc(self.bw_method)
        self.Z_factor = Z_factor
        self.interpolation_n = interpolation_n
        self.monte_carlo_n = monte_carlo_n
        self.monte_carlo_seed = monte_carlo_seed

        # will be estimated via monte carlo after fit
        self._skew = np.nan
        self._kurtosis = np.nan
        self._cvar = np.nan

        
    
    def fit(self, x):
        # check that univariate
        self.kde = stats.gaussian_kde(x, bw_method = self.bw_method)

        # getting CDF and PPF interpolations
        self._set_cdf_ppf(np.min(x), np.max(x))

        # setting variables
        self.is_fit = True
        self.n = len(x)
        self.LL = self._log_likelihood(x)
        self.kde_factor = self.kde.factor

        self._skew, self._kurtosis, self._cvar = self._monte_carlo_stats()


    def _set_cdf_ppf(self, min_x, max_x):
        # overriding method defined in parent class
        # Z: number of standard deviatons above or below to evaluate range

        x_range = np.linspace(min_x - self.Z_factor * self.kde.factor, 
                              max_x + self.Z_factor * self.kde.factor, 
                              num = self.interpolation_n)    

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
            return "user callable"
        else:
            return "user set"


    def _pdf(self, x):
        return self.kde(x)
    

    def _logpdf(self, x):
        return np.log(self._pdf(x))
    

    def cdf(self, x):
        # error handling
        return self._cdf(x)
    

    def _cdf(self, x):
        return self.interp1d_cdf_func(x)
    

    def ppf(self, q):
        # error handling
        return self._ppf(q)
    

    def _ppf(self, q):
        return self.interp1d_ppf_func(q)
    

    def _monte_carlo_stats(self):
        X = self.simulate(n = self.monte_carlo_n, seed = self.monte_carlo_seed)

        skewness = stats.skew(X); kurtosis = stats.kurtosis(X)

        quantile = np.quantile(X, 0.05)
        cvar_filter = X <= quantile
        cvar = np.mean(X[cvar_filter])

        return skewness, kurtosis, cvar
    

    def _params_to_skewness(self, *params):
        # Monte Carlo
        return self._skew
    
    
    def _params_to_kurtosis(self, *params):
        # Monte Carlo
        return self._kurtosis
    

    def _params_to_cvar(self, *params, alpha=0.95):
        # alpha is unused
        return self._cvar
    

    def _get_extra_text(self):
        return ["PPF Estimated via Numerical Interpolation of CDF",
                "Skewness, Kurtosis, and CVaR Estimated via Monte Carlo"]
    







