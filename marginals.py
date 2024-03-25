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
                               self._get_objective_func(valid_x), robust_cov = robust_cov)

    
    def simulate(self, n = 1000, seed = None):
        # rely on Scipy
        rng = np.random.default_rng(seed = seed)
        u = rng.uniform(size = n)
        return self.ppf(u)
    

    def _get_top_summary_table(self):
        now = datetime.now()
        top_left = [
            ("Model Name:", self.model_name), ("Model Family:", self.family_name),
            ("Estimation Method:", self.estimation_method),("Num. Params:", self.k), ("Num. Obs:", self.n),
            ("Date:", now.strftime("%a, %b %d %Y")),("Time:", now.strftime("%H:%M:%S")), ("", ""), ("", ""),
        ]

        top_right = [
            ("Log-Likelihood:", utils.format_func(self.LL, 10, 4)), ("AIC:", utils.format_func(self.aic, 10, 4)),
            ("BIC:", utils.format_func(self.bic, 10, 4)), ("Skewness:", utils.format_func(self.skewness, 10, 4)), 
            ("Kurtosis", utils.format_func(self.kurtosis, 10, 4)), ("Entropy:", utils.format_func(self.entropy, 10, 4)),
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
    def entropy(self):
        return self._params_to_entropy(*self.params)
    

    @property
    def cvar(self):
        return self._params_to_cvar(*self.params)
    
    
    def _params_to_skewness(self, *params):
        raise NotImplementedError
    
    
    def _params_to_kurtosis(self, *params):
        raise NotImplementedError
    
    
    def _params_to_entropy(self, *params):
        raise NotImplementedError
    

    def _params_to_cvar(self, *params):
        raise NotImplementedError


    



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
    

    def _params_to_entropy(self, mu, sigma):
        return 1/2 * np.log(2 * np.pi * np.e * sigma ** 2)
    

    def _params_to_cvar(self, mu, sigma, alpha = 0.95):
        # Matthew Norton et al 2019
        return mu + sigma * (self._pdf(self._ppf(alpha))) / (1 - alpha)
        
    

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
        

    def _params_to_entropy(self, df, mu, sigma):
        # from stack overflow / wikipedia
        # check this

        df_term = (df + 1) / 2
        digamma_term = special.digamma(df_term) - special.digamma(df / 2)
        beta_term = np.log(np.sqrt(df) * special.beta(1/2, df / 2))

        return beta_term + df_term * digamma_term
    
    
    def _params_to_cvar(self, df, mu, sigma, alpha = 0.95):
        
        term1 = (df + stats.t.ppf(alpha, df) ** 2) / ((df - 1) * (1 - alpha))
        term2 = stats.t.pdf(stats.t.ppf(alpha, df), df)

        return mu + sigma * term1 * term2



class NormalMixture(Marginal, Mixture):
    def __init__(self, p1 = 0.5, mu1 = 0, mu2 = 0, sigma1 = 1, sigma2 = 1, ppf_method = "solver", adj = 1e-4):
        p1 = self._normalize_p(p1)

        Marginal.__init__(self, None, model_name = "NormalMixture", family_name = "Parametric Mixture",
                         initial_param_guess = [0.5, 0, 0, 1, 1], 
                         param_bounds = [(0, 1), (-np.inf, np.inf), (-np.inf, np.inf), (adj, np.inf), (adj, np.inf)],
                         param_names = ["p1", "mu1", "mu2", "sigma1", "sigma2"], params = [p1, mu1, mu2, sigma1, sigma2])

        Mixture.__init__(self, Normal())

        self._ppf_method = ppf_method
        self._ppf_func = self._ppf_interp if ppf_method == "interpolation" else self._ppf_solve
        self._set_ppf(p1, mu1, mu2, sigma1, sigma2)


    def _get_random_params(self, n, rng, data):
        # simple bootstrap to get random mean and standard deviation
        # for initialization of EM algorithm

        num_obs = data.shape[0]

        # bootstrap size is smaller than sample size to increase randomness in initialialization
        random_indices = rng.integers(num_obs, size = (n, int(np.sqrt(num_obs)))) 
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
        self._set_ppf(p1, mu1, mu2, sigma1, sigma2)


    def _get_lower_upper_bound(self, p1, mu1, mu2, sigma1, sigma2, Z_factor = 5):
        # probability is unused
        lower_bound = min(mu1 - Z_factor * sigma1, mu2 - Z_factor * sigma2)
        upper_bound = max(mu1 + Z_factor * sigma1, mu2 + Z_factor * sigma2)
        return lower_bound, upper_bound


    def _set_ppf(self, p1, mu1, mu2, sigma1, sigma2, Z_factor = 5, num = 1000):
        if self._ppf_method == "solver":
            # can skip because we using brentq solver instead of interpolating
            return


        # Z standard deviations above or below the most extreme dist in normal Mixture
        lower, upper = self._get_lower_upper_bound(p1, mu1, mu2, sigma1, sigma2, Z_factor = Z_factor)
        x_range = np.linspace(lower, upper, num)
        cdf_vals = self._cdf(x_range, p1, mu1, mu2, sigma1, sigma2) # evaluating CDF with parameters across x range

        # returns CDF and PPF interpolation functions, but we don't need CDF
        self._interp1d_cdf_func, self._interp1d_ppf_func = utils.build_cdf_interpolations(x_range, cdf_vals)


    def _pdf(self, x, p1, mu1, mu2, sigma1, sigma2):
        return self._mixture_pdf(p1, (mu1, sigma1), (mu2, sigma2), x)
    

    def _logpdf(self, x, p1, mu1, mu2, sigma1, sigma2):
        return np.log(self._pdf(x, p1, mu1, mu2, sigma1, sigma2))
    

    def _cdf(self, x, p1, mu1, mu2, sigma1, sigma2):
        return self._mixture_cdf(p1, (mu1, sigma1), (mu2, sigma2), x)
    
    
    def _ppf(self, q, *params):
        return self._ppf_func(q, *params)
    

    def _ppf_solve(self, q, *params):
        # uses Brentq solver to find inverse CDF
        a, b = self._get_lower_upper_bound(*params)
        return utils.solve_for_ppf(self._cdf, q, a, b, *params)
    

    def _ppf_interp(self, q, *params):
        # uses interpolation. params are unused
        # more efficient for simulation, slightly less accurate than brentq solver
        return self._interp1d_ppf_func(q)





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




class GaussianKDE(Marginal):
    def __init__(self, bw_method = None):
        # bw_method passed to scipy.stats.gaussian_kde
        # can be scalar, "scott", "silverman", or callable

        # explicitly defaulting to Scott if not provided--this SciPy's default too
        self.bw_method = bw_method if bw_method is not None else "scott"
        self.kde_factor = np.nan
        self.bw_method_desc = self.get_bw_method_desc(self.bw_method)

        super().__init__(None, model_name = "GaussianKDE", family_name = "Non-Parametric", initial_param_guess = [], param_names = [],
                         param_bounds = [], params = [])
    
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


    def _set_cdf_ppf(self, min_x, max_x, Z = 4):
        # overriding method defined in parent class
        # Z: number of standard deviatons above or below to evaluate range

        x_range = np.linspace(min_x - Z * self.kde.factor, max_x + Z * self.kde.factor, 1000)    

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


    def _get_top_summary_table(self):
        now = datetime.now()
        top_left = [
            ("Model Name:", self.model_name), ("Model Family:", self.family_name),
            ("Method:", self.bw_method_desc),("Num. Params:", np.nan), ("Num. Obs:", self.n),
            ("Date:", now.strftime("%a, %b %d %Y")),("Time:", now.strftime("%H:%M:%S")),
        ]

        top_right = [
            ("Log-Likelihood:", utils.format_func(self.LL, 10, 4)), ("Bandwidth:", utils.format_func(self.kde_factor, 10, 4)), 
            (" ", " "), (" ", " "), ("", ""), ("", ""), ("", ""),
        ]

        return top_left, top_right


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