import numpy as np
from scipy import stats

from .parametric import CenteredNormal, Normal
from .marginals import Marginal
from mixture import Mixture
import utils



class NormalMix(Mixture, Marginal):
    def __init__(self, p1 = 0.5, loc1 = 0, loc2 = 0, scale1 = 1, scale2 = 1, adj = 1e-4):
        p1 = self._normalize_p(p1)

        Marginal.__init__(self, None, model_name = "NormalMix", family_name = "Finite Mixture",
                         initial_param_guess = [0.5, 0, 0, 1, 1], 
                         param_bounds = [(0, 1), (-np.inf, np.inf), (-np.inf, np.inf), (adj, np.inf), (adj, np.inf)],
                         param_names = ["p1", "mu1", "mu2", "sigma1", "sigma2"], params = [p1, loc1, loc2, scale1, scale2])

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


    def _pdf(self, x, p1, loc1, loc2, scale1, scale2):
        return self._mixture_pdf(p1, (loc1, scale1), (loc2, scale2), x)
    

    def _logpdf(self, x, p1, loc1, loc2, scale1, scale2):
        return np.log(self._pdf(x, p1, loc1, loc2, scale1, scale2))
    

    def _cdf(self, x, p1, loc1, loc2, scale1, scale2):
        return self._mixture_cdf(p1, (loc1, scale1), (loc2, scale2), x)
    
    
    def _ppf(self, q, *params):
        # Brent Q solver
        a, b = self._get_lower_upper_bound(*params)
        return utils.solve_for_ppf(self._cdf, q, a, b, *params)
    

    def _get_lower_upper_bound(self, p1, loc1, loc2, scale1, scale2, Z_factor = 5):
        # finds boundaries needed for Brent Q solver of ppf
        # p1 is unused

        lower_bound = min(loc1 - Z_factor * scale1, loc2 - Z_factor * scale2)
        upper_bound = max(loc1 + Z_factor * scale1, loc2 + Z_factor * scale2)
        return lower_bound, upper_bound
    

    def simulate(self, n = 1000, seed = None):
        # this is faster and simpler than default of mixture PPF

        p1, loc1, loc2, scale1, scale2 = self.params
        rng = np.random.default_rng(seed = seed)

        # drawing parameters using mixture probabilities
        param_draws = rng.choice([0, 1], p = [p1, 1 - p1], size = n)
        loc_params = np.where(param_draws == 0, loc1, loc2)
        scale_params = np.where(param_draws == 0, scale1, scale2)

        return np.array([rng.normal(loc = loc_params[i], scale = scale_params[i]) for i in range(n)])
    

    def _params_to_mean(self, p1, loc1, loc2, scale1, scale2):
        # linearity of expectation
        return p1 * loc1 + (1 - p1) * loc2 

    
    def _params_to_variance(self, p1, loc1, loc2, scale1, scale2):
        # Fruhwirth-Shnatter (2006) page 11

        mu = self._params_to_mean(p1, loc1, loc2, scale1, scale2)
        part_1 = np.power(scale1, 2) + np.power(loc1, 2)
        part_2 = np.power(scale2, 2) + np.power(loc2, 2)

        return p1 * part_1 + (1 - p1) * part_2 - np.power(mu, 2)

    
    def _params_to_skewness(self, p1, loc1, loc2, scale1, scale2):
        # Fruhwirth-Shnatter (2006) page 11

        mu = self._params_to_mean(p1, loc1, loc2, scale1, scale2)
        variance = self._params_to_variance(p1, loc1, loc2, scale1, scale2)
        
        part_1 = p1 * (np.power(loc1 - mu, 2) + 3 * np.power(scale1, 2)) * (loc1 - mu)
        part_2 = (1 - p1) * (np.power(loc2 - mu, 2) + 3 * np.power(scale2, 2)) * (loc2 - mu)
        return (part_1 + part_2) / np.power(variance, 3/2)
    

    def _params_to_kurtosis(self, p1, loc1, loc2, scale1, scale2):
        # Fruhwirth-Shnatter (2006) page 11

        mu = self._params_to_mean(p1, loc1, loc2, scale1, scale2)
        variance = self._params_to_variance(p1, loc1, loc2, scale1, scale2)

        part_1 = np.power(loc1 - mu, 4) + (6 * np.power(loc1 - mu, 2) * np.power(scale1, 2)) + 3 * np.power(scale1, 4)
        part_2 = np.power(loc2 - mu, 4) + (6 * np.power(loc2 - mu, 2) * np.power(scale2, 2)) + 3 * np.power(scale2, 4)
        fourth_central_moment = p1 * part_1 + (1 - p1) * part_2

        # 4th standard moment, excess kurtosis
        return fourth_central_moment / np.power(variance, 2) - 3
    
    
    def _params_to_cvar(self, p1, loc1, loc2, scale1, scale2, alpha = 0.95):
        # Broda and Paolella (2011) Section 2.3.2
        # See also "Estimation methods for expected shortfall" by University of Manchester
        
        # quantile level
        var = self._params_to_var(p1, loc1, loc2, scale1, scale2, alpha = alpha)

        # Z-Scores given the two components
        c1 = (var - loc1) / scale1; c2 = (var - loc2) / scale2

        part_1 = p1 * stats.norm.cdf(c1) / (1 - alpha) * (loc1 - scale1 * stats.norm.pdf(c1) / stats.norm.cdf(c1))
        part_2 = (1 - p1) * stats.norm.cdf(c2) / (1 - alpha) * (loc2 - scale2 * stats.norm.pdf(c2) / stats.norm.cdf(c2))
        
        return part_1 + part_2



class NormalVarianceMix(Mixture, Marginal):
    def __init__(self, p1 = 0.5, scale1 = 1, scale2 = 1, adj = 1e-4):

        p1 = self._normalize_p(p1)

        Marginal.__init__(self, None, model_name = "NormalVarianceMix", family_name = "Finite Mixture",
                          initial_param_guess = [0.5, 1, 1], param_bounds = [(0, 1), (adj, np.inf), (adj, np.inf)],
                          param_names = ["p1", "scale1", "scale2"], params = [p1, scale1, scale2])
        
        Mixture.__init__(self, CenteredNormal())

    
    def _get_random_params(self, n, rng, data):
        # like NormalMixture, bootstrap to get random standard deviations
        # for initialization of EM algo

        num_obs = data.shape[0]
        bootstrap_size = np.ceil(np.sqrt(num_obs))
        random_indices = rng.integers(num_obs, size = (n, int(bootstrap_size)))
        random_params = np.zeros(shape = (n, 1))

        for i in range(n):
            bootstrap_selection = data[random_indices[i]]
            random_params[i] = np.array([np.std(bootstrap_selection)])

        return random_params
    

    def fit(self, x, seed = None, n_init = 20, tol = 1e-4, max_iter = 100, optimizer = "Powell"):
        # input validation

        # running EM algo
        LL, p1, sigma1, sigma2 = self._run_em_algo_multi(x, seed = seed, n_init = n_init, tol = tol, 
                                                         max_iter = max_iter, optimizer = optimizer)
        
        self._set_params((p1, sigma1, sigma2))
        self._mini_post_process_fit(LL, x.shape[0])
    

    def _pdf(self, x, p1, scale1, scale2):
        # linear mix
        return self._mixture_pdf(p1, (scale1,), (scale2,), x)
    

    def _cdf(self, x, p1, scale1, scale2):
        # linear mix
        return self._mixture_cdf(p1, (scale1,), (scale2,), x)
    

    def _ppf(self, q, *params):
        # Brent Q solver
        a, b = self._get_lower_upper_bound(*params)
        return utils.solve_for_ppf(self._cdf, q, a, b, *params)
    

    def _get_lower_upper_bound(self, p1, scale1, scale2, Z_factor = 5):
        # finds boundaries needed for Brent Q solving of inverse cdf
        # p1 is unused

        biggest_sigma = max(scale1, scale2)
        return Z_factor * biggest_sigma, -Z_factor * biggest_sigma
    

    def simulate(self, n = 1000, seed = None):
        # potentially faster than relying on solver for large n
        # we can sidestep the ppf by
        
        p1, scale1, scale2 = self.params
        rng = np.random.default_rng(seed = seed)

        param_draws = rng.choice([0, 1], p = [p1, 1 - p1], size = n)
        sigmas = np.where(param_draws == 0, scale1, scale2)

        return np.array([rng.normal(0, sigmas[i]) for i in range(n)])
    

    def _params_to_variance(self, p1, scale1, scale2):
        # Carol Alexander I.3.45
        # linear mix
        return p1 * np.power(scale1, 2) + (1 - p1) * np.power(scale2, 2)
    

    def _params_to_skewness(self, *params):
        # always zero for variance mixture
        # can only be non-zero in normal mixture by shifting mean
        return 0
    

    def _params_to_kurtosis(self, p1, scale1, scale2):
        # Carol Alexander I.3.46

        variance = self._params_to_variance(p1, scale1, scale2)
        numerator = p1 * np.power(scale1, 4) + (1 - p1) * np.power(scale2, 4)
        
        # -3 to get in excess kurtosis
        return 3 * numerator / np.power(variance, 2) - 3
    

    def _params_to_cvar(self, p1, scale1, scale2, alpha = 0.95):
        # Carol Alexander IV.2.89
        # I couldn't get her formula for the general NormalMixture to work, but this works for the VarianceMixture case

        var = self._params_to_var(p1, scale1, scale2)
        term_1 = p1 * scale1 * stats.norm.pdf(var / scale1)
        term_2 = (1 - p1) * scale2 * stats.norm.pdf(var / scale2)

        return -(term_1 + term_2) / (1 - alpha)