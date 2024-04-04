from scipy import stats, special
import numpy as np

from .marginals import Marginal
import utils



class Normal(Marginal):
    def __init__(self, loc = 0, scale = 1, adj = 1e-4):

        # the order of these params depends on SciPy
        super().__init__(stats.norm, model_name = "Normal", family_name = "Parametric", initial_param_guess = [0, 1], 
                        param_bounds = [(-np.inf, np.inf), (adj, np.inf)], param_names = ["loc", "scale"],
                        params = [loc, scale])
        

    def _params_to_skewness(self, loc, scale):
        return 0
    

    def _params_to_kurtosis(self, loc, scale):
        return 0
    

    def _params_to_cvar(self, loc, scale, alpha = 0.95):
        # Matthew Norton et al 2019
        # not passing params to _pdf and _ppf: standard normal
        return loc - scale * (self._pdf(self._ppf(alpha))) / (1 - alpha)
    

# private
class CenteredNormal(Marginal):
    def __init__(self, scale = 1, adj = 1e-4):

        super().__init__(stats.norm, model_name = "CenteredNormal", family_name = "Parametric",
                         initial_param_guess = [1], param_bounds = [(adj, np.inf)],
                         param_names = ["sigma"], params = [scale])
        
    # relying on core methods implemented generically by parent class
    # imposing mu = 0 constraint
    def _pdf(self, x, scale):
        return super()._pdf(x, 0, scale)
    

    def _logpdf(self, x, scale):
        return super()._logpdf(x, 0, scale)
    

    def _cdf(self, x, scale):
        return super()._cdf(x, 0, scale)
    

    def _ppf(self, q, scale):
        return super()._ppf(q, 0, scale)
    

    def fit(self, x, robust_cov = True):
        # input validation
        valid_x = self._handle_input(x)

        # relying on scipy implementation of fit
        opt_params = self.rv_obj.fit(x, floc = 0)
        self._post_process_fit(valid_x, np.array([opt_params[1]]), 
                               self._get_obj_func(valid_x), robust_cov = robust_cov)
    
    def _params_to_skewness(self, scale):
        return 0
    

    def _params_to_kurtosis(self, scale):
        return 0
    

    def _params_to_cvar(self, scale, alpha = 0.95):
        # Mattew Norton et al 2019
        # _pdf and _ppf use standard normal

        return -scale * (self._pdf(self._ppf(alpha, 1), 1)) / (1 - alpha)
        
    

class StudentsT(Marginal):
    def __init__(self, df = 30, loc = 0, scale = 1, adj = 1e-4):
       
        # the order of these params depends on SciPy
        super().__init__(stats.t, model_name = "StudentsT", family_name = "Parametric", initial_param_guess = [30, 0, 1], 
                        param_bounds = [(1, np.inf), (-np.inf, np.inf), (adj, np.inf)], param_names = ["df", "loc", "scale"],
                        params = [df, loc, scale])
        

    def _params_to_skewness(self, df, loc, scale):
        return 0 if df > 3 else np.nan
        

    def _params_to_kurtosis(self, df, loc, scale):
        if df > 4:
            return 6 / (df - 4)
        elif df > 2 and df <= 4:
            return np.inf
        else:
            return np.nan
    
    
    def _params_to_cvar(self, df, loc, scale, alpha = 0.95):
        # Nortan (2019) and Carol Alexander IV.2.88å
        
        term1 = (df + stats.t.ppf(alpha, df) ** 2) / ((df - 1) * (1 - alpha))
        term2 = stats.t.pdf(stats.t.ppf(alpha, df), df)

        return loc - scale * term1 * term2
    


class StandardSkewedT(Marginal):
    # Hansen 1994

    def __init__(self, eta = 30, lam = 0, df_cap = 100, adj = 1e-4, monte_carlo_n = 10_000, monte_carlo_seed = None):
        super().__init__(None, model_name = "StandardSkewedT", family_name = "Parametric",
                         initial_param_guess = [30, 0], param_names = ["eta", "lam"],
                         param_bounds = [(2 + adj, df_cap), (-1 + adj, 1 - adj)],
                         params = [eta, lam])
        
        self._skew = np.nan
        self._kurtosis = np.nan
        self._cvar = np.nan
        self.monte_carlo_n = monte_carlo_n
        self.monte_carlo_seed = monte_carlo_seed
        

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

        f = self._get_obj_func(valid_x)
        opt_results = self._fit(f, self.initial_param_guess, self.param_bounds, optimizer = optimizer)
        self._post_process_fit(valid_x, opt_results.x, self._get_obj_func(x), robust_cov = robust_cov)

        # monte carlo
        self._skew, self._kurtosis, self._cvar = utils.monte_carlo_stats(self)


    @property
    def skewness(self):
        # bypassing / not implementing _params_to_skew
        return self._skew
    

    @property
    def kurtosis(self):
        # bypasssing / not implementing _params_to_kurtosis
        return self._kurtosis
    

    @property
    def cvar(self):
        # bypassing / not implementing params_to_cvar
        return self._cvar


    def summary(self):
        if not self.is_fit:
            # if not already estimated, on the fly monte carlo for params
            self._skew, self._kurtosis, self._cvar = utils.monte_carlo_stats(self)

        return super().summary()
    
    def _get_extra_text(self):
        return super()._get_extra_text() + ["Skewness, Kurtosis, and CVaR Estimated via Monte Carlo"]