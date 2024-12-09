from scipy import stats, special
import numpy as np


from .marginals import Marginal
import utils

from typing import Union, Tuple
from type_definitions import Vectorizable, Vectorizable1d




class Normal(Marginal):
    def __init__(self, loc: float = 0, scale: float = 1, adj: float = 1e-5):

        # the order of these params depends on SciPy
        super().__init__(stats.norm, model_name = "Normal", family_name = "Parametric", initial_param_guess = [0, 1], 
                        param_bounds = [(-np.inf, np.inf), (adj, np.inf)], param_names = ["loc", "scale"],
                        params = [loc, scale], mm_fit_available = True)
        
    def _params_to_mean(self, loc: float, scale: float) -> float:
        return loc
    
    
    def _params_to_variance(self, loc: float, scale: float) -> float:
        return scale ** 2
        

    def _params_to_skewness(self, loc: float, scale: float) -> float:
        return 0
    

    def _params_to_kurtosis(self, loc: float, scale: float) -> float:
        return 0
    

    def _params_to_cvar(self, loc: float, scale: float, alpha: float = 0.95) -> float:
        # Matthew Norton et al 2019 and Carol Alexander IV.2.85
        # not passing params to _pdf and _ppf: standard normal
        return loc - scale * (self._pdf(self._ppf(alpha))) / (1 - alpha)
    

# private
class CenteredNormal(Marginal):
    def __init__(self, scale: float = 1, adj: float = 1e-5):

        super().__init__(stats.norm, model_name = "CenteredNormal", family_name = "Parametric",
                         initial_param_guess = [1], param_bounds = [(adj, np.inf)],
                         param_names = ["sigma"], params = [scale], mm_fit_available = True)
        
    # relying on core methods implemented generically by parent class
    # imposing mu = 0 constraint
    def _pdf(self, x: Vectorizable, scale: float) -> Vectorizable:
        return super()._pdf(x, 0, scale)
    

    def _logpdf(self, x: Vectorizable, scale: float) -> Vectorizable:
        return super()._logpdf(x, 0, scale)
    

    def _cdf(self, x: Vectorizable, scale: float) -> Vectorizable:
        return super()._cdf(x, 0, scale)
    

    def _ppf(self, q: Vectorizable, scale: float) -> Vectorizable:
        return super()._ppf(q, 0, scale)
    

    def fit(self, x: Vectorizable1d, robust_cov: bool = True):
        # input validation
        valid_x = self._handle_input(x)

        # relying on scipy implementation of fit
        opt_params = self.rv_obj.fit(valid_x, floc = 0)
        self._post_process_fit(valid_x, np.array([opt_params[1]]), 
                               self._get_obj_func(valid_x), robust_cov = robust_cov)
        
    def _params_to_mean(self, scale: float) -> float:
        return 0
    

    def _params_to_variance(self, scale: float) -> float:
        return scale ** 2
    

    def _params_to_skewness(self, scale: float) -> float:
        return 0
    

    def _params_to_kurtosis(self, scale: float) -> float:
        return 0
    

    def _params_to_cvar(self, scale: float, alpha: float = 0.95) -> float:
        # Mattew Norton et al 2019
        # _pdf and _ppf use standard normal

        return -scale * (self._pdf(self._ppf(alpha, 1), 1)) / (1 - alpha)
    

# MM?
class SkewNormal(Marginal):
    # Azzalini's SkewNormal distribution

    def __init__(self, loc: float = 0, scale: float = 1, shape: float = 0, adj: float = 1e-5):

        super().__init__(None, model_name = "SkewNormal", family_name = "Parametric", initial_param_guess = [0, 1, 0],
                         param_bounds = [(-np.inf, np.inf), (adj, np.inf), (-np.inf, np.inf)], param_names = ["loc", "scale", "shape"],
                         params = [loc, scale, shape], mm_fit_available = False)
        
        
    def _z(self, x: Vectorizable, loc: float, scale: float) -> Vectorizable:
        # standard linear transform
        return (x - loc) / scale
    

    def _delta(self, shape: float) -> float:
        # helper function for delta variable
        return shape / (np.sqrt(1 + np.power(shape, 2)))
    

    def _cdf(self, x: Vectorizable, loc: float, scale: float, shape: float) -> Vectorizable:
        # Azzalini formulation
        z = self._z(x, loc, scale)
        return stats.norm.cdf(z) - 2 * special.owens_t(z, shape)
    

    def _pdf(self, x: Vectorizable, loc: float, scale: float, shape: float) -> Vectorizable:
        # Azzalini's formulation
        z = self._z(x, loc, scale)
        return (2 / scale) * stats.norm.pdf(z) * stats.norm.cdf(shape * z) 
    

    def _logpdf(self, x: Vectorizable, loc: float, scale: float, shape: float) -> Vectorizable:
        # log of Azzalini pdf
        return np.log(self._pdf(x, loc, scale, shape))
    

    def _ppf(self, q: Vectorizable, loc: float, scale: float, shape: float) -> Vectorizable:
        a, b = utils.find_x_bounds(self._cdf, loc, scale, loc, scale, shape)
        return utils.solve_for_ppf(self._cdf, q, a, b, loc, scale, shape)
    

    def _params_to_mean(self, loc: float, scale: float, shape: float) -> float:
        # wikipedia
        delta = self._delta(shape)
        return loc + scale * delta * np.sqrt(2 / np.pi)
    

    def _params_to_variance(self, loc: float, scale: float, shape: float) -> float:
        # wikipedia
        delta = self._delta(shape)
        return np.power(scale, 2) * (1 - 2 * np.power(delta, 2) / np.pi)
    

    def _params_to_skewness(self, loc: float, scale: float, shape: float) -> float:
        # wikipedia

        delta = self._delta(shape)

        term_1 = (4 - np.pi) / 2
        term_2 = np.power(delta * np.sqrt(2 / np.pi), 3)
        term_3 = np.power(1 - 2 * np.power(delta, 2) / np.pi, 3/2)

        return term_1 * term_2 / term_3
    

    def _params_to_kurtosis(self, loc: float, scale: float, shape: float) -> float:
        # wikipedia

        delta = self._delta(shape)

        term_1 = 2 * (np.pi - 3)
        term_2 = np.power(delta * np.sqrt(2 / np.pi), 4)
        term_3 = np.power(1 - 2 * np.power(delta, 2) / np.pi, 2)

        return term_1 * term_2 / term_3
    
    
    def _params_to_cvar(self, loc: float, scale: float, shape: float, alpha: float = 0.95) -> float:
        # Bernadi 2012
        # see also "Estimation methods for Expected Shortfall" by University of Manchester
        # more consice formula, though it mistakenly uses shape instead of delta in term2

        p = 1 - alpha

        x_p = self._ppf(p, loc, scale, shape)
        y_p = (x_p - loc) / scale
        z_p = np.sqrt(1 + np.power(shape, 2)) * y_p
        delta = self._delta(shape)

        term_1 = (scale * np.sqrt(2)) / (p * np.sqrt(np.pi))
        term_2 = delta * stats.norm.cdf(z_p)
        term_3 = np.sqrt(2 * np.pi) * stats.norm.pdf(y_p) * stats.norm.cdf(shape * y_p)

        return loc + term_1 * (term_2 - term_3)


    

class StudentsT(Marginal):
    def __init__(self, df: float = 30, loc: float = 0, scale: float = 1, adj: float = 1e-5):
       
        # the order of these params depends on SciPy
        super().__init__(stats.t, model_name = "StudentsT", family_name = "Parametric", initial_param_guess = [30, 0, 1], 
                        param_bounds = [(1, np.inf), (-np.inf, np.inf), (adj, np.inf)], param_names = ["df", "loc", "scale"],
                        params = [df, loc, scale], mm_fit_available = False)
        

    def _params_to_mean(self, df: float, loc: float, scale: float) -> float:
        return loc
    

    def _params_to_variance(self, df: float, loc: float, scale: float) -> float:
        return (scale ** 2) * (df / (df - 2))
        

    def _params_to_skewness(self, df: float, loc: float, scale: float) -> float:
        # wikipedia
        return 0 if df > 3 else np.nan
        

    def _params_to_kurtosis(self, df: float, loc: float, scale: float) -> float:
        # wikipedia

        if df > 4:
            return 6 / (df - 4)
        elif df > 2 and df <= 4:
            return np.inf
        else:
            return np.nan
    
    
    def _params_to_cvar(self, df: float, loc: float, scale: float, alpha: float = 0.95) -> float:
        # Matthew Nortan (2019) and Carol Alexander IV.2.88
        
        term1 = (df + stats.t.ppf(alpha, df) ** 2) / ((df - 1) * (1 - alpha))
        term2 = stats.t.pdf(stats.t.ppf(alpha, df), df)

        return loc - scale * term1 * term2
    

#class SkewedGT(Marginal):
 #   def __init__(self, k, n, lam, sigma, adj = float 1e-5):
 #       
 #       super().__init__(None, model_name = "SkewedGT", family_name = "Parametric",
 #                        initial_param_guess = [0, 0, 0, 1], param_bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)],
 #                        params = [k, n, lam, sigma])
    


class StandardSkewedT(Marginal):
    # Hansen 1994

    def __init__(self, eta: float = 30, lam: float = 0, df_cap: float = 100, adj: float = 1e-5, 
                 monte_carlo_n: int = 10_000, monte_carlo_seed: Union[int, None] = None):

        super().__init__(None, model_name = "StandardSkewedT", family_name = "Parametric",
                         initial_param_guess = [30, 0], param_names = ["eta", "lam"],
                         param_bounds = [(2 + adj, df_cap), (-1 + adj, 1 - adj)],
                         params = [eta, lam])
        
        self._skew = np.nan
        self._kurtosis = np.nan
        self._cvar = np.nan
        self.monte_carlo_n = monte_carlo_n
        self.monte_carlo_seed = monte_carlo_seed
        

    def _get_ABC(self, eta: float, lam: float) -> Tuple[float, float]:
        # helper function to get the constants A, B, C as defined in Hansen 1994

        C = special.gamma((eta + 1) / 2) / (np.sqrt(np.pi * (eta - 2))  * special.gamma(eta / 2))
        A = 4 * lam * C * (eta - 2) / (eta - 1)
        B = np.sqrt(1 + 3 * (lam**2) - (A**2))

        return A, B, C
    

    def _logpdf(self, x: Vectorizable, eta: float, lam: float) -> Vectorizable:

        # constants
        A, B, C = self._get_ABC(eta, lam)

        # this introduces skewness
        denom = np.where(x < -A/B, 1 - lam, 1 + lam)
        inside_term = 1 + 1/(eta - 2) * np.square((B * x + A)/denom)
        return np.log(B) + np.log(C) - ((eta + 1) / 2) * np.log(inside_term)
    

    def _pdf(self, x: Vectorizable, eta: float, lam: float) -> Vectorizable:
        return np.exp(self._logpdf(x, eta, lam))


    def _ppf(self, q: Vectorizable, eta: float, lam: float) -> Vectorizable:
        # source: Tino Contino (DirtyQuant)
        # constants
        A, B, _ = self._get_ABC(eta, lam)
        eta_const = np.sqrt((eta - 2) / eta)

        # switching logic
        core = np.where(q < (1 - lam) / 2, 
                        (1 - lam) * stats.t.ppf(q / (1 - lam), eta), 
                        (1 + lam) * stats.t.ppf((q + lam) / (1 + lam), eta))
        
        return (1 / B) * (eta_const * core - A)
    

    def _cdf(self, x: Vectorizable, eta: float, lam: float) -> Vectorizable:
        # source: Tino Contino (DirtyQuant)

        # constants
        A, B, _ = self._get_ABC(eta, lam)
        numerator = np.sqrt(eta / (eta - 2)) * (B * x + A)

        return np.where(x < -A/B,
                    (1 - lam) * stats.t.cdf(numerator / (1 - lam), eta),
                    (1 + lam) * stats.t.cdf(numerator / (1 + lam), eta) - lam)
    

    
    def fit(self, x: Vectorizable1d, optimizer: str = "Powell", robust_cov: bool = True):
        # error handling
        valid_x = self._handle_input(x)

        f = self._get_obj_func(valid_x)
        opt_results = self._fit(f, self.initial_param_guess, self.param_bounds, optimizer = optimizer)
        self._post_process_fit(valid_x, opt_results.x, self._get_obj_func(x), robust_cov = robust_cov)

        # monte carlo
        self._skew, self._kurtosis, self._cvar = utils.monte_carlo_stats(self)


    def _params_to_mean(self, eta: float, lam: float) -> float:
        return 0
    

    def _params_to_variance(self, eta: float, lam: float) -> float:
        return 1
    

    def _params_to_skewness(self, eta: float, lam: float) -> float:
        return


    @property
    def skewness(self) -> float:
        # bypassing / not implementing _params_to_skew
        return self._skew
    

    @property
    def kurtosis(self) -> float:
        # bypasssing / not implementing _params_to_kurtosis
        return self._kurtosis
    

    @property
    def cvar(self) -> float:
        # bypassing / not implementing params_to_cvar
        return self._cvar


    def summary(self) -> str:
        if not self.is_fit:
            # if not already estimated, on the fly monte carlo for params
            self._skew, self._kurtosis, self._cvar = utils.monte_carlo_stats(self)

        return super().summary()
    

    def _get_extra_text(self) -> str:
        return super()._get_extra_text() + ["Skewness, Kurtosis, and CVaR Estimated via Monte Carlo"]
    

# input validation for low and high
# not allow MLE
class Uniform(Marginal):
    def __init__(self, low: float = 0, high: float = 1, adj = 1e-5):



        super().__init__(None, model_name = "Uniform", family_name = "Parametric",
                         initial_param_guess = [0, 1], param_bounds = [(-np.inf, np.inf), (-np.inf, np.inf)],
                         param_names = ["low", "high"], params = [low, high], mm_fit_available = True)
        

    def _to_scipy_params(self, low: float, high: float) -> tuple[float, float]:
        return low, high - low


    def _pdf(self, x: Vectorizable, low: float, high: float) -> Vectorizable:
        return stats.uniform(x, *self._to_scipy_params(low, high))
    

    def _cdf(self, x: Vectorizable, low: float, high: float) -> Vectorizable:
        return stats.uniform.cdf(x, *self._to_scipy_params(low, high))
    

    def _ppf(self, x: Vectorizable, low: float, high: float) -> Vectorizable:
        return stats.uniform.ppf(x, *self._to_scipy_params(low, high))
    

    def _logpdf(self, x: Vectorizable, low: float, high: float) -> Vectorizable:
        return stats.uniform.logpdf(x, *self._to_scipy_params(low, high))
    

    def _params_to_mean(self, low: float, high: float) -> float:
        return 1/2 * (low + high)
    

    def _params_to_variance(self, low: float, high: float) -> float:
        return 1/12 * np.power(high - low, 2)
    

    def _params_to_skewness(self, low: float, high: float) -> float:
        return 0 
    

    def _params_to_kurtosis(self, low: float, high: float) -> float:
        return -6/5
    

    def _params_to_cvar(self, low: float, high: float, alpha = 0.95) -> float:
        return self._params_to_mean(low, self._ppf(1 - alpha))

    

        



class Exponential(Marginal):
    def __init__(self, rate: float = 1, adj: float = 1e-5):
        
        super().__init__(None, model_name = "Exponential", family_name = "Parametric",
                         initial_param_guess = [1], param_bounds = [(adj, np.inf)], 
                         param_names = ["rate"], params = [rate], mm_fit_available = True)


    def _to_scipy_params(self, rate: float) -> tuple[float, float]:
        # helper function
        return 0, 1 / rate
        

    def _pdf(self, x: Vectorizable, rate: float) -> Vectorizable:
        return stats.expon.pdf(x, *self._to_scipy_params(rate))
    

    def _logpdf(self, x: Vectorizable, rate: float) -> Vectorizable:
        return stats.expon.logpdf(x, *self._to_scipy_params(rate))
    

    def _cdf(self, x: Vectorizable, rate: float) -> Vectorizable:
        return stats.expon.cdf(x, *self._to_scipy_params(rate))
    

    def _ppf(self, q: Vectorizable, rate: float) -> Vectorizable:
        return stats.expon.ppf(q, *self._to_scipy_params(rate))
    

    def fit(self, x: Vectorizable, robust_cov: bool = True):
        valid_x = self._handle_input(x)

        opt_params = stats.expon.fit(valid_x, floc = 0)
        self._post_process_fit(valid_x, np.array([opt_params[1]]),
                               self._get_obj_func(valid_x), robust_cov = robust_cov)
        
    def _params_to_mean(self, rate: float) -> float:
        return 1 / rate
    

    def _params_to_variance(self, rate: float) -> float:
        return 1 / np.power(rate, 2)
    
        
    def _params_to_skewness(self, rate: float) -> float:
        return 2
    

    def _params_to_kurtosis(self, rate: float) -> float:
        return 6
    

    def _params_to_cvar(self, rate: float, alpha = 0.95):
        x = self._ppf(1 - alpha, rate)

        numerator = x * rate - np.exp(x * rate) + 1
        denom = rate * (np.exp(x * rate) - 1)

        return - numerator / denom

