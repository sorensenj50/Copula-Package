import numpy as np
from scipy import stats

from .marginals import Marginal
import utils

from typing import Union, Callable, Tuple
from type_definitions import Vectorizable, Vectorizable1d


class GaussianKDE(Marginal):
    def __init__(self, bw_method: Union[str, None] = None, Z_factor: float = 5, 
                 interpolation_n: int = 2000, monte_carlo_n: int = 10_000, monte_carlo_seed: Union[int, None] = None):
        
        # bw_method passed to scipy.stats.gaussian_kde
        # can be scalar, "scott", "silverman", or callable

        super().__init__(None, model_name = "GaussianKDE", family_name = "Non-Parametric", 
                         initial_param_guess = [], param_names = [],
                         param_bounds = [], params = [], mm_fit_available = False)

        # explicitly defaulting to Scott if not provided--this SciPy's default too
        self.bw_method = bw_method if bw_method is not None else "scott"
        self.kde_factor = np.nan
        self.estimation_method_str = self.get_bw_method_desc(self.bw_method)
        self.Z_factor = Z_factor
        self.interpolation_n = interpolation_n
        self.monte_carlo_n = monte_carlo_n
        self.monte_carlo_seed = monte_carlo_seed

        # will be estimated via monte carlo after fit
        self._mean = np.nan
        self._variance = np.nan
        self._skew = np.nan
        self._kurtosis = np.nan
        self._cvar = np.nan

        
    
    def fit(self, x: Vectorizable1d) -> None:
        # check that univariate
        self.kde = stats.gaussian_kde(x, bw_method = self.bw_method)

        # getting CDF and PPF interpolations
        self._set_cdf_ppf(np.min(x), np.max(x))

        # setting variables
        self.is_fit = True
        self.n = len(x)
        self.LL = self._log_likelihood(x)
        self.kde_factor = self.kde.factor

        # monte carlo estimates for skew, kurtosis, cvar
        self._mean, self._variance, self._skew, self._kurtosis = utils._monte_carlo_stats(self)


    def _set_cdf_ppf(self, min_x: float, max_x: float) -> None:
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


    def get_bw_method_desc(self, bw_method: str) -> str:
        if (bw_method == "scott") or (bw_method == "silverman"):
            return bw_method
        elif callable(bw_method):
            return "user callable"
        else:
            return "user set"


    def _pdf(self, x: Vectorizable1d) -> Vectorizable1d:
        return self.kde(x)
    

    def _logpdf(self, x: Vectorizable1d) -> Vectorizable1d:
        return np.log(self._pdf(x))
    

    def cdf(self, x: Vectorizable1d) -> Vectorizable1d:
        # error handling
        return self._cdf(x)
    

    def _cdf(self, x: Vectorizable1d) -> Vectorizable1d:
        return self.interp1d_cdf_func(x)
    

    def ppf(self, q: Vectorizable1d) -> Vectorizable1d:
        # error handling
        return self._ppf(q)
    

    def _ppf(self, q: Vectorizable1d) -> Vectorizable1d:
        return self.interp1d_ppf_func(q)
    

    def _params_to_mean(self, *params) -> float:
        return self._mean
    

    def _params_to_variance(self, *params) -> float:
        return self._variance


    def _params_to_skewness(self, *params) -> float:
        # Monte Carlo
        return self._skew
    
    
    def _params_to_kurtosis(self, *params) -> float:
        # Monte Carlo
        return self._kurtosis
    

    def _params_to_cvar(self, *params, alpha: float = 0.95) -> float:
        # alpha is unused
        return self._cvar
    

    def _get_extra_text(self) -> float:
        return ["PPF Estimated via Numerical Interpolation of CDF",
                "Skewness, Kurtosis, and CVaR Estimated via Monte Carlo"]