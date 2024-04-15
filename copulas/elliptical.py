import numpy as np
from scipy import stats, special
from .bivariate_copula import BivariateCopula

from type_definitions import Vectorizable
from typing import Tuple


class Elliptical(BivariateCopula):

    def _cov_det(self, Q: float) -> float:
        return 1 - Q ** 2
    

    def _scale_factor(self, Q: float) -> float:
        return np.sqrt(self._cov_det(Q))
    

    def _distance(self, z1: Vectorizable, z2: Vectorizable, Q: float) -> Vectorizable:
        # distance between points, used in the copula density
        raise NotImplementedError
    


class Normal(Elliptical):
    def __init__(self, Q: float = 0, adj: float = 1e-4):
        super().__init__(model_name = "Normal", family_name = "Elliptical", initial_param_guess = [0], 
                         param_bounds = [(-1 + adj, 1 - adj)], param_names = ("Q",), 
                         params = (Q,), mm_fit_available = True)
        
    
    def _distance(self, z1: Vectorizable, z2: Vectorizable, Q: float) -> Vectorizable:
        # modified mahalonobis distance helper function
        return ((Q * z1) ** 2 - (2 * Q * z1 * z2) + (Q * z2) ** 2) / self._cov_det(Q)
    

    def _cdf(self, u1: Vectorizable, u2: Vectorizable, Q: float) -> Vectorizable:
        z1 = stats.norm.ppf(u1); z2 = stats.norm.ppf(u2)
        z = np.stack([np.atleast_1d(z1), np.atleast_1d(z2)], axis = 1)
        return stats.multivariate_normal.cdf(z, cov = np.array([[1, Q],[Q, 1]]))
    
    
    def _logpdf(self, u1: Vectorizable, u2: Vectorizable, Q: float) -> Vectorizable:
        z1 = stats.norm.ppf(u1); z2 = stats.norm.ppf(u2)
        return -np.log(self._scale_factor(Q)) - 1/2 * self._distance(z1, z2, Q)
    

    def _conditional_cdf(self, u1: Vectorizable, u2: Vectorizable, Q: float, adj: float = 1e-5) -> Vectorizable:
        # adj unused but here for consistency
        # Carol Alexander II.6.61 (correcting typo)

        z1 = stats.norm.ppf(u1); z2 = stats.norm.ppf(u2)
        return stats.norm.cdf((z2 - Q * z1) / self._scale_factor(Q))
    

    def _conditional_ppf(self, u1: Vectorizable, q: Vectorizable, Q: float, adj: float = 1e-5) -> Vectorizable:
        # adj unused but here for consistency
        # Carol Alexander II.6.62

        z1 = stats.norm.ppf(u1); z2 = stats.norm.ppf(q)
        return stats.norm.cdf(Q * z1 + self._scale_factor(Q) * z2)
    

    def _params_to_tau(self, Q: float) -> float:
        return 2 * np.arcsin(Q) / np.pi
    
    
    def _tau_to_params(self, tau: float) -> Tuple[float]:
        return tuple(2 * np.sin((np.pi / 6) * tau))


    def _params_to_rho(self, Q: float) -> float:
        return 6 * np.arcsin(Q / 2) / np.pi
    

    def _lower_tail_dependance(self, *params: float) -> float:
        # McNeil 2005
        return 0
    
    def _upper_taiL_dependance(self, *params: float) -> float:
        # McNeil 2005
        return 0
    

    

class StudentsT(Elliptical):
    # Student T copula

    def __init__(self, df: float = 30, Q: float = 0, adj: float = 1e-4, df_upper_bound: float = 100):
        super().__init__(model_name = "StudentsT", family_name = "Elliptical", 
                         initial_param_guess = [30, 0], 
                         param_bounds = [(1, df_upper_bound), (-1 + adj, 1 - adj)], 
                         param_names = ("df", "Q"), params = (df, Q), mm_fit_available = False)


    def _distance(self, z1: Vectorizable, z2: Vectorizable, Q: float) -> Vectorizable:
        return ((z1 ** 2) - (2 * Q * z1 * z2) + (z2 ** 2)) / self._cov_det(Q)
    

    def _cdf(self, u1: Vectorizable, u2: Vectorizable, df: float, Q: float) -> Vectorizable:
        z1 = stats.t.ppf(u1, df); z2 = stats.t.ppf(u2, df)
        z = np.stack([np.atleast_1d(z1), np.atleast_1d(z2)], axis = 1)
        return stats.multivariate_t.cdf(z, df = df, shape = np.array([[1, Q],[Q, 1]]))
    

    def _logpdf(self, u1: Vectorizable, u2: Vectorizable, df: float, Q: float) -> Vectorizable:
        n = 2

        # to t variables
        z1 = stats.t.ppf(u1, df); z2 = stats.t.ppf(u2, df)

        log_K = np.log(special.gamma((df + n) / 2)) + (n - 1) * np.log(special.gamma(df / 2)) + -n * np.log(special.gamma((df + 1) / 2))
        log_scale = np.log(self._scale_factor(Q))
        log_numerator = (-(df + n)/2) * np.log(1 + self._distance(z1, z2, Q) / df)
        log_denom = (-(df + 1)/2) * np.log((1 + (z1 ** 2)/df) * (1 + (z2 ** 2)/df))

        return (log_K - log_scale) + (log_numerator - log_denom)
    

    def _conditional_cdf(self, u1: Vectorizable, u2: Vectorizable, df: float, Q: float, adj: float = 1e-4) -> Vectorizable:
        # adj is unused, here for consistency
        # Carol Alexander II.G.68

        t1 = stats.t.ppf(u1, df); t2 = stats.t.ppf(u2, df)
        return stats.t.cdf(np.sqrt((df + 1) / (df + t1 ** 2)) * ((t2 - Q * t1) / self._scale_factor(Q)), df + 1)


    def _conditional_ppf(self, u1: Vectorizable, q: Vectorizable, df: float, Q: float, adj: float = 1e-4) -> Vectorizable:
        # adj is unused, here for consistency
        # Carol Alexander II.6.69
        t1 = stats.t.ppf(u1, df); t2 = stats.t.ppf(q, df + 1)
        return stats.t.cdf(Q * t1 + np.sqrt(self._scale_factor(Q) / (df + 1) * (df + t1 ** 2)) * t2, df)
        

    def _params_to_tau(self, df: float, Q: float) -> float:
        # Lindskog 2003 & Carol Alexander II.6.78
        return 2 * np.arcsin(Q) / np.pi


    def _tail_dependance(self, df: float, Q: float) -> float:
        # McNeil 2005
        return 2 * stats.t.cdf(-np.sqrt((df + 1) * (1 - Q) / (1 + Q)), df + 1)
    

    def _upper_taiL_dependance(self, df: float, Q: float) -> float:
        return self._tail_dependance(df, Q)
    

    def _lower_tail_dependance(self, df: float, Q: float) -> float:
        return self._tail_dependance(df, Q)