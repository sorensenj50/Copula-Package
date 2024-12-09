import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
import seaborn as sns
from scipy.interpolate import interp1d


def mean_decay_func(initial_mean, n, decay_rate = 0.1, mean_pct_floor = 0.5):
    index = np.arange(n)
    return np.maximum(initial_mean * (1 - decay_rate) ** index, initial_mean * mean_pct_floor)


class PortfolioDist:
    def __init__(self, means, vol, corr):    
        self.means = means
        self.vol = vol
        self.n_managers = len(means)
        self.weights = np.ones(self.n_managers) / self.n_managers

        self.corr_matrix = np.full((self.n_managers, self.n_managers), corr)
        np.fill_diagonal(self.corr_matrix, 1)
        self.cov_matrix = self.corr_matrix * np.outer(np.full(self.n_managers, self.vol), np.full(self.n_managers, self.vol))
        
        self.portfolio_vol = np.sqrt(self.weights.T @ self.cov_matrix @ self.weights)
        self.portfolio_mean = np.dot(self.means, self.weights)

        self.x_range = np.linspace(self.portfolio_mean -4 * self.portfolio_vol, self.portfolio_mean + 4 * self.portfolio_vol, 100)

    def pdf(self):
        return self._pdf(self.x_range)

    def _pdf(self, x):
        return stats.norm.pdf(x, loc = self.portfolio_mean, scale = self.portfolio_vol)

    def cdf(self):
        return self._cdf(self.x_range)

    def _cdf(self, x):
        return stats.norm.cdf(x, loc = self.portfolio_mean, scale = self.portfolio_vol)
    

class PortfolioMixtureDist:
    def __init__(self, p, distributions):
        self.p = np.array(p)
        self.distributions = distributions
        self.n_regimes = len(p)

        x_max = np.max([np.max(d.x_range) for d in self.distributions])
        x_min = np.min([np.min(d.x_range) for d in self.distributions])

        self.means = np.array([d.portfolio_mean for d in self.distributions])
        self.vols = np.array([d.portfolio_vol for d in self.distributions])
        
        self.x_range = np.linspace(x_min, x_max, 100)
        self.quantile_range = np.linspace(0, 1, 100)
    
        cdf_values = self._cdf(self.x_range)
        self.cdf_interp = interp1d(self.x_range, cdf_values, bounds_error = False, fill_value = (self.x_range[0], self.x_range[-1]))
        self.ppf_interp = interp1d(cdf_values, self.x_range, bounds_error = False, fill_value = (cdf_values[0], cdf_values[-1]))
        

    def pdf(self):
        return self._pdf(self.x_range)

    def _pdf(self, x):
        return np.sum(self.p * stats.norm.pdf(x[:, None], loc=self.means, scale = self.vols), axis = 1)

    def cdf(self):
        return self._cdf(self.x_range)

    def _cdf(self, x):
        return np.sum(self.p * stats.norm.cdf(x[:, None], loc=self.means, scale = self.vols), axis = 1)

    def ppf(self):
        return self._ppf(self.quantile_range)

    def _ppf(self, q):
        return self.ppf_interp(q)

    def simulate(self, n = 1000, seed = None):
        rng = np.random.default_rng(seed = seed)

        random_q = rng.uniform(size = n)
        return self._ppf(random_q)
    
    def stats(self):
        # numerical integration
        
        mix_pdf = lambda x: np.sum(self.p * stats.norm.pdf(x, loc=self.means, scale=self.vols))
        mix_mean = np.dot(self.p, self.means)
        mix_variance = np.sum(self.p * (self.vols**2 + self.means**2)) - (mix_mean ** 2)

        def central_moment(n):
            integrand = lambda x: ((x - mix_mean) ** n) * mix_pdf(x)
            return quad(integrand, -np.inf, np.inf)[0]

        skewness = central_moment(3) / (mix_variance**1.5)
        kurtosis = central_moment(4) / (mix_variance**2)

        return {"Ann. Mean": mix_mean, "Ann. Std": np.sqrt(mix_variance), "Sharpe": mix_mean / np.sqrt(mix_variance),
                "Skew": skewness, "Kurtosis": kurtosis - 3}
    
    def prob_exceeding_hurdle(self, hurdle):
        return 1 - self._cdf(np.array([hurdle]))


class ManagerEvaluator:
    def __init__(self, p, vols, sharpes, corrs, crisis_flags, mean_decay_rate = 0.1, mean_pct_floor = 0.5, 
                 return_hurdle = 0.05, crisis_hurdle = 0.1):
        
        self.p = p
        self.vols = vols
        self.sharpes = sharpes
        self.corrs = corrs
        self.crisis_flags = crisis_flags
        self.mean_decay_rate = mean_decay_rate
        self.mean_pct_floor = mean_pct_floor
        self.return_hurdle = return_hurdle
        self.crisis_hurdle = crisis_hurdle
        
    def evaluate(self, n_managers):
    
        all_regimes = []
    
        crisis_regimes = []
        crisis_p = []
    
        for i in range(len(self.p)):
            base_mean = self.sharpes[i] * self.vols[i]
            mean_vector = mean_decay_func(base_mean, n_managers, self.mean_decay_rate, self.mean_pct_floor)
        
            regime_dist = PortfolioDist(mean_vector, self.vols[i], self.corrs[i])
            all_regimes.append(regime_dist)
        
            if self.crisis_flags[i] == 1:
                crisis_regimes.append(regime_dist)
                crisis_p.append(self.p[i])
    
        crisis_p = np.array(crisis_p) / np.sum(crisis_p)
    
    
        total_distribution = PortfolioMixtureDist(self.p, all_regimes)
        conditioned_crisis_distribution = PortfolioMixtureDist(crisis_p, crisis_regimes)
    
        return {"Dist": total_distribution,
                "Crisis Dist": conditioned_crisis_distribution}