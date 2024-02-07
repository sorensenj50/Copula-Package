from scipy import stats
from marginals import MarginalDist
import numpy as np
from bivariate import BivariateCopula


class SkewedNormal(MarginalDist):
    def __init__(self, mean = None, stdev = None, alpha = None):
        self.mean = mean
        self.stdev = stdev
        self.alpha = alpha

        # the order of these params depends on SciPy
        super().__init__(stats.skewnorm, (alpha, mean, stdev), ("alpha", "mean", "stdev"), (0, 0, 1))

    def set_params(self, alpha, mean, stdev):
        self.alpha = alpha
        self.mean = mean
        self.stdev = stdev


class SkewedStudentT(MarginalDist):
    def __init__(self):
        pass

class MixtureCopula(BivariateCopula):
    def __init__(self, copulas, p = None):
        self.copulas = copulas
        self.k = len(self.copulas)
        self.p = self._normalize_p(np.ones(shape = self.k) if p is None else p)


    def _normalize_p(self, p):
        return p / np.sum(p)


    def fit(self, u1, u2, initial_param_guess, adj = 1e-4):
        # em algorithm
        raise NotImplementedError


    def _fit(self, u1, u2, initial_param_guess, adj):
        # em algorithim
        raise NotImplementedError


    def display(self):
        # displays information about copula
        # especially details about fitting
        # LL, AIC, selection criterion
        raise NotImplementedError
    

    def _log_likelihood(self, u1, u2, *params):
        return np.sum(self._logpdf(u1, u2, *params))
    

    def pdf(self, u1, u2, adj = 1e-4):
        u1_valid, u2_valid = self._handle_uu_input(u1, u2, adj = adj)
        return self._pdf(u1_valid, u2_valid, self.copulas, self.p)
    

    def _pdf(self, u1, u2, copulas, p):
        # both the density and the cumulative density are probability weighted
        return np.dot([c._pdf(u1, u2, *c.params) for c in copulas], p)
    
    # what is the log pdf of a mixture copula?
    
    
    def cdf(self, u1, u2, adj = 1e-4):
        u1_valid, u2_valid = self._handle_uu_input(u1, u2, adj = adj)
        return self._reshape_wrapper(u1_valid, u2_valid, self._cdf, self.copulas, self.p)
    

    def _cdf(self, u1, u2, copulas, p):
        # both the density and the cumulative density are probability weighted
        return np.dot([c._cdf(u1, u2, *c.params) for c in copulas], p)
    
    def conditional_quantile(self, u1, q, adj = 1e-5):
        raise NotImplementedError
    
    def _conditional_quantile(self, u1, q):
        raise NotImplementedError

        

    def simulate(self, n = 1000, seed = None):
        gen = np.random.default_rng(seed = seed)

        copula_indices = gen.random.choice(np.array(range(self.k)), 
                                           p = self.p, replace = True, size = n)
        
        u1 = gen.uniform(size = n)
        u2 = np.empty(shape = n)
        q = gen.uniform(size = n)

        for i, cop_idx in enumerate(copula_indices):
            c = self.copulas[cop_idx]
            u2[i] = c._quantile(u1[i], q[i], *c.params)

        return u1, u2
