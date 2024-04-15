import numpy as np

from .bivariate_copula import BivariateCopula
from .elliptical import Normal
from mixture import Mixture

from numpy.random import Generator
from type_definitions import Vectorizable
from typing import Union, Tuple



class NormalMix(Mixture, BivariateCopula):
    def __init__(self, p1: float = 0.5, Q1: float = 0, Q2: float = 0, adj: float = 1e-5):

        # case if lengths of p and Q disagree / with n_normals
        p1 = self._normalize_p(p1)

        BivariateCopula.__init__(self, model_name = "NormalMix", family_name = "Finite Mixture", initial_param_guess = [np.nan, np.nan, np.nan],
                         param_bounds = [(adj, 1 - adj), (-1 + adj, 1 - adj), (-1 + adj, 1 - adj)],
                         param_names = ["p1", "Q1", "Q2"], params = [p1, Q1, Q2], mm_fit_available = False)
        
        Mixture.__init__(self, Normal())
    

    def _get_random_params(self, n: int, rng: Generator, *data: Vectorizable, adj: float = 1e-5) -> np.ndarray:
        # ensuring that correlation parameter is safely not 1 or -1
        # data argument is unused
        return rng.uniform(-1 + adj, 1 - adj, size = (n, 1))    


    def fit(self, u1: Vectorizable, u2: Vectorizable, seed: Union[int, None] = None, 
            n_init: int = 20, tol: float = 1e-4, max_iter: int = 100, optimizer: str = "Powell") -> None:
        
        LL, p1, Q1, Q2 = self._run_em_algo_multi(u1, u2, seed = seed, n_init = n_init, tol = tol, 
                                                 max_iter = max_iter, optimizer = optimizer)
        
        self._mini_post_process_fit(LL, u1.shape[0])
        self._set_params((p1, Q1, Q2))


    def _pdf(self, u1: Vectorizable, u2: Vectorizable, p1: float, Q1: float, Q2: float) -> Vectorizable:
        return self._mixture_pdf(p1, (Q1,), (Q2,), u1, u2)
    

    def _logpdf(self, u1: Vectorizable, u2: Vectorizable, p1: float, Q1: float, Q2: float) -> Vectorizable:
        return np.log(self._pdf(u1, u2, p1, Q1, Q2))


    def _cdf(self, u1: Vectorizable, u2: Vectorizable, p1: float, Q1: float, Q2: float) -> Vectorizable:
        return self._mixture_cdf(p1, (Q1,), (Q2,), u1, u2)
    

    def _conditional_cdf(self, u1: Vectorizable, u2: Vectorizable, p1: float, Q1: float, Q2: float) -> Vectorizable:
        # cdf of u2 conditioned on u1
        return p1 * self._base_model._conditional_cdf(u1, u2, Q1) + (1 - p1) * self._base_model._conditional_cdf(u1, u2, Q2)
    

    def simulate(self, n: int = 1000, seed: Union[int, None] = None, adj: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:

        p1, Q1, Q2 = self.params

        rng = np.random.default_rng(seed = seed)
        param_draw = rng.choice([Q1, Q2], p = [p1, 1 - p1], replace = True, size = n)

        u1 = rng.uniform(size = n); q = rng.uniform(size = n)
        u2 = np.empty(shape = n)

        for i, Q in enumerate(param_draw):
            u2[i] = self._base_model.conditional_ppf(u1[i], q[i], Q, adj = adj)

        return u1, u2
    

    def _lower_tail_dependance(self, *params: float) -> float:
        return 0
    

    def _upper_taiL_dependance(self, *params: float) -> float:
        return 0
    

    def _get_extra_text(self) -> str:

        text_list = ["Tau and Rho calculated using numerical integration of CDF", 
                     "Conditional PPF solved using Brent's Method"]
        
        if self.is_fit:
            return [super()._get_extra_text()[0]] + text_list
        
        return text_list