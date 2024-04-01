import numpy as np
from concurrent.futures import ProcessPoolExecutor
from base import Base


class Mixture(Base):
    # mixture of two probability distributions or copulas
    # implements shared code for the EM algorithm

    def __init__(self, base_model):
        self._base_model = base_model
        self.estimation_method_str = "EM Algo"


    def _normalize_p(self, p):
        return max(min(p, 1), 0)


    def _get_random_p(self, n, rng):
        # uniformly distributed random initial p1
        # ensuring that initial probability is not to close to 0 or 1
        return np.clip(rng.uniform(size = n), 1e-2, 1 - 1e-2)
    

    def _get_random_params(self, n, rng, *data):
        # needs to be implemented for each child
        raise NotImplementedError


    def _run_em_algo_multi(self, *data, seed = None, n_init = 20, 
                           tol = 1e-4, max_iter = 100, optimizer = "Powell"):

        # random generation of parameters for multiple initializations
        rng = np.random.default_rng(seed = seed)
        random_p = self._get_random_p(n_init, rng)
        random_params1 = self._get_random_params(n_init, rng, *data)
        random_params2 = self._get_random_params(n_init, rng, *data)

        params_arr = np.empty(shape = (n_init, 1 + random_params1.shape[1] + random_params2.shape[1]))
        LL_list = []

        # parrelell execution of EM algorithm with different initializations
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._run_em_algo, random_p[i], tuple(random_params1[i]), tuple(random_params2[i]),
                                       data, tol = tol, max_iter = max_iter, optimizer = optimizer) 
                                       for i in range(n_init)]
            
        # unpacking results
        for i, future in enumerate(futures):
            LL, p1, params1, params2 = future.result()
            LL_list.append(LL)
            params_arr[i] = [p1] + list(params1) + list(params2)

        # selecting best initialization
        best_index = np.argmin(LL_list)
        best_LL = LL_list[best_index]

        return best_LL, *params_arr[best_index]
    
    
    def _mini_post_process_fit(self, LL, n):
        self.is_fit = True; self.LL = LL; self.n = n

        self.aic = self._calc_aic(LL, self.k)
        self.bic = self._calc_bic(LL, n)


    def _run_em_algo(self, p1, params1, params2, data,
                     tol = 1e-4, max_iter = 100, optimizer = "Powell"):
        
        i = 0; LL = 0
        while i < max_iter:
            gamma1, gamma2 = self._e_step(p1, params1, params2, *data)
            new_LL, new_p1, new_params1, new_params2 = self._m_step(gamma1, gamma2, params1, params2, 
                                                                    *data, optimizer = optimizer)
            
            # stopping condition 2
            if np.abs(new_LL - LL) < tol:
                return new_LL, new_p1, new_params1, new_params2
            
            p1, params1, params2, LL = new_p1, new_params1, new_params2, new_LL

        # stopping condition 1, hit max iterations
        return LL, p1, params1, params2


    def _e_step(self, p1, params1, params2, *data):

        gamma1 = p1 * self._base_model._pdf(*data, *params1)
        gamma2 = (1 - p1) * self._base_model._pdf(*data, *params2)
        gamma_sum = gamma1 + gamma2

        # normalizing to sum to 1
        return gamma1 / gamma_sum, gamma2 / gamma_sum
    

    def _m_step(self, gamma1, gamma2, params1, params2, *data, optimizer = "Powell"):

        new_p1 = np.mean(gamma1)

        f1 = self._base_model._get_weighted_obj_func(gamma1, *data)
        f2 = self._base_model._get_weighted_obj_func(gamma2, *data)

        # for marginal distributions, not using Scipy fit but generic implementation in Base

        results1 = self._base_model._fit(f1, params1, self._base_model.param_bounds,
                                        optimizer = optimizer)
        
        results2 = self._base_model._fit(f2, params2, self._base_model.param_bounds,
                                         optimizer = optimizer)
        
        new_LL = -1 * (results1.fun + results2.fun)
        return new_LL, new_p1, tuple(results1.x), tuple(results2.x)
    

    def _mixture_pdf(self, p1, params1, params2, *data):
        return p1 * self._base_model._pdf(*data, *params1) + (1 - p1) * self._base_model._pdf(*data, *params2)
    

    def _mixture_cdf(self, p1, params1, params2, *data):
        return p1 * self._base_model._cdf(*data, *params1) + (1 - p1) * self._base_model._cdf(*data, *params2)
    
    
    def _get_extra_text(self):
        brent_string = "Inverse CDF (PPF) solved using Brent's Method"
        if self.is_fit:
            return ["Standard Errors not available", brent_string]
        else:
            return [brent_string]
    



    

    


