import bivariate
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class Mixture:
    def __init__(self, p, models):

        # implement check that p and models have same length

        self.p = self._normalize_p(p)
        self.models = models
        self.model_params = [model.params for model in self.models]
        self.m = len(models)


    def _normalize_p(self, p):
        return p / np.sum(p)
    

    def _get_random_p(self, rng, n):
        return rng.dirichlet(np.ones(self.k), size = n)
    

    def _set_params(self, p, model_params):
        self.p = self._normalize_p(p)

        for i, params in enumerate(model_params):
            self.models[i]._set_params(params)
    

    def _get_random_model_params(self, data, rng, n):
        multi_list = []
        for _ in range(n):
            multi_list.append([model._generate_random_params(data, rng, 1)[0] for i, model in enumerate(self.models)])

        return multi_list
    

    def _get_weighted_obj_func(self, gamma, model, *data):
        return lambda params: -np.sum(gamma * model._logpdf(*data, *params))
    

    def fit(self, *data, initial_p_guess = None, initial_model_params_guess = None, n_init = 20, tol = 1e-4, max_iter = 100, optimizer = "Powell", seed = None):

        # check if they are arrays and same shape
        if initial_p_guess is not None and initial_model_params_guess is not None:
            p, model_params, LL = self._run_em_algo(initial_p_guess, initial_model_params_guess, *data, tol = tol, 
                                                    max_iter = max_iter, optimizer = "Powell")
            
        else:
            # random initial parameters
            rng = np.random.default_rng(seed = seed)
            initial_p = self._get_random_p(rng, n = n_init)
            initial_model_params = self._get_random_model_params(*data, rng = rng, n = n_init)
            
            p_list = []; params_list = []; LL_list = []

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._run_em_algo, initial_p[i], initial_model_params[i], *data, 
                                             tol = tol, max_iter = max_iter, optimizer = optimizer) for i in range(n_init)]

            for i, future in enumerate(futures):
                res = future.result()

                p_list.append(res[0]);  params_list.append(res[1]); LL_list.append(res[2])


            best_index = np.argmin(LL_list)
            self.LL = LL_list[best_index]
            self._set_params(p_list[best_index], params_list[best_index])

    

    def _run_em_algo(self, p, model_params, *data, tol = 1e-4, max_iter = 100, m_method = "MLE", optimizer = "Powell"):
        
        i = 0
        LL = 0

        while i < max_iter:
            gamma = self._e_step(p, model_params, *data)
            new_p, new_model_params, new_LL = self._m_step(gamma, model_params, *data, optimizer = optimizer)

            # convergence criterion
            if (LL - new_LL) ** 2 < tol:
                return new_p, new_model_params, new_LL
            
            # setting parameters
            p, model_params, LL = new_p, new_model_params, new_LL
            i += 1

        # if hits max iteration limit
        # should I raise warning?
        return p, model_params, LL



    def _e_step(self, p, model_params, *data):
        gammas = np.zeros(shape = (data.shape[0], self.m))

        for i in range(self.m):
            gammas[:, i] = p[i] * self.models[i]._pdf(*data, *model_params[i])

        return gammas / np.sum(gammas, axis = 1, keepdims = True)
    

    def _m_step(self, gamma, model_params, *data, optimizer = "Powell"):
        new_p = np.mean(gamma, axis = 0)

        new_model_params = []
        new_LL = []

        for i in range(self.m):
            model = self.models[i]
            f_i = self._get_weighted_obj_func(gamma[:, i], model, *data)
            results_i = model._fit(f_i, model_params[i], model.param_bounds,
                                   optimizer = optimizer)
            
            new_model_params.append(results_i.x)
            new_LL.append(results_i.fun)

        return new_p, new_model_params, np.sum(new_LL)
    


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
        return self._pdf(u1_valid, u2_valid, self.models, self.p)
    

    def _pdf(self, u1, u2, models, p):
        # both the density and the cumulative density are probability weighted
        return np.dot([c._pdf(u1, u2, *c.params) for c in models], p)
    
    # what is the log pdf of a mixture copula?
    
    
    def cdf(self, u1, u2, adj = 1e-4):
        u1_valid, u2_valid = self._handle_uu_input(u1, u2, adj = adj)
        return self._reshape_wrapper(u1_valid, u2_valid, self._cdf, self.models, self.p)
    

    def _cdf(self, u1, u2, models, p):
        # both the density and the cumulative density are probability weighted
        return np.dot([c._cdf(u1, u2, *c.params) for c in models], p)
    

    def conditional_quantile(self, u1, q, adj = 1e-5):
        raise NotImplementedError
    

    def _conditional_quantile(self, u1, q):
        raise NotImplementedError

        

class CopulaMixture:
    def __init__(self):
        pass

    def simulate(self, n = 1000, seed = None):
        rng = np.random.default_rng(seed = seed)

        model_indices = rng.random.choice(np.array(range(self.k)), p = self.p, replace = True, size = n)
        
        u1 = rng.uniform(size = n)
        u2 = np.empty(shape = n)
        q = rng.uniform(size = n)

        for i, model_idx in enumerate(model_indices):
            m = self.models[model_idx]
            u2[i] = m._quantile(u1[i], q[i], *m.params)

        return u1, u2

