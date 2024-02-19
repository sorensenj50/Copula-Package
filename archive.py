from scipy import stats
from marginals import MarginalDist, UndefinedCDF, Marginal
import utils
import numpy as np
from bivariate import BivariateCopula
from scipy.integrate import cumulative_trapezoid
from scipy.special import beta


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




class SkewedGeneralizedT:
    def __init__(self, mu = 0, sigma = 1, lam = 0, p = 2, q = 1e4, adj = 1e-4):
        super().__init__(None, model_name = "SkewedGeneralizedT", 
                         initial_param_guess = [0, 1, 0, 2, 1000], 
                         param_bounds = [(-np.inf, np.inf), (adj, np.inf), (-1 + adj, 1 - adj), (adj, np.inf), (adj, np.inf)],
                         param_names = ["mu", "sigma", "lambda", "p", "q"],
                         params = [mu, sigma, lam, p, q])


    def _pdf(self, x, mu, sigma, lam, p, q):

        # common terms
        beta_term1 = beta(1/p, q)
        beta_term2 = beta(2/q, q - 1/p)
    
        # v variable
        v1 = np.power(q, -1/p)
        v2 = 1 + 3 * np.power(lam, 2)
        v3 = beta(3/p, q - 2/p) / beta_term1
        v4 = (4 * np.power(lam, 2)) * np.power(beta_term2 / beta_term1, 2)
        v = v1 / np.sqrt(v2 * v3 - v4)
    
        # m variable
        sigma_term = 2 * sigma * v * np.power(q, 1/p)
        m = lam * sigma_term * beta_term2 / beta_term1
    
    
        # the final pdf
        x_term = x - mu + m
        pdf1 = sigma_term * beta_term1
        pdf2 = np.power(np.abs(x_term), p)
        pdf3 = q * np.power(v * sigma, p) * np.power(1 + lam * np.sign(x_term), p)
    
        return p / (pdf1 * np.power(1 + pdf2 / pdf3, 1/p + q))
    
    def _logpdf(self, x, *params):
        return np.log(self._pdf(x, *params))
    

class GeneralizedFamily(UndefinedCDF):

    def __init__(self, rv_obj, model_name, initial_param_guess, param_bounds, param_names, params):

        # preemptively fitting CDF and PPF interpolations
        # fitting with new params will override
        self._set_cdf_ppf(params[0], params[1], *params)
        super().__init__(rv_obj, model_name, initial_param_guess, param_bounds, param_names, params)


    def _post_process_fit(self, opt_params, objective_func, n):
        super()._post_process_fit(opt_params, objective_func, n)
        self._set_cdf_ppf(opt_params[0], opt_params[1], *opt_params)

class UndefinedCDF(Marginal):

    def _set_cdf_ppf(self, loc, scale, *params, n = 1000):

        bounds = utils.find_x_bounds(loc, scale, self._pdf, *params)
        x_range = np.linspace(bounds[0], bounds[1], n)

        pdf_values = self._pdf(x_range, *params)
        
        # cumulative numerical integration
        cdf_values = cumulative_trapezoid(pdf_values, x_range, initial = 0)

        self.interp1d_cdf_func, self.interp1d_ppf_func = utils.build_cdf_interpolations(x_range, cdf_values)

    
    # redefining CDF and PPF methods to not only passing of parameters
    # CDF and PPF operate on set interpolations of numerical integration
        
    def cdf(self, x):
        # error handling
        return self._cdf(x)
    

    def _cdf(self, x):
        return self.interp1d_cdf_func(x)
    

    def ppf(self, u):
        # error handling
        return self._ppf(u)
    

    def _ppf(self, u):
        return self.interp1d_ppf_func(u)
    

