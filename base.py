import utils

import numpy as np
from scipy.optimize import minimize
from scipy import stats
from statsmodels.tools.numdiff import approx_hess3, approx_fprime
from statsmodels.iolib.summary import Summary, fmt_params, fmt_2cols
from statsmodels.iolib.table import SimpleTable



class Base:
    def __init__(self, model_name, initial_param_guess, param_bounds, param_names, params):

        self.model_name = model_name

        self.initial_param_guess = initial_param_guess
        self.param_bounds = param_bounds
        self.param_names = param_names

        self._validate_params(params, param_names, param_bounds)
        self._set_params(params)

        self.k = len(params)
        self.is_fit = False
        self.robust_cov = None

        # these variables are calculated post-fit
        # initializing them here for clarity of what variables are included in this object
        self.n = np.nan
        self.LL = np.nan; self.aic = np.nan; self.bic = np.nan
        self.hess_matrix = np.full((self.k, self.k), np.nan); self.se = np.full(self.k, np.nan)
        self.conf_int = np.full((self.k, 2), np.nan); self.t = np.full(self.k, np.nan); self.p = np.full(self.k, np.nan)


    def _validate_params(self, params, param_names, param_bounds):
        for i, param in enumerate(params):
            if param < param_bounds[i][0] or param > param_bounds[i][1]:
                print(param_names[i], "parameter outside of valid boundaries")
                raise SyntaxError


    def _set_params(self, params):
        self.params = params
        self.params_dict = {k:v for k, v in zip(self.param_names, params)}


    # has to handle multiple datapoints because bivariate copulas have two data inputs

    def _fit(self, f, initial_param_guess, param_bounds, optimizer = "Powell"):
        # defualt mle optimization (aka canonical likelihood implementation)
        return minimize(f, initial_param_guess, bounds = param_bounds, method = optimizer)
    

    def _get_objective_func(self, *data):
        return lambda params: -1 * self._log_likelihood(*data, *params)
    

    def _get_gradient_func(self, opt_params_arr):
        return lambda data: approx_fprime(opt_params_arr, lambda params: self._log_likelihood(*data, *params), epsilon = 1e-5)
    

    def _get_inv_hessian_matrix(self, opt_params_arr, objective_func):
        return np.linalg.inv(approx_hess3(opt_params_arr, objective_func))
    

    def _se_from_matrix(self, matrix):
        return np.sqrt(np.diag(matrix))
    

    def _get_se(self, opt_params_arr, objective_func):
        return self._se_from_matrix(self._get_inv_hessian_matrix(opt_params_arr, objective_func))
    

    def _get_robust_se(self, data_arr, opt_params_arr, objective_func):

        inv_hess_matrix = self._get_inv_hessian_matrix(opt_params_arr, objective_func)
        grad_func = self._get_gradient_func(opt_params_arr)

        S = np.zeros((len(opt_params_arr), len(opt_params_arr)))

        if data_arr.ndim == 1:
            data_arr = data_arr.reshape(-1, 1)

        for x in data_arr:
            score_vec = grad_func(x)
            S += np.outer(score_vec, score_vec)

        return self._se_from_matrix(inv_hess_matrix @ S @ inv_hess_matrix)


    def _get_aic(self, LL, k):
        return 2 * k - LL
    

    def _get_bic(self, LL, n):
        return 2 * np.log(n) - 2 * LL


    def _post_process_fit(self, data_arr, opt_params_arr, objective_func, robust_cov = True):
        
        self.is_fit = True
        self.LL = -1 * objective_func(opt_params_arr)
        self.robust_cov = robust_cov

        # saving most recent 
        self.k = len(opt_params_arr)
        self.n = len(data_arr)

        self.aic = 2 * self.k - self.LL
        self.bic = 2 * np.log(self.n) - 2 * self.LL

        if robust_cov:
            self.se = self._get_robust_se(data_arr, opt_params_arr, objective_func)
        else:
            self.se = self._get_se(opt_params_arr, objective_func)

        t_factor = stats.t.ppf(0.975, df = self.n)

        # 95% confidence intervals
        self.conf_int = np.array([
            opt_params_arr - self.se * t_factor, # lower
            opt_params_arr + self.se * t_factor, # upper
            
        ]).T

        # initial param guess is assumed to be null hypothesis param (or independence)
        self.t = (opt_params_arr - np.array(self.initial_param_guess)) / self.se
        self.p = 2 * stats.t.sf(np.abs(self.t), df = self.n - self.k)

        # setting params
        self._set_params(tuple(opt_params_arr))

    def summary(self):
        return self._summary([self], [None])


    def _summary(self, model_objects, model_names):
        # returns a printable "statsmodel like" summary of the model
        # I heavily relied on the code from the Arch package which also used the statsmodel iolibrary to implement a summary

        # if not fit, raise error

        top_left, top_right = self._get_top_summary_table()

        stubs = []; vals = []
        for stub, val in top_left:
            stubs.append(stub); vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=self.summary_title + " Summary", stubs=stubs)


        smry = Summary()
        
        fmt = fmt_2cols
        fmt["data_fmts"][1] = "%18s"
        
        top_right = [(("  " + k), v) for k, v in top_right]
        stubs = []; vals = []
        for stub, val in top_right:
            stubs.append(stub); vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        if len(self.params) <= 0:
            return smry

        for model_obj, model_name in zip(model_objects, model_names):
            smry.tables.append(self.make_summary_table(model_obj, model_name, fmt_params))
        
        #extra_text = ["Covariance Method: robust",]
        #if self.robust_cov is not None:
        #     smry.add_extra_txt(["Covariance Method: {}".format("robust" if self.robust_cov else "classical")]) 
        
        return smry
    
    def make_summary_table(self, model_obj, model_name, fmt_params):
        data = []
        for _, (param, guess, std_err, t_val, p_val, ci) in enumerate(zip(model_obj.params, model_obj.initial_param_guess, model_obj.se, model_obj.t, model_obj.p, model_obj.conf_int)):
            data.append([
                utils.format_func(param, 2, 4),
                utils.format_func(std_err, 10, 4),
                utils.format_func(guess, 2, 2),
                utils.format_func(t_val, 10, 4),
                utils.format_func(p_val, 10, 4),
                f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            ])
            
        fmt_params_table = fmt_params.copy()
        fmt_params_table["colwidths"] = 8

        return SimpleTable(
            data,
            headers=["coef", "std err", "h0", "t", "P>|t|", "95% Conf. Int."],
            stubs=model_obj.param_names,
            title="{} Parameter Estimates".format(model_name) if model_name is not None else "Parameter Estimates",
            txt_fmt=fmt_params_table
        )