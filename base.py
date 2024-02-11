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

        # these variables are calculated post-fit
        # initializing them here for clarity of what variables are included in this object
        self.n = None
        self.LL = None; self.aic = None; self.bic = None
        self.hess_matrix = np.array([]); self.se = np.array([])
        self.conf_int = np.array([]); self.t = np.array([]); self.p = np.array([])


    def _validate_params(self, params, param_names, param_bounds):
        for i, param in enumerate(params):
            if param < param_bounds[i][0] or param > param_bounds[i][1]:
                print(param_names[i], "parameter outside of valid boundaries")
                raise SyntaxError


    def _set_params(self, params):
        self.params = params
        self.params_dict = {k:v for k, v in zip(self.param_names, params)}


    def _get_objective_func(self, *data):
        return lambda params: -1 * self._log_likelihood(*data, *params)
    

    def _fit(self, f, initial_param_guess, param_bounds, optimizer = "Powell"):
        
        # defualt mle optimization (aka canonical likelihood implementation)
        return minimize(f, initial_param_guess, bounds = param_bounds, method = optimizer)


    def _post_process_fit(self, opt_params, objective_func, n):
        
        self.is_fit = True
        self.LL = -1 * objective_func(opt_params)

        # saving most recent 
        self.k = len(opt_params)
        self.n = n

        self.aic = 2 * self.k - self.LL
        self.bic = 2 * np.log(self.n) - 2 * self.LL

        self.hess_matrix = approx_hess3(opt_params, objective_func)

        self.se = np.sqrt(np.diag(np.linalg.inv(self.hess_matrix)))

        t_factor = stats.t.ppf(0.975, df = n)

        self.conf_int = np.array([
            opt_params - self.se * t_factor, # lower
            opt_params + self.se * t_factor, # upper
            
        ]).T

        # initial param guess is assumed to be null hypothesis param (or independence)
        self.t = (opt_params - np.array(self.initial_param_guess)) / self.se
        self.p = 2 * stats.t.sf(np.abs(self.t), df = n - self.k)

        # setting params
        self._set_params(tuple(opt_params))


    def summary(self):
        # returns a printable "statsmodel like" summary of the model
        # I heavily relied on the code from the Arch package which also used the statsmodel iolibrary to implement a summary

        # if not fit, raise error

        if not self.is_fit:
            raise SyntaxError
        
        

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
    
        data = []
        for i, (param, guess, std_err, t_val, p_val, ci) in enumerate(zip(self.params, self.initial_param_guess, self.se, self.t, self.p, self.conf_int)):
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

        param_table = SimpleTable(
            data,
            headers=["coef", "std err", "h0", "t", "P>|t|", "95% Conf. Int."],
            stubs=self.param_names,
            title="Parameter Estimates",
            txt_fmt=fmt_params_table
        )
        smry.tables.append(param_table)
        extra_text = ["Covariance Method: robust",]
        smry.add_extra_txt(extra_text)
        return smry


    
