import numpy as np
from scipy import stats, integrate
from .bivariate_copula import BivariateCopula


class Archimedean(BivariateCopula):
    def __init__(self, rotation, model_name, *args, **kwargs):

        # model name has to be set before rotation
        self.model_name = model_name
        self._set_rotation(rotation)
        super().__init__(model_name, *args, **kwargs)

        # setting rotation again because parent class init overrides model_name
        self._set_rotation(rotation)


    def _set_rotation(self, rotation):
        self._check_rotation(rotation)
        self.rotation = rotation

        if rotation > 0:
            self.model_name += " (Rot. {})".format(self.rotation)
        
        # assigning rotation transformation function
            
        if rotation == 0:
            self._pdf_rot_func = lambda u1, u2: (u1, u2)
            self._cdf_rot_func = lambda u1, u2, C: C
            self._cond_rot_func1 = lambda u1, q: (u1, q)
            self._cond_rot_func2 = lambda u2: u2
            self._corr_rot_func = lambda x: x
            self._upper_tail_rot = self._unrotated_upper_tail_dependance
            self._lower_tail_rot = self._unrotated_lower_tail_dependance
        
        elif rotation == 90:
            self._pdf_rot_func = lambda u1, u2: (1 - u2, u1)
            self._cdf_rot_func = lambda u1, u2, C: u1 - C
            self._cond_rot_func1 = lambda u1, q : (u1, 1 - q)
            self._cond_rot_func2 = lambda u2: 1 - u2
            self._corr_rot_func = lambda x: -x
            self._upper_tail_rot = self._unrotated_lower_upper_dependance
            self._lower_tail_rot = self._unrotated_upper_lower_dependance

        elif rotation == 180:
            self._pdf_rot_func = lambda u1, u2: (1 - u1, 1 - u2)
            self._cdf_rot_func = lambda u1, u2, C: u1 + u2 -1 + C
            self._cond_rot_func1 = lambda u1, q: (1 - u1, 1 - q)
            self._cond_rot_func2 = lambda u2: 1 - u2
            self._corr_rot_func = lambda x: x
            self._upper_tail_rot = self._unrotated_lower_tail_dependance
            self._lower_tail_rot = self._unrotated_upper_tail_dependance

        elif rotation == 270:
            self._pdf_rot_func = lambda u1, u2: (u2, 1 - u1)
            self._cdf_rot_func = lambda u1, u2, C: u2 - C
            self._cond_rot_func1 = lambda u1, q: (1 - u1, q)
            self._cond_rot_func2 = lambda u2: u2
            self._corr_rot_func = lambda x: -x
            self._upper_tail_rot = self._unrotated_upper_lower_dependance
            self._lower_tail_rot = self._unrotated_lower_upper_dependance
        

    def _check_rotation(self, rotation):
        if rotation not in [0, 90, 180, 270]:
            # input error
            raise SyntaxError
        
    # default is zero, implementation can be overriden
    def _unrotated_upper_tail_dependance(self, *params):
        return 0
    
    
    def _unrotated_lower_tail_dependance(self, *params):
        return 0
    
   
    def _unrotated_upper_lower_dependance(self, *params):
        return 0
    

    def _unrotated_lower_upper_dependance(self, *params):
        return 0
    

    def _upper_taiL_dependance(self, theta):
        return self._upper_tail_rot(theta)
    

    def _lower_tail_dependance(self, theta):
        return self._lower_tail_rot(theta)
    


class Clayton(Archimedean):
    def __init__(self, theta = 1e-4, rotation = 0, adj = 1e-4):
        super().__init__(rotation = rotation, model_name = "Clayton", family_name = "Archimedean", 
                         initial_param_guess = [adj],
                         param_bounds = [(adj, np.inf)], param_names = ("theta",), params = (theta,))
        
    
    def _cdf(self, u1, u2, theta):

        # rotation variables if necessary
        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)
        C = np.power((np.power(rot_u1, -theta) + np.power(rot_u2, -theta) - 1), -1/theta)

        # passng original variables for additional handling of rotation
        return self._cdf_rot_func(u1, u2, C)


    def _logpdf(self, u1, u2, theta):
        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)

        log_1 = np.log(theta + 1)
        log_2 = (-2 - 1/theta) * np.log(np.power(rot_u1, -theta) + np.power(rot_u2, -theta) - 1)
        log_3 = (-theta - 1) * (np.log(rot_u1) + np.log(rot_u2))

        return log_1 + log_2 + log_3
    

    def _conditional_cdf(self, u1, u2, theta):
        rot_u1, rot_u2 = self._cond_rot_func1(u1, u2)

        A = np.power(u1, -(1 + theta))
        B = np.power(np.power(rot_u1, -theta) + np.power(rot_u2, -theta) - 1, -(1 + theta) / theta)

        return self._cond_rot_func2(A * B)
    

    def _conditional_ppf(self, u1, q, theta, adj = 1e-4):
        rot_u1, rot_q = self._cond_rot_func1(u1, q)
        return self._cond_rot_func2(np.power((1 + np.power(rot_u1, -theta) * (np.power(rot_q, -theta/(1+theta)) -1)), -1/theta))
    
    
    def _params_to_tau(self, theta):
        return self._corr_rot_func(theta / (theta + 2))
    

    def _tau_to_params(self, tau):
        return tuple(2 * tau * (1 / (1 - tau)))
    

    def _unrotated_lower_tail_dependance(self, theta):
        return 2 ** (-1 / theta)
    


class Frank(Archimedean):
    def __init__(self, theta = 1e-4, rotation = 0, adj = 0):
        super().__init__(rotation = rotation, model_name = "Frank", family_name = "Archimedean",
                         initial_param_guess = [adj],
                         param_bounds = [(adj, np.inf)], param_names = ("theta",),
                         params = (theta,))
        
    def _g(self, u, theta):
        # helper function used in pdf and cdf
        return np.exp(-theta * u) - 1
    

    def _D(self, theta, k = 1):
        # numerical implementation of order k Debye function

        integrand = lambda t: np.power(t, k) / (np.exp(t) - 1)
        integral, _ = integrate.quad(integrand, 0, theta)
        return k * np.power(theta, -k) * integral 


    def _cdf(self, u1, u2, theta):

        # independance copula if theta is 0
        if theta == 0:
            return u1 * u2
        

        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)
        num = self._g(rot_u1, theta) * self._g(rot_u2, theta)
        denom = self._g(1, theta)
        C = -1/theta * np.log(1 + num / denom)

        return self._cdf_rot_func(u1, u2, C)
    

    def _pdf(self, u1, u2, theta):
        # independance copula if theta is 0
        # handles number or array input
        if theta == 0:
            return u1 * 0 + 1

        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)    
        num = (- theta * self._g(1, theta)) * (1 + self._g(rot_u1 + rot_u2, theta))
        denom = np.power(self._g(rot_u1, theta) * self._g(rot_u2, theta) + self._g(1, theta), 2)

        return num / denom
    

    def _logpdf(self, u1, u2, theta):
        return np.log(self._pdf(u1, u2, theta))
    

    def _conditional_cdf(self, u1, u2, theta):
        if theta == 0:
            return u2
        
        rot_u1, rot_u2 = self._cond_rot_func1(u1, u2)
        
        num = self._g(rot_u1, theta) * self._g(rot_u2, theta) + self._g(rot_u2, theta)
        denom = self._g(rot_u1, theta) * self._g(rot_u2, theta) + self._g(1, theta)

        return self._cond_rot_func2(num / denom)
    
    
    def _conditional_ppf(self, u1, q, theta, adj=1e-4):
        rot_u1, rot_q = self._cond_rot_func1(u1, q)

        if theta == 0:
            return rot_q

        denom = -theta
        num = np.log(1 + (rot_q * self._g(1, theta)) / (1 + self._g(rot_u1, theta) * (1 - rot_q)))
        return self._cond_rot_func2(num / denom)
    

    def _params_to_tau(self, theta):
        # Joe 4.5.1

        if theta == 0:
            return 0

        return self._corr_rot_func(1 + 4 * (1 / theta) * (self._D(theta) - 1))
    
    def _params_to_rho(self, theta):
        # Joe 4.5.1

        if theta == 0:
            return 0
        
        return self._corr_rot_func(1 + 12 / theta * (self._D(theta, k = 2) - self._D(theta, k = 1)))
    


class Gumbel(Archimedean):
    def __init__(self, theta = 1, rotation = 0):
        super().__init__(rotation = rotation, model_name = "Gumbel", family_name = "Archimedean",
                         initial_param_guess = [1], 
                         param_bounds = [(1, np.inf)], param_names = ("theta",),
                         params = (theta,))


    def _A(self, u1, u2, theta):
        # helper A function
        # Carol Alexander II.6.54

        return np.power(np.power(-np.log(u1), theta) + np.power(-np.log(u2), theta), 1/theta)
    

    def _B(self, w, theta):
        # helper B function, see Joe 4.8.1
        # note that Joe's "A" function is different from one used above from Carol Alexander

        return np.power(np.power(w, theta) + np.power(1 - w, theta), 1/theta)


    def _cdf(self, u1, u2, theta):
        # rotating inputs if necessary
        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)
        C = np.exp(-self._A(rot_u1, rot_u2, theta))

        # final transformation on cdf using original inputs
        return self._cdf_rot_func(u1, u2, C)
    

    def _conditional_cdf(self, u1, u2, theta):
        # conditional of u2 given u1
        rot_u1, rot_u2 = self._cond_rot_func1(u1, u2)    

        prod1 = 1/rot_u1
        prod2 = np.power(-np.log(rot_u1), theta - 1)
        prod3 = np.power(np.power(-np.log(rot_u1), theta) + np.power(-np.log(rot_u2), theta), (1 - theta)/theta)
        return self._cond_rot_func2(prod1 * prod2 * prod3 * np.exp(-self._A(rot_u1, rot_u2, theta)))
    

    def _logpdf(self, u1, u2, theta):
 
        # rotating inputs if necessary
        rot_u1, rot_u2 = self._pdf_rot_func(u1, u2)

        A = self._A(rot_u1, rot_u2, theta)

        log_1 = np.log(A + theta - 1)
        log_2 = (1-2*theta) * np.log(A)
        log_3 = -A - (np.log(rot_u1) + np.log(rot_u2))
        log_4 = (theta - 1) * (np.log(-np.log(rot_u1)) + np.log(-np.log(rot_u2)))

        return log_1 + log_2 + log_3 + log_4
    

    def _params_to_tau(self, theta):
        # Joe 4.8.1
        return self._corr_rot_func(1 - 1 / theta)
    

    def _params_to_rho(self, theta):
        # numerical integration
        # see Joe 4.8.1

        integral, _ = integrate.quad(lambda w: np.power(1 + self._B(w, theta), -2), 0, 1)
        return self._corr_rot_func(12 * integral - 3)
    

    def _tau_to_params(self, tau):
        return tuple(1 / (1 - tau))


    def _unrotated_upper_tail_dependance(self, theta):
        return 2 - np.power(2, 1 / theta)
    

    def _get_extra_text(self):
        return super()._get_extra_text() + ["Conditional PPF solved using Brent's Method"]