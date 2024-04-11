import numpy as np
from datetime import datetime

from base import Base
import utils


class JointModel(Base):
    def __init__(self, copula, marginal1, marginal2):
        self.copula = copula
        self.marginal1 = marginal1
        self.marginal2 = marginal2

        self.k = self.copula.k + self.marginal1.k + self.marginal2.k; self.n = np.nan
        self.LL = np.nan; self.aic = np.nan; self.bic = np.nan
        self.params = [None]

        self.summary_title = "Bivariate Model"

        # not initializing parent class

    def fit(self, x1, x2, method = "CMLE", copula_fit_kwargs = {}, marginal1_fit_kwargs = {}, marginal2_fit_kwargs = {}):
        # error handling

        # copula and marginals can be fit seperately
        self.copula.fit(utils.rank_transform(x1), utils.rank_transform(x2), **copula_fit_kwargs)
        self.marginal1.fit(x1, **marginal1_fit_kwargs)
        self.marginal2.fit(x2, **marginal2_fit_kwargs)

        # LL, AIC, and BIC at the joint model level will only be defined if this fit method is called
        # otherwise, possibility of marginals or copulas being fit with different datasets
        self._post_process_fit(len(x1))


    def _post_process_fit(self, n):
        # only called if fit is called at the joint level

        self.LL = self.copula.LL + self.marginal1.LL + self.marginal2.LL
        self.n = n

        self.aic = self._get_aic(self.LL, self.k)
        self.bic = self._get_bic(self.LL, self.n)


    def pdf(self, x1, x2):
        # insert formula here
        # error handlingx
        return np.exp(self.logpdf(x1, x2))

        
    def logpdf(self, x1, x2):
        # error handling
        log_density1, log_density2 = self.marginal1.logpdf(x1), self.marginal2.logpdf(x2)
        u1, u2 = self.marginal1.cdf(x1), self.marginal2.cdf(x2)
        return log_density1 + log_density2 + self.copula.logpdf(u1, u2)
    

    def cdf(self, x1, x2):
        # insert formula here
        # error handling
        u1, u2 = self.marginal1.cdf(x1), self.marginal2.cdf(x2)
        return self.copula.cdf(u1, u2)


    def simulate(self, n = 1000, seed = None):
        u1, u2 = self.copula.simulate(n = n, seed = seed)
        x1, x2 = self.marginal1.ppf(u1), self.marginal2.ppf(u2)
        return x1, x2
    

    def conditional_ppf(self, x1, q, adj = 1e-5):

        u1 = self.marginal1.cdf(x1)
        u2 = self.copula.conditional_ppf(u1, q, adj = adj)
        return self.marginal2.ppf(u2, adj = adj)
    

    def _get_top_summary_table(self):
        now = datetime.now()
        top_left = [
            ("Method:", "MLE"),("Num. Params:", self.k), ("Num. Obs:", self.n),
            ("Date:", now.strftime("%a, %b %d %Y")),("Time:", now.strftime("%H:%M:%S")), ("", ""),
        ]

        top_right = [
            ("Log-Likelihood:", utils.format_func(self.LL, 10, 4)), ("AIC:", utils.format_func(self.aic, 10, 4)),
            ("BIC:", utils.format_func(self.bic, 10, 4)), (" ", " "), (" ", " "), (" ", " "),
            ("", ""),
        ]

        return top_left, top_right
    
    def summary(self):
        return self._summary([self.copula, self.marginal1, self.marginal2], ["Copula", "Marginal1", "Marginal2"])
    



    

    


    