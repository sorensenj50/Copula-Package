import numpy as np
from base import Base


class BivariateModel(Base):
    def __init__(self, copula, marginal1, marginal2):
        self.copula = copula
        self.marginal1 = marginal1
        self.marginal2 = marginal2

    # don't need core and inner implementations because this will not have child classes
        
    def pdf(self, x1, x2):
        # insert formula here
        # error handling
        return np.exp(self._logpdf(x1, x2))

        
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
    
    
    def conditional_quantile(self, x1, q):
        u1 = self.marginal1.cdf(x1)
        u2 = self.copula.conditional_quantile(u1, q)
        return self.marginal2.cdf(u2)
    



    

    


    