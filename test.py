import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import special


import importlib
from copulas import bivariate_copula
import base, copulas, marginals, utils, tests, plots, joint_model, mixture

from copulas import archimedean


plt.rcParams['figure.figsize'] = (6.4, 4.8)
plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.dpi'] = 75


m = marginals.StandardSkewedT(eta = 10, lam = 0.75)

print(m.summary())