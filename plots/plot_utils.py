import matplotlib.pyplot as plt
import numpy as np
import utils


def _handle_ax(ax = None, **f_kwargs):
    if ax is None:
        f, ax = plt.subplots(**f_kwargs)
    
    return ax


def _merge_kw(kwargs_1, kwargs_2):
    return kwargs_1 | kwargs_2


def _get_joint_model_x_range(joint_model, adj = 1e-4, range_num = 250):
    min1 = joint_model.marginal1.ppf(adj); max1 = joint_model.marginal1.ppf(1 - adj)
    min2 = joint_model.marginal2.ppf(adj); max2 = joint_model.marginal2.ppf(1 - adj)

    x_range1 = utils.get_x_range(low = min1, high = max1, range_num = range_num)
    x_range2 = utils.get_x_range(low = min2, high = max2, range_num = range_num)

    X1, X2 = np.meshgrid(x_range1, x_range2)

    return X1, X2


def _cumulative_level(density):
    # Flatten the array and get the indices that would sort it
    flattened = density.flatten()
    sorted_indices = np.argsort(flattened)
    
    sorted_arr = flattened[sorted_indices]
    cumsum_arr = np.cumsum(sorted_arr)
    cumsum_arr /= cumsum_arr[-1] # normalizing to have sum of 1
    
    # map back to original indice
    cum = np.empty_like(flattened)
    cum[sorted_indices] = cumsum_arr
    
    return cum.reshape(density.shape)

    