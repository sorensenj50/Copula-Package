import bivariate
import utils

import matplotlib.pyplot as plt
import numpy as np

def check_copula_obj(copula_obj):
    if not isinstance(copula_obj, bivariate.BivariateCopula):
        # create error for this
        raise SyntaxError
    
    

def copula_3d_surf(copula_obj, ax = None, adj = 1e-2, range_num = 100, cmap = "viridis", 
                   elev = None, azim = None, antialiased = False, **kwargs):
    
    check_copula_obj(copula_obj)

    if ax is None:
        f, ax = plt.subplots(subplot_kw={"projection": "3d"})

    u1, u2 = utils.get_u_grid(adj = adj, range_num=range_num)

    density = copula_obj.pdf(u1, u2)

    _ = ax.plot_surface(u1, u2, density, cmap = cmap, antialiased = antialiased)
    ax.set_zlim(0, None)
    ax.view_init(elev = elev, azim = azim)
    
    return ax

# remove optional density parameter
def copula_contour(copula_obj, density = None, ax = None, adj = 1e-2, range_num = 100, cmap = "viridis", num_levels = 5, level_sum = "cumulative", fill = False, color = None, label = True, **kwargs):

    #check_copula_obj(copula_obj)
    u1, u2 = utils.get_u_grid(adj = adj, range_num=range_num) 

    if density is None: 
        density = copula_obj.pdf(u1, u2)

    if ax is None:
        f, ax = plt.subplots()

    if level_sum == "cumulative":
        density = cumulative_level(density)

    if fill:
        func = ax.contourf
    else:
        func = ax.contour
           
    
    args = (u1, u2, density)
    kwargs = {"cmap": cmap, "levels": num_levels} | kwargs

    if color is not None:
        del kwargs["cmap"]
        kwargs["colors"] = color

    CS = func(*args, **kwargs)

    if label:
        ax.clabel(CS, inline=True, fontsize=10)

    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_yticks(ticks); ax.set_xticks(ticks)

    return ax


def cumulative_level(density):
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


def copula_quantile_curves(copula_obj, ax = None, quantiles = [0.95, 0.75, 0.5, 0.25, 0.05], adj = 1e-3, range_num = 100):

    #check_copula_obj(copula_obj)

    u = utils.get_u_range(adj = adj, range_num = range_num)
    u1, q = np.meshgrid(u, quantiles)
    curves = copula_obj.conditional_ppf(u1, q)

    if ax is None:
        f, ax = plt.subplots()

    for q, curve in zip(quantiles, curves):
        ax.plot(u, curve, label = q)


    ax.legend(bbox_to_anchor = (1, 1))

    return ax


def model_quantile_curves(model_obj, ax = None, quantiles = [0.95, 0.75, 0.5, 0.25, 0.05], adj = 1e-4, range_num = 100):

    x = utils.get_x_range(low = model_obj.marginal1.ppf(adj), high = model_obj.marginal1.ppf(1 - adj), range_num = range_num)

    x1, q = np.meshgrid(x, quantiles)
    curves = model_obj.conditional_ppf(x1, q, adj = adj / 10)

    if ax is None:
        f, ax = plt.subplots()

    for q, curve in zip(quantiles, curves):
        ax.plot(x, curve, label = q)

    ax.legend(bbox_to_anchor = (1, 1))
    return ax


def rank_scatter(x1, x2, u1, u2, ax = None, x_alpha = 0.5, u_alpha = 0.5):
    
    if ax is None:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6))

    ax1.scatter(x1, x2, alpha = x_alpha)
    ax2.scatter(u1, u2, alpha = u_alpha)

    return ax

def quantile_dependance(u1, u2, copula = None, copula_label = None, show_indep = False,
                        adj = 5e-2, range_num = 30, ax = None, boot = False, 
                        boot_n = 500, boot_conf_int = 0.95, boot_seed = None, 
                        boot_fill_color = "gray", boot_fill_alpha = 0.35):
    # handle input

    q_range = utils.get_u_range(adj = adj, range_num = range_num)
    q_dep_data = [utils.quantile_dependance(u1, u2, q) for q in q_range]

    if ax is None:
        f, ax = plt.subplots()

    if boot:
        q_dep_boot = utils.bootstrap_quantile_dependance(u1, u2, q_range, n = boot_n, seed = boot_seed)
        
        div_2 = (1 - boot_conf_int) / 2
        lower_int = np.quantile(q_dep_boot, div_2, axis = 0)
        upper_int = np.quantile(q_dep_boot, boot_conf_int + div_2, axis = 0)

        # plotting band
        ax.fill_between(q_range, lower_int, upper_int, color = boot_fill_color, alpha = boot_fill_alpha, label = "{}% Bootstrap Conf Int".format(boot_conf_int * 100))

    ax.scatter(q_range, q_dep_data, s = 65, label = "Observed Data", facecolors = "none", edgecolors = "blue")
    ax.plot(q_range, copula.quantile_dependance(q_range), label = copula_label, color = "red", lw = 2)

    if show_indep:
        ax.plot(q_range, np.where(q_range > 0.5, 1 - q_range, q_range), label = "Indepedance", color = "black", linestyle = "dashed", lw = 1)
    
    
    ax.legend(bbox_to_anchor = (1, 1))

    return ax



    

    



    


    
    



