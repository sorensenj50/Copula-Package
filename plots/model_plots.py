import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

import utils
from .plot_utils import _handle_ax, _merge_kw, _get_joint_model_x_range, _cumulative_level



def marginal_pdf(marginal, ax = None, range_num = 250, **kwargs):
    ax = _handle_ax(ax = ax)

    u = utils.get_u_range(range_num = range_num)
    x = marginal.ppf(u)
    pdf_x = marginal.pdf(x)

    ax.plot(x, pdf_x, **kwargs)
    return ax


def marginal_cdf(marginal, ax = None, range_num = 250, **kwargs):
    ax = _handle_ax(ax = ax)

    u = utils.get_u_range(range_num = range_num)
    x = marginal.ppf(u)
    cdf_x = marginal.cdf(x)

    ax.plot(x, cdf_x, **kwargs)
    return ax


def marginal_ppf(marginal, ax = None, range_num = 250, **kwargs):
    ax = _handle_ax(ax = ax)

    u = utils.get_u_range(range_num = range_num)
    x = marginal.ppf(u)

    ax.plot(u, x, **kwargs)
    return ax


def marginal_qq(marginal, ax = None, range_num = 250, **kwargs):
    ax = _handle_ax(ax = ax)

    u = utils.get_u_range(range_num = range_num)

    x_norm = stats.norm.ppf(u)
    x_marginal = marginal.ppf(u)

    ax.plot(x_norm, x_norm, label = "Normal", color = "black", linestyle = "dashed")
    ax.plot(x_norm, x_marginal, **kwargs)
    ax.set_ylabel("Marginal Quantiles")
    ax.set_xlabel("Normal Quantiles")
    ax.legend()
    return ax


def copula_scatter(copula, n = 1000, seed = None, ax = None, **kwargs):
    ax = _handle_ax(ax = ax)

    u1, u2 = copula.simulate(n = n, seed = None)
    ax.scatter(u1, u2, **kwargs)
    return ax


def copula_quantile_dependance(copula, adj = 1e-2, range_num = 250, ax = None, **kwargs):
    ax = _handle_ax(ax = ax)

    q_range = utils.get_u_range(adj = adj, range_num = range_num)
    q_dep = copula.quantile_dependance(q_range)
    indep = np.where(q_range > 0.5, 1 - q_range, q_range)


    ax.plot(q_range, indep, label = "Independent Copula", color = "black", linestyle = "dashed")
    ax.plot(q_range, q_dep, **kwargs)
    ax.legend()
    return ax


def copula_quantile_curves(copula, quantiles = [0.95, 0.75, 0.5, 0.25, 0.05], adj = 1e-3, range_num = 250, ax = None):
    ax = _handle_ax(ax = ax)

    u = utils.get_u_range(adj = adj, range_num = range_num)
    u1, q = np.meshgrid(u, quantiles)

    curves = copula.conditional_ppf(u1, q)

    for q, curve in zip(quantiles, curves):
        ax.plot(u, curve, label = q)

    ax.legend()
    return ax


def copula_3d_surf(copula, to_plot = "pdf", adj = 5e-2, range_num = 250, elev = None, azim = None, cmap = "viridis", ax = None, **kwargs):
    
    ax = _handle_ax(ax = ax, subplot_kw = {"projection": "3d"})

    u1, u2 = utils.get_u_grid(adj = adj, range_num = range_num)

    if to_plot == "pdf":
        surf = copula.pdf(u1, u2)
    elif to_plot == "cdf":
        surf = copula.cdf(u1, u2)
    else:
        raise ValueError('"to_plot" argument must be "pdf" or "cdf"')

    _ = ax.plot_surface(u1, u2, surf, antialiased = False, **merge_kw({"cmap": cmap}, kwargs))
    ax.set_zlim(0, None)
    ax.view_init(elev = elev, azim = azim)
    
    return ax


def copula_contour(copula, adj = 1e-2, range_num = 100, num_levels = 5, cmap = "viridis", ax = None, **kwargs):

    ax = _handle_ax(ax = ax)

    u1, u2 = utils.get_u_grid(adj = adj, range_num=range_num) 
    density = copula.pdf(u1, u2)
    cum_density = utils._cumulative_level(density)
    
    CS = ax.contour(u1, u2, cum_density, **_merge_kw({"cmap": cmap, "levels": num_levels}, kwargs))
    ax.clabel(CS, inline=True, fontsize=10)

    return ax


def joint_scatter(joint_model, n = 1000, seed = None, ax = None, **kwargs):

    ax = _handle_ax(ax = ax)

    x1, x2 = joint_model.simulate(n = n, seed = seed)
    ax.scatter(x1, x2, **kwargs)
    return ax


# abstract this into copula surf
def joint_3d_surf(joint_model, to_plot = "pdf", adj = 1e-3, range_num = 250, elev = None, azim = None, cmap = "viridis", ax = None, **kwargs):
    ax = _handle_ax(ax = ax, subplot_kw = {"projection": "3d"})

    X1, X2 = _get_joint_model_x_range(joint_model, adj = adj, range_num = range_num)

    if to_plot == "pdf":
        surf = joint_model.pdf(X1, X2)
    elif to_plot == "cdf":
        surf = joint_model.cdf(X1, X2)
    else:
        raise ValueError('"to_plot" argument must be "pdf" or "cdf"')
    
    _ = ax.plot_surface(X1, X2, surf, antialiased = False, **_merge_kw({"cmap": cmap}, kwargs))
    ax.set_zlim(0, None)
    ax.view_init(elev = elev, azim = azim)
    return ax
    

def joint_contour(joint_model, adj = 1e-3, range_num = 250, cmap = "viridis", num_levels = 5, ax = None, **kwargs):
    ax = _handle_ax(ax = ax)

    X1, X2 = _get_joint_model_x_range(joint_model, adj = adj, range_num = range_num)

    density = joint_model.pdf(X1, X2)
    cum_density = _cumulative_level(density)

    CS = ax.contour(X1, X2, cum_density, **_merge_kw({"cmap": cmap, "levels": num_levels}, kwargs))
    ax.clabel(CS, inline=True, fontsize=10)
    return ax











