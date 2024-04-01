import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor


def get_u_range(adj = 1e-4, range_num = 100):
    return np.linspace(0 + adj, 1 - adj, range_num)


def get_x_range(low = -3, high = 3, range_num = 1000):
    return np.linspace(low, high, range_num)


def get_u_grid(adj = 1e-4, range_num = 100):
    u = get_u_range(adj = adj, range_num = range_num)
    return np.meshgrid(u, u)


def clip_u_input(u, adj = 1e-5):
    return np.clip(u, adj, 1 - adj)


def is_number(x):
    return isinstance(x, float) or isinstance(x, int)


def is_arraylike(x):
    # expand to pandas series
    return isinstance(x, np.ndarray)


def format_func(value, width, precision):
    return f"{value:{width}.{precision}f}"


def quantile_dependance(u1, u2, q):
    u1_above_below = np.where(q > 0.5, u1 > q, u1 < q)
    u2_cond_u1 = u2[u1_above_below]
    return np.mean(np.where(q > 0.5, u2_cond_u1 > q, u2_cond_u1 < q))


def rank_transform(x):
    # x is 1d numpy array
    n = len(x)

    # Sort the data and get the indices
    sorted_indices = np.argsort(x)
    
    # Initialize an array to hold the ranks
    ranks = np.empty_like(sorted_indices, dtype=float)
    
    # Assign ranks to the data, handling ties by averaging
    ranks[sorted_indices] = np.arange(1, n + 1)
    
    # Handle ties by averaging ranks
    unique_vals, inv = np.unique(x, return_inverse=True)
    avg_ranks = np.array([ranks[x == val].mean() for val in unique_vals])
    percentile_ranks = avg_ranks[inv]
    
    # Convert ranks to percentiles
    percentiles = percentile_ranks / n
    
    return percentiles


def normal_transform(x, adj = 1e-6):
    u = clip_u_input(rank_transform(x), adj = adj)
    return stats.norm.ppf(u)


def rank_iteration(x1, x2, indices, rank_transform):
    # This function is executed in parallel for each bootstrap sample
    x1_boot = x1[indices]
    x2_boot = x2[indices]

    x1_boot_ranked = rank_transform(x1_boot)
    x2_boot_ranked = rank_transform(x2_boot)

    return x1_boot_ranked, x2_boot_ranked


def bootstrap_ranks(x1, x2, n=500, seed=0):
    rng = np.random.default_rng(seed=seed)
    random_indices = rng.integers(low=0, high=len(x1), size=(n, len(x1)))

    bootstrapped_ranks = np.empty(shape=(n, len(x1), 2))

    with ProcessPoolExecutor() as executor:
        # Schedule the bootstrap computations to run in parallel
        futures = [executor.submit(rank_iteration, x1, x2, indices, rank_transform)
                   for indices in random_indices]

        # Collect the results as they are completed
        for i, future in enumerate(futures):
            x1_boot_ranked, x2_boot_ranked = future.result()
            bootstrapped_ranks[i, :, 0] = x1_boot_ranked
            bootstrapped_ranks[i, :, 1] = x2_boot_ranked

    return bootstrapped_ranks


def bootstrap_quantile_dependance(x1, x2, q_range, n = 500, seed = None):
    BOOT = bootstrap_ranks(x1, x2, n = n, seed = seed)
    
    q_dep = np.empty(shape = (n, len(q_range)))

    for i in range(n):
        for j, q in enumerate(q_range):
            q_dep[i, j] = quantile_dependance(BOOT[i, :, 0], BOOT[i, :, 1], q)

    return q_dep


def build_cdf_interpolations(x_range, cdf_values):

    # cdf and ppf or inverse cdf
    cdf = interp1d(x_range, cdf_values, bounds_error = False, fill_value = (x_range[0], x_range[-1]))
    ppf = interp1d(cdf_values, x_range, bounds_error = False, fill_value = (cdf_values[0], cdf_values[-1]))
    return cdf, ppf


def find_x_bounds(loc, scale, pdf_func, *pdf_params, tol = 5e-4, expansion_factor = 0.5):
    
    step = expansion_factor * scale
    left_bound = -3 * scale + loc; right_bound = 3 * scale + loc
    
    while True:
        
        pdf_right = pdf_func(right_bound, *pdf_params)
        pdf_left = pdf_func(left_bound, *pdf_params)
        
        if pdf_left > tol:
            left_bound -= step
        
        if pdf_right > tol:
            right_bound += step
            
        if pdf_left < tol and pdf_right < tol:
            return left_bound, right_bound
        

def flatten_concatenate(data1, data2):
    return np.stack([data1.flatten(), data2.flatten()], axis = 1)


def sample_kendall_tau(x1, x2):
    return stats.kendalltau(x1, x2).statistic


def sample_spearman_rho(x1, x2):
    return stats.spearmanr(x1, x2).statistic


def monte_carlo_kendall_tau(copula, n = 10_000):
    u1, u2 = copula.simulate(n = n)
    return sample_kendall_tau(u1, u2)


def monte_carlo_spearman_rho(copula, n = 10_000):
    u1, u2 = copula.simulate(n = n)
    return sample_spearman_rho(u1, u2)


def solve_for_ppf(cdf_func, q, a, b, *params, adj = 1e-6):
    # input validation for q

    def F(q):
        f = lambda x: cdf_func(x, *params) - q
        return brentq(f, a = a, b = b)
    
    if is_number(q):
        return F(q)
    
    # else, q is array
    x = [F(q_i) for q_i in q.flatten()]
    return np.array(x).reshape(q.shape)


def solve_for_conditional_ppf(conditional_cdf_func, u1, q, *params, adj = 1e-6):

    # input validation for u1 and q
    # catch f(a) or f(b) error?

    def F(u1, q, *params, adj = 1e-6):
        f = lambda u2: conditional_cdf_func(u1, u2, *params) - q
        return brentq(f, a = adj, b = 1 - adj)
    
    if is_number(u1) and is_number(q):
        return F(u1, q, *params)
    

    # else, u1 and q are arrays
    # assuming same shape
    
    u2 = [F(u1_i, q_i, *params, adj = adj) for u1_i, q_i in zip(u1.flatten(), q.flatten())]
    return np.array(u2).reshape(u1.shape)


def monte_carlo_cvar(marginal_dist, n = 1000, seed = None, alpha = 0.95):
    x = marginal_dist.simulate(n = n, seed = seed)
    thresh = np.quantile(x, 1 - alpha)
    return np.mean(x[x <= thresh])



def monte_carlo_stats(marginal):
    X = marginal.simulate(n = marginal.monte_carlo_n, seed = marginal.monte_carlo_seed)

    skewness = stats.skew(X); kurtosis = stats.kurtosis(X)

    quantile = np.quantile(X, 0.05)
    cvar_filter = X <= quantile
    cvar = np.mean(X[cvar_filter])

    return skewness, kurtosis, cvar




