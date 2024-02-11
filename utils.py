import numpy as np
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor


def get_u_range(adj = 1e-3, range_num = 100):
    return np.linspace(0 + adj, 1 - adj, range_num)


def get_u_grid(adj = 1e-3, range_num = 100):
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
    cdf = interp1d(x_range, cdf, bounds_error = False, fill_value = (x_range[0], x_range[-1]))
    ppf = interp1d(cdf_values, x_range, bounds_error = False, fill_value = (cdf_values[0], cdf_values[-1]))
    return cdf, ppf


def find_x_bounds(loc, scale, pdf_func, *pdf_params, tol = 5e-4, expansion_factor = 0.5):
    
    step = expansion_factor * scale
    left_bound = -3 * scale + loc; right_bound = 3 * scale + loc
    
    while True:
        
        pdf_right = pdf_func(right_bound, *pdf_params)[0]
        pdf_left = pdf_func(left_bound, *pdf_params)[0]
        
        if pdf_left > tol:
            left_bound -= step
        
        if pdf_right > tol:
            right_bound += step
            
        if pdf_left < tol and pdf_right < tol:
            return left_bound, right_bound




