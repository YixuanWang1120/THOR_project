# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.stats import logistic, norm
from thor.utils import calculate_kl_distances, calculate_feature_distances
from thor.data_preprocessing import setup_data_groups

def calculate_taus(data_groups, mus, sigmas, tau_type):
    if tau_type == "none":
        taus = {(i + 1, j + 1): 1 for i in range(len(data_groups)) for j in range(i, len(data_groups))}
    elif tau_type == "distance":
        distances, max_distance = calculate_feature_distances(data_groups)
        taus = {key: 1 - value / max_distance for key, value in distances.items()}
    elif tau_type == "kl_divergence":
        kl_distances = calculate_kl_distances(mus, sigmas)
        max_kl = max(kl_distances.values())
        taus = {key: 1 - value / max_kl for key, value in kl_distances.items()}
    return taus

def fused_likelihood(params, data_groups, max_clone_label, p, weights, roots, lambda_, gamma, tau_type, taus):
    feature_num = max_clone_label * p
    K = len(data_groups)
    likelihood_sum = 0
    reg_sum = 0
    group_diff_sum = 0

    # Iterate over each group
    for k in range(K):
        subgroup_data = data_groups[k]
        X, R, T, delta = subgroup_data
        alpha_k = params[k * feature_num:(k + 1) * feature_num]
        beta_k = params[(K + k) * feature_num:(K + k + 1) * feature_num]
        sigma_k = 0.5

        integral_approx = 0
        for r, w in zip(roots, weights):
            omega = sigma_k * np.sqrt(2) * r
            logit_p = X @ alpha_k + omega
            prob = logistic.cdf(logit_p)
            hazard_ratios = np.exp(X @ beta_k + omega)
            likelihoods = hazard_ratios ** delta * np.exp(-T * hazard_ratios)
            log_fR = np.log(prob ** R * (1 - prob) ** (1 - R) + 1e-8)
            log_fT = np.log(likelihoods + 1e-8)
            log_re = norm.logpdf(omega, 0, scale=sigma_k)
            integral_approx += w * (log_fR + log_fT + log_re)

        likelihood_sum += np.sum(integral_approx)
        reg_sum += lambda_ * (np.sum(alpha_k ** 2) + np.sum(beta_k ** 2))

        # Apply fusion penalty
        for j in range(k + 1, K):
            alpha_j = params[j * feature_num:(j + 1) * feature_num]
            beta_j = params[(K + j) * feature_num:(K + j + 1) * feature_num]
            if tau_type == "distance" or tau_type == "kl_divergence":
                tau_value = taus.get((k, j), 1)  # Default tau to 1 if not specified
            else:
                tau_value = 1  # uniform fusion penalty
            group_diff_sum += tau_value * (np.sum((alpha_k - alpha_j) ** 2) + np.sum((beta_k - beta_j) ** 2))

    total_penalty = reg_sum + gamma * group_diff_sum
    return -likelihood_sum + total_penalty


def THOR(df, max_clone_label, p, tau_type="kl_divergence", n_points=5, lambda_=0.5, gamma=0.5):
   # Feature columns for TMB
    feature_columns_tmb = [col for col in df.columns if 'feature_' in col and int(col.split('_')[-1]) % 2 != 0]

    # Normalize TMB features
    scaler = MinMaxScaler()
    df[feature_columns_tmb] = scaler.fit_transform(df[feature_columns_tmb])

    # Set feature columns
    feature_columns = [f'feature_{i+1}' for i in range(max_clone_label * p)]
    data_groups, mus, sigmas, K = setup_data_groups(df, feature_columns)

    # Gauss-Hermite Quadrature
    roots, weights = hermgauss(n_points)

    # Calculate Taus
    taus = calculate_taus(data_groups, mus, sigmas, tau_type)

    # Initial parameters and optimization
    initial_guess = np.random.normal(size=(2 * K * max_clone_label * p)).astype(np.float32)
    param_fusion = np.zeros((2 * max_clone_label * p, K))

    # Perform optimization
    result = minimize(
        fused_likelihood,
        initial_guess,
        args=(data_groups, max_clone_label, p, weights, roots, lambda_, gamma, tau_type, taus),
        method='L-BFGS-B',
        options={'maxiter': 10000, 'maxcor': 10}
    )

    if result.success:
        result_array = result.x.reshape((2 * K, max_clone_label * p))
        param_fusion[:max_clone_label * p, :] = result_array[:K, :].T
        param_fusion[max_clone_label * p:, :] = result_array[K:, :].T

    return param_fusion, result.success, result
