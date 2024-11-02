# -*- coding: utf-8 -*-
import numpy as np

# Function to calculate Euclidean distances between data groups
def calculate_feature_distances(data_groups):
    distances, max_distance = {}, 0
    for k, group_k in enumerate(data_groups):
        mean_k = np.mean(group_k[0], axis=0)
        for j, group_j in enumerate(data_groups[k+1:], k+1):
            distance = np.linalg.norm(mean_k - np.mean(group_j[0], axis=0))
            distances[(k+1, j+1)] = distance
            max_distance = max(max_distance, distance)
    return distances, max_distance

# Functions for KL divergence calculations
def kl_divergence(mu1, sigma1, mu2, sigma2):
    epsilon = 1e-5
    regularized_sigma1 = sigma1 + epsilon * np.eye(sigma1.shape[0])
    regularized_sigma2 = sigma2 + epsilon * np.eye(sigma2.shape[0])
    inv_sigma2 = np.linalg.inv(regularized_sigma2)
    tr_term = np.trace(inv_sigma2 @ regularized_sigma1)
    diff_mu = mu2 - mu1
    quad_term = diff_mu.T @ inv_sigma2 @ diff_mu
    log_det_term = np.log(np.linalg.det(regularized_sigma2) / np.linalg.det(regularized_sigma1))
    return 0.5 * (tr_term + quad_term - len(mu1) + log_det_term)

def symmetric_kl(mu1, sigma1, mu2, sigma2):
    return 0.5 * (kl_divergence(mu1, sigma1, mu2, sigma2) + kl_divergence(mu2, sigma2, mu1, sigma1))

# Calculate Taus based on distance or KL divergence
def calculate_kl_distances(mus, sigmas):
    kl_distances = {}
    for i in range(len(mus)):
        for j in range(i + 1, len(mus)):
            kl_dist = symmetric_kl(mus[i], sigmas[i], mus[j], sigmas[j])
            kl_distances[(i+1, j+1)] = kl_distances[(j+1, i+1)] = kl_dist
    return kl_distances
