# -*- coding: utf-8 -*-
# Import custom modules
from thor.data_preprocessing import load_and_prepare_data, setup_data_groups
from thor.model import THOR

# Example usage:
# Load data
clinical_path = r'/Users/wangyixuan/Documents/02 research_study/01 TMB课题/Penalized_Joint_THOR/THOR_project/data/Experimental_Cohort.xlsx'
genomic_path = r'/Users/wangyixuan/Documents/02 research_study/01 TMB课题/Penalized_Joint_THOR/THOR_project/data/Experimental_Cohort.xlsx'
df, max_clone_label, p = load_and_prepare_data(clinical_path, genomic_path,)
K = df['Study ID'].nunique()
# tau_type: "none", "distance", or "kl_divergence"
param_fusion, success, result = THOR(df, max_clone_label, p,
                             tau_type="kl_divergence", n_points=5, lambda_=0.5, gamma=0.1)
if success:
    print("Optimization succeeded.")
    print("Optimized parameters for each group of clonality features:")
    for k in range(K):
        feature_num = max_clone_label * p
        alpha_k = result.x[k * feature_num:(k + 1) * feature_num]
        beta_k = result.x[(K + k) * feature_num:(K + k + 1) * feature_num]
        print(f"Group {k + 1}:")
        for i in range(max_clone_label):
            feature_idx = i * p
            print(
                f" Clone {i + 1}: feature 1 (cTMB) coefficient for ORR = {alpha_k[feature_idx]:.4f}, "
                f"feature 2 (CCF) coefficient for ORR = {alpha_k[feature_idx + 1]:.4f}, "
                f"feature 1 (cTMB) coefficient for TTE = {beta_k[feature_idx]:.4f}, "
                f"feature 2 (CCF) coefficient for TTE = {beta_k[feature_idx + 1]:.4f}")
else:
    print(f"Optimization failed: {result.message}")
