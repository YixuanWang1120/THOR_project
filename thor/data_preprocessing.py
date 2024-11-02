# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# Function to prepare feature matrices from genomic data
def prepare_feature_matrices(df):
    feature_matrices = {}
    max_clone_label = int(df['clone'].max())
    p = 2  # Number of predictors per subclone
    patient_ids = df['Patient_ID'].unique()

    for patient_id in patient_ids:
        patient_data = df[df['Patient_ID'] == patient_id]
        summary = patient_data.groupby('clone').agg(
            clone_tmb=('clone', 'size'),
            CCF_mean=('CCF', 'mean')
        ).reset_index()
        matrix = np.zeros((max_clone_label, p))
        for _, row in summary.iterrows():
            clone_index = int(row['clone']) - 1
            matrix[clone_index, :] = [row['clone_tmb'], row['CCF_mean']]
        feature_matrices[patient_id] = matrix.ravel()

    return feature_matrices, max_clone_label, p


# Function to merge feature matrices with clinical data
def create_feature_dataframe(feature_matrices, df_clinical):
    features_df = pd.DataFrame.from_dict(feature_matrices, orient='index')
    features_df.columns = [f'feature_{i + 1}' for i in range(features_df.shape[1])]
    return df_clinical.join(features_df, on='Patient_ID')


# Load and prepare data
def load_and_prepare_data(clinical_path, genomic_path):
    df_clinical = pd.read_excel(clinical_path, sheet_name='Clinical')
    df_genomic = pd.read_excel(genomic_path, sheet_name='Genomic')
    feature_matrices, max_clone_label, p = prepare_feature_matrices(df_genomic)
    df = create_feature_dataframe(feature_matrices, df_clinical)
    return df.dropna(), max_clone_label, p


# Setup data for model fitting
def setup_data_groups(df, feature_columns):
    K = df['Study ID'].max()
    data_groups, mus, sigmas = [], [], []

    for k in range(1, K + 1):
        subgroup_data = df[df['Study ID'] == k]
        X = subgroup_data[feature_columns].values
        mus.append(np.mean(X, axis=0))
        sigmas.append(np.cov(X, rowvar=False))
        R, T, delta = subgroup_data['ORR'].values, subgroup_data['Survival'].values, subgroup_data['Status'].values
        data_groups.append((X, R, T, delta))

    return data_groups, mus, sigmas, K

