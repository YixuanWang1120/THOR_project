import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss
from scipy.stats import logistic
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

# Import the THOR model functions
from thor.data_preprocessing import load_and_prepare_data, setup_data_groups
from thor.model import THOR

# Set Seaborn style for plots
sns.set(style="whitegrid")

def load_data(clinical_path, genomic_path):
    """
    Load and prepare data for evaluation.

    Parameters:
    - clinical_path: Path to the clinical data Excel file.
    - genomic_path: Path to the genomic data Excel file.

    Returns:
    - df: Prepared DataFrame.
    - max_clone_label: Maximum clone label.
    - p: Number of predictors per subclone.
    - feature_columns: List of feature column names.
    - feature_columns_tmb: List of TMB feature column names.
    """
    df, max_clone_label, p = load_and_prepare_data(clinical_path, genomic_path)
    feature_columns = [f'feature_{i+1}' for i in range(max_clone_label * p)]
    feature_columns_tmb = [col for col in df.columns if 'feature_' in col and int(col.split('_')[-1]) % 2 != 0]

    # Normalize TMB features
    scaler = MinMaxScaler()
    df[feature_columns_tmb] = scaler.fit_transform(df[feature_columns_tmb])

    return df, max_clone_label, p, feature_columns, feature_columns_tmb

def cross_validation_analysis(df, max_clone_label, p, feature_columns, feature_columns_tmb, n_splits=5):
    """
    Perform cross-validation analysis and compare models.

    Parameters:
    - df: DataFrame containing the data.
    - max_clone_label: Maximum clone label.
    - p: Number of predictors per subclone.
    - feature_columns: List of feature column names.
    - feature_columns_tmb: List of TMB feature column names.
    - n_splits: Number of folds for cross-validation.

    Returns:
    - results: Dictionary containing performance metrics and ROC data.
    """
    X = df[feature_columns].values
    y = df['ORR'].values

    # Initialize dictionaries to store performance metrics
    metrics = {
        'THOR': {'logloss': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'auc': []},
        'THOR_TMB': {'logloss': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'auc': []},
        'LogisticRegression': {'logloss': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'auc': []},
        'LogisticRegression_TMB': {'logloss': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'auc': []},
        'XGBoost': {'logloss': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'auc': []},
        'XGBoost_TMB': {'logloss': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'auc': []},
        'NeuralNetwork': {'logloss': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'auc': []},
        'NeuralNetwork_TMB': {'logloss': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'auc': []}
    }

    # Create StratifiedKFold cross-validator
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")

        # Split data into training and testing sets
        df_train = df.iloc[train_index].reset_index(drop=True)
        df_test = df.iloc[test_index].reset_index(drop=True)
        X_train_pooled = df_train[feature_columns].values
        X_test_pooled = df_test[feature_columns].values
        X_train_pooled_tmb = df_train[feature_columns_tmb].values.sum(axis=1).reshape(-1, 1)
        X_test_pooled_tmb = df_test[feature_columns_tmb].values.sum(axis=1).reshape(-1, 1)
        R_train_pooled = df_train['ORR'].values
        R_test_pooled = df_test['ORR'].values

        # Train the THOR model on the training set
        param_thor_train, success, _ = THOR(df_train, max_clone_label, p,
                                            tau_type="kl_divergence", n_points=5, lambda_=0.5, gamma=0.1)

        # Evaluate the THOR model on the test set
        data_groups_test, _, _, K = setup_data_groups(df_test, feature_columns)
        thor_scores_test_k = []
        for k in range(K):
            X_test_group = df_test[df_test['Study ID'] == k + 1][feature_columns].values
            if X_test_group.size == 0:
                continue  # Skip if the group has no data in the test set
            alpha_k = param_thor_train[:max_clone_label * p, k]
            probabilities = logistic.cdf(X_test_group @ alpha_k)
            thor_scores = probabilities.reshape(-1, 1)
            thor_scores_test_k.append((thor_scores, df_test[df_test['Study ID'] == k + 1]['ORR'].values))

        # Concatenate scores and true labels from all groups
        thor_scores_test = np.concatenate([arr[0].ravel() for arr in thor_scores_test_k])
        R_test_pooled_thor = np.concatenate([arr[1] for arr in thor_scores_test_k])

        # Calculate performance metrics for THOR
        logloss_thor = log_loss(R_test_pooled_thor, thor_scores_test)
        accuracy_thor = accuracy_score(R_test_pooled_thor, (thor_scores_test >= 0.5).astype(int))
        fpr_thor, tpr_thor, _ = roc_curve(R_test_pooled_thor, thor_scores_test)
        auc_thor = auc(fpr_thor, tpr_thor)

        metrics['THOR']['logloss'].append(logloss_thor)
        metrics['THOR']['accuracy'].append(accuracy_thor)
        metrics['THOR']['fpr'].append(fpr_thor)
        metrics['THOR']['tpr'].append(tpr_thor)
        metrics['THOR']['auc'].append(auc_thor)

        # Train the THOR model on TMB features
        df_train_tmb = df_train.drop(columns=[col for col in feature_columns if col not in feature_columns_tmb])
        df_train_tmb['feature_1'] = df_train_tmb[feature_columns_tmb].values.sum(axis=1)
        df_test_tmb = df_test.drop(columns=[col for col in feature_columns if col not in feature_columns_tmb])
        df_test_tmb['feature_1'] = df_test_tmb[feature_columns_tmb].values.sum(axis=1)

        # Train the THOR model on the training set for TMB
        param_thor_train_tmb, success_tmb, _ = THOR(df_train_tmb, 1, 1,
                                                    tau_type="none", n_points=5, lambda_=0.1, gamma=0.1)

        # Evaluate the THOR model on the test set
        thor_scores_test_k_tmb = []
        for k in range(K):
            X_test_group_tmb = df_test_tmb[df_test_tmb['Study ID'] == k + 1]['feature_1'].values.reshape(-1, 1)
            if X_test_group_tmb.size == 0:
                continue
            alpha_tmb_k = param_thor_train_tmb[:1, k]
            probabilities_tmb = logistic.cdf(X_test_group_tmb @ alpha_tmb_k)
            thor_scores_tmb = probabilities_tmb.reshape(-1, 1)
            thor_scores_test_k_tmb.append((thor_scores_tmb, df_test_tmb[df_test_tmb['Study ID'] == k + 1]['ORR'].values))

        # Concatenate scores and true labels from all groups
        thor_scores_test_tmb = np.concatenate([arr[0].ravel() for arr in thor_scores_test_k_tmb])
        R_test_pooled_thor_tmb = np.concatenate([arr[1] for arr in thor_scores_test_k_tmb])

        # Calculate performance metrics for THOR on TMB
        logloss_thor_tmb = log_loss(R_test_pooled_thor_tmb, thor_scores_test_tmb)
        accuracy_thor_tmb = accuracy_score(R_test_pooled_thor_tmb, (thor_scores_test_tmb >= 0.5).astype(int))
        fpr_thor_tmb, tpr_thor_tmb, _ = roc_curve(R_test_pooled_thor_tmb, thor_scores_test_tmb)
        auc_thor_tmb = auc(fpr_thor_tmb, tpr_thor_tmb)

        metrics['THOR_TMB']['logloss'].append(logloss_thor_tmb)
        metrics['THOR_TMB']['accuracy'].append(accuracy_thor_tmb)
        metrics['THOR_TMB']['fpr'].append(fpr_thor_tmb)
        metrics['THOR_TMB']['tpr'].append(tpr_thor_tmb)
        metrics['THOR_TMB']['auc'].append(auc_thor_tmb)

        # Train Logistic Regression model
        lr_model = LogisticRegression(max_iter=500, C=0.5, solver='liblinear')
        lr_model.fit(X_train_pooled, R_train_pooled)
        # Predict probabilities on test set
        lr_probs = lr_model.predict_proba(X_test_pooled)[:, 1]
        # Calculate performance metrics for Logistic Regression
        logloss_lr = log_loss(R_test_pooled, lr_probs)
        accuracy_lr = accuracy_score(R_test_pooled, (lr_probs >= 0.5).astype(int))
        fpr_lr, tpr_lr, _ = roc_curve(R_test_pooled, lr_probs)
        auc_lr = auc(fpr_lr, tpr_lr)

        metrics['LogisticRegression']['logloss'].append(logloss_lr)
        metrics['LogisticRegression']['accuracy'].append(accuracy_lr)
        metrics['LogisticRegression']['fpr'].append(fpr_lr)
        metrics['LogisticRegression']['tpr'].append(tpr_lr)
        metrics['LogisticRegression']['auc'].append(auc_lr)

        # Logistic Regression on TMB
        lr_model_tmb = LogisticRegression(max_iter=500, C=0.5, solver='liblinear')
        lr_model_tmb.fit(X_train_pooled_tmb, R_train_pooled)
        lr_probs_tmb = lr_model_tmb.predict_proba(X_test_pooled_tmb)[:, 1]
        logloss_lr_tmb = log_loss(R_test_pooled, lr_probs_tmb)
        accuracy_lr_tmb = accuracy_score(R_test_pooled, (lr_probs_tmb >= 0.5).astype(int))
        fpr_lr_tmb, tpr_lr_tmb, _ = roc_curve(R_test_pooled, lr_probs_tmb)
        auc_lr_tmb = auc(fpr_lr_tmb, tpr_lr_tmb)

        metrics['LogisticRegression_TMB']['logloss'].append(logloss_lr_tmb)
        metrics['LogisticRegression_TMB']['accuracy'].append(accuracy_lr_tmb)
        metrics['LogisticRegression_TMB']['fpr'].append(fpr_lr_tmb)
        metrics['LogisticRegression_TMB']['tpr'].append(tpr_lr_tmb)
        metrics['LogisticRegression_TMB']['auc'].append(auc_lr_tmb)

        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_pooled, R_train_pooled)
        # Predict probabilities on test set
        xgb_probs = xgb_model.predict_proba(X_test_pooled)[:, 1]
        # Calculate performance metrics for XGBoost
        logloss_xgb = log_loss(R_test_pooled, xgb_probs)
        accuracy_xgb = accuracy_score(R_test_pooled, (xgb_probs >= 0.5).astype(int))
        fpr_xgb, tpr_xgb, _ = roc_curve(R_test_pooled, xgb_probs)
        auc_xgb = auc(fpr_xgb, tpr_xgb)

        metrics['XGBoost']['logloss'].append(logloss_xgb)
        metrics['XGBoost']['accuracy'].append(accuracy_xgb)
        metrics['XGBoost']['fpr'].append(fpr_xgb)
        metrics['XGBoost']['tpr'].append(tpr_xgb)
        metrics['XGBoost']['auc'].append(auc_xgb)

        # XGBoost on TMB
        xgb_model_tmb = xgb.XGBClassifier(
            eval_metric='logloss'
        )
        xgb_model_tmb.fit(X_train_pooled_tmb, R_train_pooled)
        xgb_probs_tmb = xgb_model_tmb.predict_proba(X_test_pooled_tmb)[:, 1]
        logloss_xgb_tmb = log_loss(R_test_pooled, xgb_probs_tmb)
        accuracy_xgb_tmb = accuracy_score(R_test_pooled, (xgb_probs_tmb >= 0.5).astype(int))
        fpr_xgb_tmb, tpr_xgb_tmb, _ = roc_curve(R_test_pooled, xgb_probs_tmb)
        auc_xgb_tmb = auc(fpr_xgb_tmb, tpr_xgb_tmb)

        metrics['XGBoost_TMB']['logloss'].append(logloss_xgb_tmb)
        metrics['XGBoost_TMB']['accuracy'].append(accuracy_xgb_tmb)
        metrics['XGBoost_TMB']['fpr'].append(fpr_xgb_tmb)
        metrics['XGBoost_TMB']['tpr'].append(tpr_xgb_tmb)
        metrics['XGBoost_TMB']['auc'].append(auc_xgb_tmb)

        # Train Neural Network model
        nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
        nn_model.fit(X_train_pooled, R_train_pooled)
        # Predict probabilities on test set
        nn_probs = nn_model.predict_proba(X_test_pooled)[:, 1]
        # Calculate performance metrics for Neural Network
        logloss_nn = log_loss(R_test_pooled, nn_probs)
        accuracy_nn = accuracy_score(R_test_pooled, (nn_probs >= 0.5).astype(int))
        fpr_nn, tpr_nn, _ = roc_curve(R_test_pooled, nn_probs)
        auc_nn = auc(fpr_nn, tpr_nn)

        metrics['NeuralNetwork']['logloss'].append(logloss_nn)
        metrics['NeuralNetwork']['accuracy'].append(accuracy_nn)
        metrics['NeuralNetwork']['fpr'].append(fpr_nn)
        metrics['NeuralNetwork']['tpr'].append(tpr_nn)
        metrics['NeuralNetwork']['auc'].append(auc_nn)

        # Neural Network on TMB
        nn_model_tmb = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
        nn_model_tmb.fit(X_train_pooled_tmb, R_train_pooled)
        nn_probs_tmb = nn_model_tmb.predict_proba(X_test_pooled_tmb)[:, 1]
        logloss_nn_tmb = log_loss(R_test_pooled, nn_probs_tmb)
        accuracy_nn_tmb = accuracy_score(R_test_pooled, (nn_probs_tmb >= 0.5).astype(int))
        fpr_nn_tmb, tpr_nn_tmb, _ = roc_curve(R_test_pooled, nn_probs_tmb)
        auc_nn_tmb = auc(fpr_nn_tmb, tpr_nn_tmb)

        metrics['NeuralNetwork_TMB']['logloss'].append(logloss_nn_tmb)
        metrics['NeuralNetwork_TMB']['accuracy'].append(accuracy_nn_tmb)
        metrics['NeuralNetwork_TMB']['fpr'].append(fpr_nn_tmb)
        metrics['NeuralNetwork_TMB']['tpr'].append(tpr_nn_tmb)
        metrics['NeuralNetwork_TMB']['auc'].append(auc_nn_tmb)

    results = {}

    for model in metrics.keys():
        results[model] = {
            'avg_logloss': np.mean(metrics[model]['logloss']),
            'avg_accuracy': np.mean(metrics[model]['accuracy']),
            'avg_auc': np.mean(metrics[model]['auc']),
            'mean_fpr': np.linspace(0, 1, 100),
            'mean_tpr': np.mean([np.interp(np.linspace(0, 1, 100), fpr, tpr) for fpr, tpr in zip(metrics[model]['fpr'], metrics[model]['tpr'])], axis=0)
        }

    return results

def plot_roc_curves(results):
    """
    Plot ROC curves for all models.

    Parameters:
    - results: Dictionary containing performance metrics and ROC data.
    """
    plt.figure(figsize=(8, 6))
    # Colors for each model
    colors = {
        'THOR': '#DD6E60',
        'LogisticRegression': '#e5c185',
        'XGBoost': '#adc178',
        'NeuralNetwork': '#8CAED1'
    }

    # Plot ROC curves
    for model in ['THOR', 'LogisticRegression', 'XGBoost', 'NeuralNetwork']:
        plt.plot(results[model]['mean_fpr'], results[model]['mean_tpr'], color=colors[model],
                 lw=2, label=f'{model} (AUC = {results[model]["avg_auc"]:.2f})')

    plt.plot([0, 1], [0, 1], color='#ABBBBF', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curves Comparison (Cross-Validation)', fontsize=13)
    plt.legend(loc='lower right', fontsize=13)
    plt.tight_layout()
    plt.savefig('Comparison_ROC_AllModels_CV.png', dpi=300)
    plt.show()

def perform_kaplan_meier_analysis(df, param_thor, max_clone_label, p):
    """
    Perform Kaplan-Meier analysis using the THOR model.

    Parameters:
    - df: DataFrame containing the data.
    - param_thor: Parameters from the THOR model.
    - max_clone_label: Maximum clone label.
    - p: Number of predictors per subclone.
    """
    feature_columns = [f'feature_{i+1}' for i in range(max_clone_label * p)]
    T_pooled = df['Survival'].values
    delta_pooled = df['Status'].values
    data_groups, _, _, K = setup_data_groups(df, feature_columns)

    hazard_k = []
    for k in range(K):
        X, _, _, _ = data_groups[k]
        beta_k = param_thor[max_clone_label * p:2 * max_clone_label * p, k]
        hazard_k.append(X @ beta_k)

    hazard = np.concatenate([arr.ravel() for arr in hazard_k])
    median_hazard = np.percentile(hazard, 50)
    groups = (hazard > median_hazard).astype(int)  # High-risk and low-risk groups
    high_risk = T_pooled[groups == 1]
    low_risk = T_pooled[groups == 0]
    high_risk_status = delta_pooled[groups == 1]
    low_risk_status = delta_pooled[groups == 0]

    logrank = logrank_test(high_risk, low_risk, event_observed_A=high_risk_status, event_observed_B=low_risk_status)
    print(f"Log-rank Test p-value for THOR: {logrank.p_value:.4f}")

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    plt.figure(figsize=(6, 4.5))
    ax = plt.subplot(111)
    kmf_high.fit(high_risk, event_observed=high_risk_status, label='High Risk')
    kmf_high.plot_survival_function(ax=ax)
    kmf_low.fit(low_risk, event_observed=low_risk_status, label='Low Risk')
    kmf_low.plot_survival_function(ax=ax)
    plt.title('Kaplan-Meier Survival Curves for THOR')
    plt.annotate(f'Log-rank test p-value: {logrank.p_value:.4f}', xy=(0.6, 0.7), xycoords='axes fraction', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('Kaplan_Meier_Curves_THOR.png', dpi=300)
    plt.show()

def main():
    # Load data
    clinical_path = r'/Users/wangyixuan/Documents/02 research_study/01 TMB课题/Penalized_Joint_THOR/THOR_project/data/Experimental_Cohort.xlsx'
    genomic_path = r'/Users/wangyixuan/Documents/02 research_study/01 TMB课题/Penalized_Joint_THOR/THOR_project/data/Experimental_Cohort.xlsx'
    df, max_clone_label, p, feature_columns, feature_columns_tmb = load_data(clinical_path, genomic_path,)

    # Cross-validation analysis
    results = cross_validation_analysis(df, max_clone_label, p, feature_columns, feature_columns_tmb, n_splits=5)

    # Print average performance metrics for all models
    for model in ['THOR', 'LogisticRegression', 'XGBoost', 'NeuralNetwork']:
        print(f"{model} Model Average Log-Loss: {results[model]['avg_logloss']:.4f}")
        print(f"{model} Model Average Accuracy: {results[model]['avg_accuracy']:.4f}")
        print(f"{model} Model Average AUC: {results[model]['avg_auc']:.4f}\n")

    # Plot ROC curves
    plot_roc_curves(results)

    # Print average performance metrics for all models
    for model in ['THOR_TMB', 'LogisticRegression_TMB', 'XGBoost_TMB', 'NeuralNetwork_TMB']:
        print(f"{model} Model Average Log-Loss: {results[model]['avg_logloss']:.4f}")
        print(f"{model} Model Average Accuracy: {results[model]['avg_accuracy']:.4f}")
        print(f"{model} Model Average AUC: {results[model]['avg_auc']:.4f}\n")

    # Plot ROC curves
    plot_roc_curves(results)

    # Perform Kaplan-Meier analysis using THOR model
    # First, train THOR on the entire dataset
    param_thor, success, _ = THOR(df, max_clone_label, p, tau_type="distance", n_points=5, lambda_=0.1, gamma=0.01)
    if success:
        perform_kaplan_meier_analysis(df, param_thor, max_clone_label, p)
    else:
        print("THOR optimization failed. Cannot perform Kaplan-Meier analysis.")

if __name__ == "__main__":
    main()
