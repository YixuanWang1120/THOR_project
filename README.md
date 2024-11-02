
# THOR Algorithm: a TMB Heterogeneity-Adaptive Optimization Model Predicts Immunotherapy Response Using Clonal Genomic Features in Group-Structured Data

## Overview

The **THOR Algorithm** is a TMB heterogeneity-adaptive optimization model designed to predict immunotherapy response using clonal genomic features in group-structured data. It integrates both genomic and clinical data to perform survival analysis, specifically focusing on predicting Overall Response Rate (ORR) and Time to Event (TTE) outcomes in patients undergoing immunotherapy.

This project provides a comprehensive implementation of the THOR algorithm, along with evaluation and comparison experiments against standard models such as Logistic Regression, XGBoost, and Neural Networks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Example Datasets](#example-datasets)
- [Code Explanation](#code-explanation)
- [Usage Guide](#usage-guide)
  - [Step-by-Step Instructions](#step-by-step-instructions)
  - [Usage Examples](#usage-examples)
  - [Troubleshooting Tips](#troubleshooting-tips)
- [FAQ](#faq)
- [License](#license)

## Features

- **Group-Structured Data Support**: Tailored for datasets with inherent group structures (e.g., different studies or cohorts).
- **TMB Heterogeneity-Adaptive**: Accounts for tumor mutational burden heterogeneity across clones.
- **Fusion Penalty Options**: Supports various fusion penalty types (`none`, `distance`, `kl_divergence`) to control the similarity between group parameters.
- **Cross-Validation and Evaluation**: Includes functions for cross-validation, model evaluation, and comparison with other machine learning models.
- **Visualization Tools**: Generates ROC curves and Kaplan-Meier survival plots for performance assessment.

## Installation

### Prerequisites

- **Python 3.7** or higher
- **pip** package manager
- Recommended: Use a virtual environment (e.g., `venv` or `conda`)

### Dependencies

Install the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
`requirements.txt` Content:** 

```
numpy
pandas
scipy
scikit-learn
xgboost
matplotlib
seaborn
lifelines
```

### Installation Steps 
 
1. **Clone the Repository** 

```bash
git clone https://github.com/your_username/THOR-Algorithm.git
cd THOR-Algorithm
```
 
2. **Set Up a Virtual Environment (Optional but Recommended)** 

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
 
3. **Install Dependencies** 

```bash
pip install -r requirements.txt
```

## Example Datasets 

The project expects two Excel files:
 
1. **Clinical Data** : `clinical_data.xlsx` with a sheet named `Clinical`, containing columns like: 
  - `Patient_ID`
 
  - `Study ID`
 
  - `ORR` (Overall Response Rate)
 
  - `Survival` (Time to Event)
 
  - `Status` (Event Indicator)
 
2. **Genomic Data** : `genomic_data.xlsx` with a sheet named `Genomic`, containing columns like: 
  - `Patient_ID`
 
  - `clone` (Clone Identifier)
 
  - `CCF` (Cancer Cell Fraction)
**Note** : Ensure that your datasets follow the same structure as described above.
## Code Explanation 

The project is organized into several modules, each responsible for different aspects of the analysis.

### 1. Data Preparation 
 
- **prepare_feature_matrices** : Processes genomic data to create feature matrices representing clonal TMB and CCF for each patient.

```python
def prepare_feature_matrices(df):
    # Code to build feature matrices
    return feature_matrices, max_clone_label, p
```
 
- **create_feature_dataframe** : Merges the feature matrices with clinical data to form a unified dataset.

```python
def create_feature_dataframe(feature_matrices, df_clinical):
    # Code to merge dataframes
    return df_clinical.join(features_df, on='Patient_ID')
```
 
- **load_and_prepare_data** : Loads data from Excel files and prepares it for modeling.

```python
def load_and_prepare_data(clinical_path, genomic_path):
    # Code to load and prepare data
    return df.dropna(), max_clone_label, p
```

### 2. Model Setup 
 
- **setup_data_groups** : Organizes data into groups based on the 'Study ID', and computes means and covariances for each group.

```python
def setup_data_groups(df, feature_columns):
    # Code to set up data groups
    return data_groups, mus, sigmas, K
```

### 3. Fusion Penalty Calculations 
 
- **calculate_feature_distances** : Computes Euclidean distances between group means.

```python
def calculate_feature_distances(data_groups):
    # Code to calculate distances
    return distances, max_distance
```
 
- **kl_divergence**  and **symmetric_kl** : Calculate the (symmetric) KL divergence between group distributions.

```python
def kl_divergence(mu1, sigma1, mu2, sigma2):
    # Code to calculate KL divergence
    return kl_value

def symmetric_kl(mu1, sigma1, mu2, sigma2):
    # Code to calculate symmetric KL divergence
    return symmetric_kl_value
```
 
- **calculate_kl_distances** : Computes KL divergences between all pairs of groups.

```python
def calculate_kl_distances(mus, sigmas):
    # Code to compute KL distances
    return kl_distances
```

### 4. THOR Model 
 
- **calculate_taus** : Determines the fusion penalties (taus) based on the chosen method.

```python
def calculate_taus(data_groups, mus, sigmas, tau_type):
    # Code to calculate taus
    return taus
```
 
- **fused_likelihood** : Defines the fused likelihood function to be minimized during optimization.

```python
def fused_likelihood(params, data_groups, max_clone_label, p, weights, roots, lambda_, gamma, tau_type, taus):
    # Code for fused likelihood
    return -likelihood_sum + total_penalty
```
 
- **THOR** : The main function that performs model fitting using the fused likelihood optimization.

```python
def THOR(df, max_clone_label, p, tau_type="kl_divergence", n_points=5, lambda_=0.5, gamma=0.5):
    # Code to train the THOR model
    return param_fusion, result.success, result
```

### 5. Evaluation and Comparison 
 
- **cross_validation_analysis** : Performs cross-validation and compares the THOR model with Logistic Regression, XGBoost, and Neural Network models.

```python
def cross_validation_analysis(df, max_clone_label, p, feature_columns, feature_columns_tmb, n_splits=5):
    # Code for cross-validation and evaluation
    return results
```
 
- **plot_roc_curves** : Generates ROC curves for model performance visualization.

```python
def plot_roc_curves(results):
    # Code to plot ROC curves
    plt.show()
```
 
- **perform_kaplan_meier_analysis** : Performs survival analysis using Kaplan-Meier curves.

```python
def perform_kaplan_meier_analysis(df, param_thor, max_clone_label, p):
    # Code for Kaplan-Meier analysis
    plt.show()
```

## Usage Guide 

### Step-by-Step Instructions 
 
1. **Prepare Your Data**  
  - Ensure that your clinical and genomic data are correctly formatted and saved as `clinical_data.xlsx` and `genomic_data.xlsx`.
 
2. **Modify File Paths**  
  - Update the `clinical_path` and `genomic_path` variables in the code to point to your data files.
 
3. **Run the Main Script**  
  - Create a `main.py` file (or use the code provided in the last section) and run it:

```bash
python main.py
```
 
  - This script will perform data loading, model training, evaluation, and plotting.

### Usage Examples 

#### Loading and Preparing Data 


```python
from data_preparation import load_and_prepare_data

clinical_path = 'data/clinical_data.xlsx'
genomic_path = 'data/genomic_data.xlsx'
df, max_clone_label, p = load_and_prepare_data(clinical_path, genomic_path)
```

#### Training the THOR Model 


```python
from model import THOR

param_fusion, success, result = THOR(
    df,
    max_clone_label,
    p,
    tau_type="kl_divergence",
    n_points=5,
    lambda_=0.5,
    gamma=0.1
)

if success:
    print("Model training successful.")
else:
    print("Model training failed.")
```

#### Evaluating and Comparing Models 


```python
from evaluation import cross_validation_analysis, plot_roc_curves

results = cross_validation_analysis(
    df,
    max_clone_label,
    p,
    feature_columns,
    feature_columns_tmb,
    n_splits=5
)

plot_roc_curves(results)
```

#### Performing Kaplan-Meier Analysis 


```python
from evaluation import perform_kaplan_meier_analysis

perform_kaplan_meier_analysis(df, param_fusion, max_clone_label, p)
```

### Troubleshooting Tips 
 
- **Optimization Failure**  
  - **Issue** : The THOR model fails to converge during optimization.
 
  - **Solutions** : 
    - Adjust the `lambda_` (regularization parameter) and `gamma` (fusion penalty parameter) values.
 
    - Increase the maximum number of iterations in the `minimize` function (e.g., set `options={'maxiter': 20000}`).
 
- **Data Loading Errors**  
  - **Issue** : Errors occur when reading data from Excel files.
 
  - **Solutions** :
    - Ensure that the file paths are correct.

    - Verify that the Excel files have the required sheets and columns.
 
- **Import Errors**  
  - **Issue** : Python cannot find certain modules or functions.
 
  - **Solutions** :
    - Check that all required packages are installed.

    - Verify that your Python environment is correctly set up.
 
- **Plotting Issues**  
  - **Issue** : Plots are not displayed or saved.
 
  - **Solutions** : 
    - Ensure that `matplotlib` and `seaborn` are installed.
 
    - Use `%matplotlib inline` if running in a Jupyter notebook.

## FAQ 

### 1. What is the purpose of the THOR Algorithm? 

The THOR Algorithm is designed to predict immunotherapy response by integrating clonal genomic features with clinical data, specifically accounting for TMB heterogeneity in group-structured datasets.
2. How do I choose the appropriate fusion penalty (`tau_type`)? 
- **`none`** : No fusion penalty; groups are treated independently.
 
- **`distance`** : Fusion penalty based on Euclidean distance between group feature means.
 
- **`kl_divergence`** : Fusion penalty based on KL divergence between group distributions.

Choose based on your data characteristics and whether you expect similarities between groups.
3. What do the `lambda_` and `gamma` parameters control? 
- **`lambda_`** : Regularization strength to prevent overfitting.
 
- **`gamma`** : Fusion penalty weight to control the degree of similarity enforced between group parameters.

Adjust these to balance model fit and generalization.

### 4. Can I use this code with my own dataset? 
Yes, as long as your dataset follows the structure expected by the code (see the [Example Datasets](https://chatgpt.com/c/67249750-7d30-8013-aece-5209b196f92f#example-datasets)  section).
### 5. What should I do if the optimization process is too slow? 
 
- Reduce the number of Gauss-Hermite quadrature points (`n_points`).

- Simplify the model by reducing the number of features or clones.

- Use more powerful computing resources.

### 6. How can I interpret the model coefficients? 
The optimized parameters (`alpha_k` and `beta_k`) represent the effect sizes of the features on ORR and TTE, respectively. They can be used to understand the influence of clonal TMB and CCF on patient outcomes.
### 7. How do I add more models for comparison? 
Extend the `cross_validation_analysis` function by including additional models, following the structure used for the existing models.
## License 
This project is licensed under the MIT License - see the [LICENSE]()  file for details.

---

**Disclaimer** : This project is intended for research purposes only. Ensure compliance with all relevant ethical guidelines and data protection regulations when using clinical and 
