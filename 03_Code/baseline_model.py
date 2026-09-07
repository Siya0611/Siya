"""
Baseline Phishing URL Detection Model
PhD Research: AI-Enabled Social Engineering Detection
Scholar: Siya0611 | Institution: Rashtriya Raksha University

Description:
    This script implements a baseline Random Forest classifier for phishing URL detection.
    It includes comprehensive feature extraction, cross-validation, performance metrics,
    and SHAP-based feature importance analysis.

Dependencies:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
    - shap
    - pickle
"""

import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import shap

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def has_ip_address(url):
    """
    Check if URL contains an IP address instead of a domain name.
    
    Args:
        url (str): URL string to analyze
    
    Returns:
        int: 1 if IP address detected, 0 otherwise
    """
    # IPv4 pattern
    ipv4_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    if re.search(ipv4_pattern, url):
        return 1
    return 0


def count_special_chars(url):
    """
    Count the number of special characters in URL (excluding http://, https://, etc.).
    
    Args:
        url (str): URL string to analyze
    
    Returns:
        int: Count of special characters
    """
    special_chars = r'[@_~?#&=;%+]'
    return len(re.findall(special_chars, url))


def get_url_depth(url):
    """
    Calculate URL depth as the number of slashes in the path.
    
    Args:
        url (str): URL string to analyze
    
    Returns:
        int: Number of slashes (depth)
    """
    parsed = urlparse(url)
    depth = parsed.path.count('/')
    return depth


def extract_features(url):
    """
    Extract all features from a single URL.
    
    Args:
        url (str): URL string to extract features from
    
    Returns:
        dict: Dictionary containing all extracted features
    """
    features = {}
    
    # Feature 1: URL Length
    features['url_length'] = len(url)
    
    # Feature 2: Number of dots
    features['num_dots'] = url.count('.')
    
    # Feature 3: Number of hyphens
    features['num_hyphens'] = url.count('-')
    
    # Feature 4: HTTPS presence (1 if HTTPS, 0 otherwise)
    features['has_https'] = 1 if url.startswith('https') else 0
    
    # Feature 5: Number of special characters
    features['num_special_chars'] = count_special_chars(url)
    
    # Feature 6: IP address detection
    features['has_ip_address'] = has_ip_address(url)
    
    # Feature 7: URL depth
    features['url_depth'] = get_url_depth(url)
    
    return features


def extract_features_from_dataset(df):
    """
    Extract features from all URLs in a dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'url' column
    
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    print("[INFO] Extracting features from URLs...")
    features_list = []
    
    for idx, url in enumerate(df['url']):
        if idx % 1000 == 0:
            print(f"  Processed {idx} URLs...")
        
        try:
            features = extract_features(url)
            features_list.append(features)
        except Exception as e:
            print(f"  [WARNING] Error processing URL at index {idx}: {e}")
            # Add default features for failed URLs
            features_list.append({
                'url_length': 0, 'num_dots': 0, 'num_hyphens': 0,
                'has_https': 0, 'num_special_chars': 0,
                'has_ip_address': 0, 'url_depth': 0
            })
    
    features_df = pd.DataFrame(features_list)
    print(f"[INFO] Successfully extracted features for {len(features_df)} URLs")
    return features_df


# ============================================================================
# SECTION 2: MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate_model(X, y, model_name='phishing_model.pkl', confusion_matrix_path='confusion_matrix.png'):
    """
    Train Random Forest classifier with 5-fold cross-validation and evaluate performance.
    
    Args:
        X (pd.DataFrame or np.array): Feature matrix
        y (pd.Series or np.array): Target labels (0: legitimate, 1: phishing)
        model_name (str): Path to save trained model
        confusion_matrix_path (str): Path to save confusion matrix plot
    
    Returns:
        dict: Evaluation metrics and trained model
    """
    print("\n[INFO] Initializing Random Forest Classifier...")
    
    # Initialize Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=20,          # Maximum tree depth
        min_samples_split=5,   # Minimum samples to split
        min_samples_leaf=2,    # Minimum samples in leaf
        random_state=42,       # For reproducibility
        n_jobs=-1,            # Use all CPU cores
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Setup 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    print("[INFO] Performing 5-fold cross-validation...")
    
    # Perform cross-validation
    cv_results = cross_validate(
        rf_model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    # Calculate False Positive Rate (FPR)
    fpr_scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        # FPR = False Positives / (False Positives + True Negatives)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_scores.append(fpr)
    
    # Compile evaluation metrics
    evaluation_results = {
        'accuracy': cv_results['test_accuracy'],
        'precision': cv_results['test_precision'],
        'recall': cv_results['test_recall'],
        'f1': cv_results['test_f1'],
        'fpr': np.array(fpr_scores),
        'roc_auc': cv_results['test_roc_auc']
    }
    
    # Train final model on full dataset
    print("[INFO] Training final model on complete dataset...")
    rf_model.fit(X, y)
    
    # Save trained model
    with open(model_name, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"[SUCCESS] Model saved as '{model_name}'")
    
    return evaluation_results, rf_model, X, y


# ============================================================================
# SECTION 3: PERFORMANCE METRICS REPORTING
# ============================================================================

def report_metrics(evaluation_results):
    """
    Print comprehensive performance metrics with mean and standard deviation.
    
    Args:
        evaluation_results (dict): Dictionary containing cross-validation scores
    """
    print("\n" + "="*70)
    print("CROSS-VALIDATION PERFORMANCE METRICS (5-Fold)")
    print("="*70)
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'roc_auc']
    
    for metric in metrics_names:
        scores = evaluation_results[metric]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\n{metric.upper():15s}:")
        print(f"  Scores:     {[f'{s:.4f}' for s in scores]}")
        print(f"  Mean:       {mean_score:.4f}")
        print(f"  Std Dev:    {std_score:.4f}")
        print(f"  Min:        {np.min(scores):.4f}")
        print(f"  Max:        {np.max(scores):.4f}")
    
    print("\n" + "="*70)


# ============================================================================
# SECTION 4: VISUALIZATION AND MODEL ARTIFACTS
# ============================================================================

def plot_confusion_matrix(rf_model, X, y, save_path='confusion_matrix.png'):
    """
    Generate and plot confusion matrix for the trained model.
    
    Args:
        rf_model: Trained Random Forest model
        X (pd.DataFrame or np.array): Feature matrix
        y (pd.Series or np.array): Target labels
        save_path (str): Path to save confusion matrix plot
    """
    print("\n[INFO] Generating confusion matrix...")
    
    # Make predictions on full dataset
    y_pred = rf_model.predict(X)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Phishing'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Phishing URL Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Confusion matrix saved as '{save_path}'")


def plot_shap_importance(rf_model, X, save_path='shap_plot.png'):
    """
    Generate SHAP feature importance plot for model interpretability.
    
    Args:
        rf_model: Trained Random Forest model
        X (pd.DataFrame or np.array): Feature matrix
        save_path (str): Path to save SHAP plot
    """
    print("\n[INFO] Computing SHAP values for feature importance...")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(rf_model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # For binary classification, take SHAP values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Plot SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance - Phishing Detection Model', fontsize=14, fontweight='bold')
        plt.xlabel('Mean |SHAP value| (Average impact on model output)', fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SUCCESS] SHAP feature importance plot saved as '{save_path}'")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate SHAP plot: {e}")


# ============================================================================
# SECTION 5: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function: orchestrates data loading, feature extraction,
    model training, evaluation, and visualization.
    """
    print("\n" + "="*70)
    print("PHISHING URL DETECTION - BASELINE MODEL")
    print("PhD Research: AI-Enabled Social Engineering Detection")
    print("="*70)
    
    # Step 1: Load dataset
    print("\n[STEP 1] Loading dataset...")
    try:
        df = pd.read_csv('../../04_Data/dataset.csv')
        print(f"[SUCCESS] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("[ERROR] dataset.csv not found in 04_Data/ directory")
        print("Please ensure the CSV file exists with 'url' and 'label' columns")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return
    
    # Validate dataset structure
    if 'url' not in df.columns or 'label' not in df.columns:
        print("[ERROR] Dataset must contain 'url' and 'label' columns")
        return
    
    # Step 2: Extract features
    print("\n[STEP 2] Extracting URL features...")
    features_df = extract_features_from_dataset(df)
    
    # Verify feature extraction
    print(f"[INFO] Extracted features shape: {features_df.shape}")
    print(f"[INFO] Feature columns: {list(features_df.columns)}")
    print(f"[INFO] Sample feature values:\n{features_df.head()}")
    
    # Step 3: Prepare data for modeling
    print("\n[STEP 3] Preparing data for modeling...")
    X = features_df
    y = df['label']
    
    # Check class distribution
    print(f"[INFO] Class distribution:")
    print(f"  Legitimate (0): {(y == 0).sum()} samples")
    print(f"  Phishing (1): {(y == 1).sum()} samples")
    print(f"  Class ratio: {(y == 1).sum() / (y == 0).sum():.4f}")
    
    # Step 4: Train model and perform cross-validation
    print("\n[STEP 4] Training Random Forest classifier with cross-validation...")
    eval_results, trained_model, X_full, y_full = train_and_evaluate_model(
        X, y,
        model_name='phishing_model.pkl',
        confusion_matrix_path='confusion_matrix.png'
    )
    
    # Step 5: Report metrics
    print("\n[STEP 5] Reporting performance metrics...")
    report_metrics(eval_results)
    
    # Step 6: Generate confusion matrix visualization
    print("\n[STEP 6] Generating confusion matrix...")
    plot_confusion_matrix(trained_model, X_full, y_full, save_path='confusion_matrix.png')
    
    # Step 7: Generate SHAP feature importance
    print("\n[STEP 7] Generating SHAP feature importance analysis...")
    plot_shap_importance(trained_model, X_full, save_path='shap_plot.png')
    
    # Step 8: Summary
    print("\n" + "="*70)
    print("MODEL TRAINING AND EVALUATION COMPLETE")
    print("="*70)
    print("\nGenerated artifacts:")
    print("  ✓ phishing_model.pkl - Trained Random Forest model")
    print("  ✓ confusion_matrix.png - Confusion matrix visualization")
    print("  ✓ shap_plot.png - SHAP feature importance plot")
    print("\nNext steps:")
    print("  1. Review metrics and model performance")
    print("  2. Analyze SHAP plots for feature interpretation")
    print("  3. Consider hyperparameter tuning if needed")
    print("  4. Compare with advanced models (XGBoost, Neural Networks, etc.)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
