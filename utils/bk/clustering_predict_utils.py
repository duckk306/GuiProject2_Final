# %% [markdown]
# # Clustering Analysis with Automated Model Selection
# 
# This notebook demonstrates an automated approach to clustering analysis with:
# - Automatic selection of optimal number of clusters
# - Comparison between K-Means and GMM
# - Parallel processing for faster execution
# - Comprehensive model evaluation

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from joblib import Parallel, delayed
import joblib
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Import clustering utilities
import sys
sys.path.append('../utils')
from clustering_utils import (
    perform_clustering_analysis,
    find_optimal_clusters,
    parallel_clustering_analysis,
    cluster_summary_report
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %%
def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
    # Load data
    df = pd.read_excel(filepath)
    
    # Basic preprocessing (adjust as needed)
    df = df.dropna()  # Remove rows with missing values

    # Comprehensive column cleaning
    def clean_column_name(col):
        if not isinstance(col, str):
            col = str(col)
        # Remove all types of whitespace and replace with underscore
        col = col.strip()
        col = re.sub(r'\s+', '_', col)
        return col
    
    df.columns = [clean_column_name(col) for col in df.columns]

    return df

# %%
def prepare_features(df, num_cols, cat_cols, n_components=0.95):
    """
    Prepare features for clustering.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    num_cols : list
        List of numerical column names
    cat_cols : list
        List of categorical column names
    n_components : float or int
        Number of components for PCA (if float, percentage of variance to explain)
        
    Returns:
    --------
    tuple: (X, X_scaled, feature_names, preprocessor)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Create transformers
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    # Apply transformations
    X_processed = preprocessor.fit_transform(df)
    
    # Get feature names
    num_features = num_cols
    if len(cat_cols) > 0:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(cat_cols)
        feature_names = np.concatenate([num_features, cat_features])
    else:
        feature_names = num_features
    
    # Apply PCA if n_components is specified
    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_processed)
        print(f"Reduced dimensions from {X_processed.shape[1]} to {X_pca.shape[1]} with PCA")
        return X_pca, X_processed, feature_names, preprocessor, pca
    
    return X_processed, X_processed, feature_names, preprocessor, None

# %%
def train_best_model(X, method='auto', k_range=range(2, 11), n_runs=5):
    """
    Train and select the best clustering model.
    
    Parameters:
    -----------
    X : array-like
        Input features
    method : str
        Clustering method ('kmeans', 'gmm', or 'auto' for both)
    k_range : range
        Range of k values to try
    n_runs : int
        Number of runs for each k
        
    Returns:
    --------
    dict: Best model and metrics
    """
    if method == 'auto':
        # Try both methods and select the best one
        kmeans_results = parallel_clustering_analysis(
            X, method='kmeans', k_range=k_range, n_runs=n_runs, verbose=False
        )
        gmm_results = parallel_clustering_analysis(
            X, method='gmm', k_range=k_range, n_runs=n_runs, verbose=False
        )
        
        # Get best model from each method
        best_kmeans = max(kmeans_results.values(), 
                         key=lambda x: x['silhouette_score'])
        best_gmm = max(gmm_results.values(), 
                      key=lambda x: x['silhouette_score'])
        
        # Select best overall model
        if best_kmeans['silhouette_score'] >= best_gmm['silhouette_score']:
            best_model = best_kmeans
            best_model['method'] = 'kmeans'
            all_results = kmeans_results
        else:
            best_model = best_gmm
            best_model['method'] = 'gmm'
            all_results = gmm_results
            
        return {
            'best_model': best_model,
            'all_results': all_results,
            'method': best_model['method']
        }
    else:
        # Use specified method
        results = parallel_clustering_analysis(
            X, method=method, k_range=k_range, n_runs=n_runs, verbose=False
        )
        best_model = max(results.values(), 
                        key=lambda x: x['silhouette_score'])
        return {
            'best_model': best_model,
            'all_results': results,
            'method': method
        }

# %%
def plot_cluster_metrics(results, title):
    """Plot clustering metrics across different k values."""
    k_values = sorted(results.keys())
    silhouette_scores = [results[k]['silhouette_score'] for k in k_values]
    inertias = [results[k]['inertia'] for k in k_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Silhouette score plot
    ax1.plot(k_values, silhouette_scores, 'bo-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title(f'Silhouette Score vs k\n{title}')
    
    # Elbow plot (inertia)
    ax2.plot(k_values, inertias, 'ro-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Inertia')
    ax2.set_title(f'Elbow Method\n{title}')
    
    plt.tight_layout()
    plt.show()

# %%
def save_clustering_results(model_info, X, df, output_dir='output'):
    """Save clustering results and models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best model
    model = model_info['best_model']['model']
    method = model_info['method']
    k = model_info['best_model']['n_clusters']
    
    model_path = os.path.join(output_dir, f'best_{method}_k{k}.joblib')
    joblib.dump(model, model_path)
    
    # Save cluster assignments
    labels = model_info['best_model']['labels']
    df['cluster'] = labels
    
    # Save cluster statistics
    cluster_stats = df.groupby('cluster').describe().T
    stats_path = os.path.join(output_dir, 'cluster_statistics.csv')
    cluster_stats.to_csv(stats_path)
    
    # Save model info
    info = {
        'method': method,
        'n_clusters': k,
        'silhouette_score': model_info['best_model']['silhouette_score'],
        'model_path': model_path,
        'stats_path': stats_path
    }
    
    info_path = os.path.join(output_dir, 'model_info.json')
    import json
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    return info

# %%
def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('data/your_data.csv')
    
    # Define numerical and categorical columns (adjust as needed)
    num_cols = ['Giá', 'Khoảng_giá_min', 'Khoảng_giá_max', 'Năm_đăng_ký', 'Số_Km_đã_đi', 'age']
    cat_cols = ['Thương_hiệu', 'Dòng_xe', 'Loại_xe', 'Dung_tích_xe', 'Xuất_xứ', 'Phân_khúc_giá']
    
    # Prepare features
    print("Preparing features...")
    X, X_original, feature_names, preprocessor, pca = prepare_features(
        df, num_cols, cat_cols, n_components=0.95
    )
    
    # Train and select best model
    print("\nTraining clustering models...")
    model_info = train_best_model(
        X, method='auto', k_range=range(2, 11), n_runs=5
    )
    
    # Print best model info
    best = model_info['best_model']
    print(f"\nBest model: {model_info['method'].upper()} with k={best['n_clusters']}")
    print(f"Silhouette Score: {best['silhouette_score']:.3f}")
    
    # Plot metrics
    plot_cluster_metrics(model_info['all_results'], 
                        f"Method: {model_info['method'].upper()}")
    
    # Save results
    print("\nSaving results...")
    info = save_clustering_results(model_info, X, df)
    print(f"Results saved to {os.path.dirname(info['model_path'])}")
    
    return model_info, df

# %%
#if __name__ == "__main__":
#    model_info, df = main()