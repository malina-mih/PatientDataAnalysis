import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gower
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from kmedoids import KMedoids
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from scipy.cluster.hierarchy import linkage, fcluster

# --- Configuration ---
OPTIMAL_K_RANGE = range(2, 11)

PREDEFINED_HEALTH_INDICATORS = [
    'Body Mass Index (kg/m²)', 'Systolic Blood Pressure (1st reading) mm Hg',
    'Diastolic Blood Pressure (1st reading) mm Hg', 'LDL-cholesterol (mg/dL)',
    'Direct HDL-Cholesterol (mg/dL)', 'Total cholesterol (mg/dL)',
    'Triglyceride (mg/dL)', 'Ever have Diabetes', 'Ever have Hypertension',
    'Ever have Heart Attack', 'Number of Medicines',  'Health Condition',
    'Waist Circumference (cm)',
    'Glycohemoglobin (HbA1c) (%)'
]

HEALTH_INDICATOR_BEST_CRITERIA = {
    'Body Mass Index (kg/m²)': 'lower', 'Systolic Blood Pressure (1st reading) mm Hg': 'lower',
    'Diastolic Blood Pressure (1st reading) mm Hg': 'lower', 'LDL-cholesterol (mg/dL)': 'lower',
    'Direct HDL-Cholesterol (mg/dL)': 'higher', 'Total cholesterol (mg/dL)': 'lower',
    'Triglyceride (mg/dL)': 'lower', 'Ever have Diabetes': 'lower', 'Ever have Hypertension': 'lower',
    'Ever have Heart Attack': 'lower', 'Number of Medicines': 'lower',
    'Health Condition': 'higher',
    'Waist Circumference (cm)': 'lower',
    'Glycohemoglobin (HbA1c) (%)': 'lower'
}

ORDINAL_HEALTH_MAP = {"Excellent": 4, "Very good": 3, "Good": 2, "Fair": 1, "Poor": 0, "Don't know": np.nan, "Refused": np.nan}
YES_NO_MAP = {"Yes": 1, "No": 0, "Don't know": np.nan, "Refused": np.nan}


# Helper Functions
def calculate_clustering_metrics(data_input, labels, method_name, k_val, distance_matrix=False):
    """
    Calculates various clustering evaluation metrics.
    Excludes noise points (labels < 0) from metric calculations.
    """
    metrics = {
        'Method': method_name,
        'Num_Clusters': k_val,
        'Silhouette': np.nan,
        'Calinski-Harabasz': np.nan,
        'Davies-Bouldin': np.nan,
        'Cluster_Sizes_Distribution': "N/A"
    }

    labels_array = np.asarray(labels, dtype=int)
    valid_label_mask = (labels_array >= 0)
    effective_labels = labels_array[valid_label_mask]
    num_effective_clusters = len(np.unique(effective_labels))

    # Determine cluster size distribution
    if num_effective_clusters > 0:
        cluster_counts = pd.Series(effective_labels).value_counts().sort_index()
        metrics['Cluster_Sizes_Distribution'] = ", ".join([f"C{lbl}:{cnt}" for lbl, cnt in cluster_counts.items()])
    elif labels_array.shape[0] > 0 and np.all(labels_array < 0):
        metrics['Cluster_Sizes_Distribution'] = "All noise"
    else:
        metrics['Cluster_Sizes_Distribution'] = "No valid data for clustering"

    # Calculate metrics
    if num_effective_clusters > 1 and effective_labels.shape[0] > num_effective_clusters:
        data_for_metrics = None
        if data_input is not None and data_input.shape[0] == len(labels_array):
            data_for_metrics = data_input[valid_label_mask]

        if data_for_metrics is not None and data_for_metrics.shape[0] > 1:
            try:
                metric_param = 'precomputed' if distance_matrix else 'euclidean'
                metrics['Silhouette'] = silhouette_score(data_for_metrics, effective_labels, metric=metric_param)
            except Exception as e:
                st.warning(f"Silhouette Error for {method_name} (k={num_effective_clusters}): {e}")

            if not distance_matrix:  
                try:
                    metrics['Calinski-Harabasz'] = calinski_harabasz_score(data_for_metrics, effective_labels)
                except Exception as e:
                    st.warning(f"Calinski-Harabasz Error for {method_name} (k={num_effective_clusters}): {e}")
                try:
                    metrics['Davies-Bouldin'] = davies_bouldin_score(data_for_metrics, effective_labels)
                except Exception as e:
                    st.warning(f"Davies-Bouldin Error for {method_name} (k={num_effective_clusters}): {e}")
    return metrics
    
    
def calculate_bootstrap_stability(
    data_for_fitting, original_labels, model_name, k_clusters,
    preproc_type, hac_linkage=None, gower_dist_matrix=None,
    n_bootstraps=300, progress_bar=None
):
    """
    Calculates clustering stability using bootstrap resampling and Adjusted Rand Index (ARI).
    """
    st.info(f"Running Bootstrap Stability for {model_name} (k={k_clusters}, {preproc_type})...")
    ari_scores = []
    n_samples = len(original_labels)

    for i in range(n_bootstraps):
        bootstrap_indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples, random_state=i)

        if len(np.unique(bootstrap_indices)) < k_clusters:
            st.warning(f"Bootstrap sample {i+1} has too few unique samples ({len(np.unique(bootstrap_indices))}) for k={k_clusters}. Skipping.")
            if progress_bar:
                progress_bar.progress((i + 1) / n_bootstraps)
            continue

        try:
            bootstrap_data_fit = None
            bootstrap_gower_matrix = None
            if preproc_type == "PCA":
                bootstrap_data_fit = data_for_fitting[bootstrap_indices]
            elif preproc_type == "Gower":
                if gower_dist_matrix is None:
                    raise ValueError("Gower distance matrix is required for bootstrap.")
                bootstrap_gower_matrix = gower_dist_matrix[np.ix_(bootstrap_indices, bootstrap_indices)]

            fitted_model, bootstrap_labels = fit_clustering_model(
                data_fit=bootstrap_data_fit,
                full_method_name=model_name,
                k=k_clusters,
                preproc_type=preproc_type,
                linkage_type=hac_linkage,
                gower_dist_matrix=bootstrap_gower_matrix
            )

            if bootstrap_labels is not None:
                ari = adjusted_rand_score(original_labels[bootstrap_indices], bootstrap_labels)
                ari_scores.append(ari)

        except Exception as e:
            st.error(f"Bootstrap sample {i+1} for {model_name} failed: {e}")

        if progress_bar:
            progress_bar.progress((i + 1) / n_bootstraps, text=f"Completed bootstrap sample {i+1}/{n_bootstraps}")

    if progress_bar:
        progress_bar.empty()

    if not ari_scores:
        return np.nan, np.nan, 0

    return np.mean(ari_scores), np.std(ari_scores), len(ari_scores)
    
@st.cache_data
def evaluate_optimal_k(data_input_for_fitting, algorithm_name, preprocessing_type,
                      gower_for_metrics=None, 
                       linkage_type='ward'):
    """
    Evaluates optimal k for various clustering methods, returning key metrics.
    """
    k_values = list(OPTIMAL_K_RANGE)
    results = []

    if data_input_for_fitting is None or data_input_for_fitting.shape[0] < OPTIMAL_K_RANGE.start + 1:
        st.warning(f"Insufficient samples to evaluate optimal k for {algorithm_name}.")
        nan_results = ([np.nan] * len(k_values) for _ in range(5))
        return k_values, *nan_results

    for k in k_values:
        result_for_k = {'sil': np.nan, 'ch': np.nan, 'db': np.nan, 'sizes': 'N/A'}
        if data_input_for_fitting.shape[0] < k + 1:
            results.append(result_for_k)
            continue

        try:
            
            gower_matrix_arg = data_input_for_fitting if preprocessing_type == "Gower" else None
            
            model, cluster_labels = fit_clustering_model(
                data_fit=data_input_for_fitting,
                full_method_name=algorithm_name,
                k=k,
                preproc_type=preprocessing_type,
                linkage_type=linkage_type,
                gower_dist_matrix=gower_matrix_arg
            )

            if cluster_labels is not None:
                
                is_dist_matrix_for_metrics = (preprocessing_type == "Gower")
                
                metrics_for_k = calculate_clustering_metrics(
                    data_input=data_input_for_fitting,
                    labels=cluster_labels,
                    method_name=f"{algorithm_name} k={k}", 
                    k_val=k,
                    distance_matrix=is_dist_matrix_for_metrics
                )
                result_for_k.update({
                    'sil': metrics_for_k['Silhouette'],
                    'ch': metrics_for_k['Calinski-Harabasz'],
                    'db': metrics_for_k['Davies-Bouldin'],
                    'sizes': metrics_for_k['Cluster_Sizes_Distribution']
                })
        except Exception as error:
            st.warning(f"Error during optimal k for {algorithm_name} ({preprocessing_type}), k={k}: {error}")

        results.append(result_for_k)

    sils = [r['sil'] for r in results]
    chs = [r['ch'] for r in results]
    dbs = [r['db'] for r in results]
    sizes = [r['sizes'] for r in results]

    return k_values, sils, chs, dbs, sizes
@st.cache_data
def fit_clustering_model(data_fit, full_method_name, k, preproc_type, linkage_type=None, gower_dist_matrix=None):
    """
    Fits a specified clustering model.
    Returns the fitted model object (if applicable) and the cluster labels.
    """
    
    model, labels = None, None

    try:
        if "K-Means" in full_method_name and preproc_type == "PCA":
            model = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(data_fit)
            labels = model.labels_
        elif "K-Medoids" in full_method_name and preproc_type == "Gower":
            if gower_dist_matrix is None:
                raise ValueError("Gower distance matrix is required for K-Medoids.")
            model = KMedoids(n_clusters=k, method='pam', random_state=42).fit(gower_dist_matrix)
            labels = model.labels_
        elif "GMM" in full_method_name and preproc_type == "PCA":
            model = GaussianMixture(n_components=k, random_state=42, covariance_type='full').fit(data_fit)
            labels = model.predict(data_fit)
        elif "HAC" in full_method_name or "Agglo" in full_method_name:
            if preproc_type == "PCA":
                if linkage_type == 'ward' and data_fit.ndim > 1 and data_fit.shape[1] == 1:
                    raise ValueError("Ward linkage is not appropriate for 1D PCA data.")
                model = AgglomerativeClustering(n_clusters=k, linkage=linkage_type).fit(data_fit)
                labels = model.labels_
            elif preproc_type == "Gower":
                if gower_dist_matrix is None:
                    raise ValueError("Gower distance matrix is required for HAC Gower.")
                if gower_dist_matrix.shape[0] < 2: 
                    labels = np.zeros(gower_dist_matrix.shape[0], dtype=int) if gower_dist_matrix.shape[0] > 0 else np.array([])
                else:
                    condensed_dist = gower_dist_matrix[np.triu_indices(gower_dist_matrix.shape[0], k=1)]
                    linked_matrix = linkage(condensed_dist, method=linkage_type)
                    labels = fcluster(linked_matrix, t=k, criterion='maxclust') - 1
                model = None 
        else:
            raise ValueError(f"Unknown method combination: {full_method_name}, {preproc_type}")
    except Exception as error:
        st.error(f"Error fitting {full_method_name} (k={k}): {error}")
        return None, None
    return model, labels
    
def plot_optimal_k_charts(k_values, sils, chs, dbs, method_display_name, ax=None):
    """
    Plots optimal k charts.
    """
    plot_data = {
        "Silhouette": sils,
        "Calinski-Harabasz": chs,
        "Davies-Bouldin": dbs
    }
    valid_metrics = {name: data for name, data in plot_data.items() if any(pd.notna(val) for val in data)}

    if not valid_metrics:
        if ax is None: 
            st.write(f"No valid internal metrics to plot for {method_display_name}.")
        return

    if ax is not None:
        # gower methods
        metric_name = list(valid_metrics.keys())[0]
        data_points = list(valid_metrics.values())[0]
        
        ax.plot(k_values, data_points, marker='o', color='g', label="Silhouette")
        ax.set_title(f"Silhouette Score\n{method_display_name}")
        ax.set_ylabel("Silhouette Score")
        ax.set_xlabel("Number of clusters (k)")
        ax.grid(True)
        return 

    # PCA methods
    num_plots = len(valid_metrics)
    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), squeeze=False)
    axs = axs.flatten()

    plot_info = {
        "Silhouette": {"title": "Silhouette Score", "color": 'g'},
        "Calinski-Harabasz": {"title": "Calinski-Harabasz Index", "color": 'm'},
        "Davies-Bouldin": {"title": "Davies-Bouldin Index", "color": 'r'}
    }

    for idx, (metric_name, data_points) in enumerate(valid_metrics.items()):
        info = plot_info[metric_name]
        ax = axs[idx]
        ax.plot(k_values, data_points, marker='o', label=metric_name, color=info["color"])
        ax.set_title(f"{info['title']}\n{method_display_name}")
        ax.set_ylabel(info['title'])
        ax.set_xlabel("Number of clusters (k)")
        ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
def plot_silhouette_per_cluster(data_for_plot, labels, k_value, method_name, preproc_type, gower_matrix=None):
    """
    Generates a boxplot of silhouette scores for each cluster.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]

    if len(np.unique(valid_labels)) < 2 or len(valid_labels) < (len(np.unique(valid_labels)) + 1):
        ax.text(0.5, 0.5, "Not enough clusters for silhouette plot.", ha='center', va='center', transform=ax.transAxes)
        st.pyplot(fig)
        plt.close(fig)
        return

    sil_data, sil_metric = None, 'euclidean'

    if preproc_type == "Gower":
        if gower_matrix is None or gower_matrix.shape[0] != len(labels):
            ax.text(0.5, 0.5, "Gower matrix error.", ha='center', va='center', transform=ax.transAxes)
            st.pyplot(fig)
            plt.close(fig)
            return
        row_indices_sil = np.where(valid_mask)[0]
        if len(row_indices_sil) > 0:
            sil_data = gower_matrix[np.ix_(row_indices_sil, row_indices_sil)]
            sil_metric = 'precomputed'
        else:
            ax.text(0.5, 0.5, "No valid samples for Gower silhouette.", ha='center', va='center', transform=ax.transAxes)
            st.pyplot(fig)
            plt.close(fig)
            return
    else: # PCA preprocessing
        if data_for_plot is None or data_for_plot.shape[0] != len(labels):
            ax.text(0.5, 0.5, "PCA data missing for silhouette calculation.", ha='center', va='center', transform=ax.transAxes)
            st.pyplot(fig)
            plt.close(fig)
            return
        sil_data = data_for_plot[valid_mask]

    if sil_data.shape[0] != len(valid_labels) or sil_data.shape[0] < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
        st.pyplot(fig)
        plt.close(fig)
        return

    try:
        sample_sil_values = silhouette_samples(sil_data, valid_labels, metric=sil_metric)
        overall_avg_score = silhouette_score(sil_data, valid_labels, metric=sil_metric)

        df_sil_plot = pd.DataFrame({'Cluster': valid_labels.astype(str), 'Silhouette Score': sample_sil_values})
        df_sil_plot.sort_values('Cluster', inplace=True)

        sns.boxplot(x='Cluster', y='Silhouette Score', data=df_sil_plot, ax=ax, palette='tab10',
                    order=sorted(df_sil_plot['Cluster'].unique()))
        ax.axhline(overall_avg_score, color="red", linestyle="--", label=f"Average: {overall_avg_score:.2f}")
        ax.set_title(f"Silhouette Scores per Cluster for {method_name} (k={k_value})")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Silhouette Score")
        ax.legend()
    except Exception as e:
        ax.text(0.5, 0.5, f"Error generating silhouette plot: {e}", ha='center', va='center', transform=ax.transAxes, wrap=True)
    st.pyplot(fig)
    plt.close(fig)
    
    
def plot_pca_clusters(pca_data, labels, k_value, method_name, preproc_type):
    """
    Plots the first two PCA components 
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if pca_data is None or pca_data.shape[1] < 2:
        ax.text(0.5, 0.5, "PCA data not available.", ha='center', va='center', transform=ax.transAxes)
        st.pyplot(fig)
        plt.close(fig)
        return
    if len(labels) != pca_data.shape[0]:
        ax.text(0.5, 0.5, "PCA data error", ha='center', va='center', transform=ax.transAxes)
        st.pyplot(fig)
        plt.close(fig)
        return

    df_pca_plot = pd.DataFrame(pca_data[:, :2], columns=['PCA1', 'PCA2'])
    df_pca_plot['Cluster'] = labels.astype(str)

    cluster_order = sorted(df_pca_plot['Cluster'].unique(), key=lambda x: int(x) if x.isdigit() and x != '-1' else -999)

    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca_plot, palette='tab10',
                    legend='full', ax=ax, s=50, alpha=0.7, hue_order=cluster_order)
    ax.set_title(f"PCA Plot (2D Projection) for {method_name} (k={k_value})") # Updated title
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
    
    
    
@st.cache_data
def get_tsne_projection(data_tsne_input, perplexity=30, n_iter=300):
    """Computes t-SNE projection."""

    if data_tsne_input is None or data_tsne_input.shape[0] < 5:
        if data_tsne_input is not None:
             print(f"Skipping t-SNE: n_samples ({data_tsne_input.shape[0]}) is too small.")
        return None

    actual_perplexity = min(perplexity, data_tsne_input.shape[0] - 1)

    tsne_model = TSNE(n_components=2, random_state=42, perplexity=actual_perplexity, n_iter=n_iter,
                      init='pca', learning_rate='auto')
    try:
        return tsne_model.fit_transform(data_tsne_input)
    except Exception as e:
        print(f"Error in t-SNE projection: {e}")
        return None
        
        
def plot_tsne_clusters(tsne_data, labels, k_value, method_name, preproc_type, tsne_source_desc=""):
    """
    Plots t-SNE 
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    error_message = None

    if tsne_data is None:
        error_message = f"t-SNE data not available."
    elif len(labels) != tsne_data.shape[0]:
        error_message = "t-SNE data and label length mismatch."

    if error_message:
        ax.text(0.5, 0.5, error_message, ha='center', va='center', transform=ax.transAxes)
    else:
        df_tsne_plot = pd.DataFrame(tsne_data, columns=['TSNE1', 'TSNE2'])
        df_tsne_plot['Cluster'] = labels.astype(str)
        cluster_order = sorted(df_tsne_plot['Cluster'].unique(), key=lambda x: int(x) if x.isdigit() and x != '-1' else -999)

        sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', data=df_tsne_plot, palette='tab10',
                        legend='full', ax=ax, s=50, alpha=0.7, hue_order=cluster_order)
        ax.set_title(f"t-SNE Plot for {method_name} (k={k_value})")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    
def plot_pca_diagnostics(preprocessed_data):
    """
    Generates Cumulative Explained Variance and Scree Plots for PCA.
    """
    if preprocessed_data is None or preprocessed_data.shape[1] < 2:
        st.warning("Cannot generate PCA plots.")
        return

    pca_full = PCA(n_components=None, random_state=42)
    pca_full.fit(preprocessed_data)

    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Cumulative Explained Variance Plot
    ax1.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Cumulative Explained Variance")
    ax1.set_title("Cumulative Explained Variance by Components")
    ax1.grid(True)
    ax1.axhline(y=0.95, color='r', linestyle=':', label='95% Variance')
    ax1.axhline(y=0.90, color='g', linestyle=':', label='90% Variance')
    ax1.axhline(y=0.85, color='b', linestyle=':', label='85% Variance')
    ax1.legend(loc='best')

    # Scree Plot
    ax2.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center',
            label='Individual explained variance')
    ax2.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='.', color='r',
             label='Cumulative explained variance')
    ax2.set_xlabel("Principal Component Index")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_title("Scree Plot")
    ax2.legend(loc='best')
    ax2.set_xticks(range(1, len(explained_variance_ratio) + 1))
    plt.setp(ax2.get_xticklabels(), rotation=90, ha="right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    
@st.cache_data
def prepare_features_for_clustering(raw_df, adult_age_threshold=18):
    """
    Prepares data for clustering 
    Adult participants only
    """
    st.info(f"Running Feature Preparation for Clustering (age >= {adult_age_threshold})...")
    df = raw_df.copy()

    df = df[df['Age'] >= adult_age_threshold]

    # IQR Filter
    if 'Energy (kcal)' in df.columns:
        df['Energy (kcal)'] = pd.to_numeric(df['Energy (kcal)'], errors='coerce')
        df.dropna(subset=['Energy (kcal)'], inplace=True)
        
        if not df.empty:
            Q1 = df['Energy (kcal)'].quantile(0.25)
            Q3 = df['Energy (kcal)'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(0, Q1 - 1.5 * IQR)
            upper_bound = Q3 + 1.5 * IQR
            df = df[df['Energy (kcal)'].between(lower_bound, upper_bound)]
            st.info(f"Filtered 'Energy (kcal)' using IQR: kept values between {lower_bound:.2f} and {upper_bound:.2f}")

    if df.empty:
        st.error(f"No adults (Age >= {adult_age_threshold}) found after filtering.")
        return None

    # Feature Engineering 
    df["Gender_Cat"] = df.get("RIAGENDR", "Unknown").astype(str)

    if 'Protein (gm)' in df.columns and 'Energy (kcal)' in df.columns:
        df['Protein (gm)'] = pd.to_numeric(df['Protein (gm)'], errors='coerce')
        df['% Energy from Protein'] = np.where(df['Energy (kcal)'] > 0, (df['Protein (gm)'].fillna(0) * 4) / df['Energy (kcal)'] * 100, 0)
        df['% Energy from Protein'] = df['% Energy from Protein'].replace([np.inf, -np.inf], np.nan)

    fat_cols = ['Total fat (gm)', 'Total saturated fatty acids (gm)']
    for col in fat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Total fat (gm)' in df.columns and 'Total saturated fatty acids (gm)' in df.columns:
        df['Saturated Fat as % of Total Fat'] = np.where(df['Total fat (gm)'] > 0, (df['Total saturated fatty acids (gm)'].fillna(0) / df['Total fat (gm)']) * 100, 0)
    
    convenience_cols = ['Fast Food Meals', 'Ready-to-eat Meals', 'Frozen Meals']
    existing_convenience_cols = [col for col in convenience_cols if col in df.columns]
    if existing_convenience_cols:
        df['Convenience_Food_Sum'] = df[existing_convenience_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    else:
        df['Convenience_Food_Sum'] = 0

    if 'Alcohol (gm)' in df.columns and df['Alcohol (gm)'].nunique() > 1:
        numeric_series = pd.to_numeric(df['Alcohol (gm)'], errors='coerce').dropna()
        if not numeric_series.empty and len(numeric_series.unique()) >= 3:
            try:
                df['Alcohol_Cat'] = pd.qcut(numeric_series, q=3, labels=["Low", "Medium", "High"], duplicates='drop').reindex(df.index).astype(str).fillna("Low")
            except ValueError:
                df['Alcohol_Cat'] = "Medium"
        else:
            df['Alcohol_Cat'] = "Low"
    else:
        df['Alcohol_Cat'] = "Low"

    df["Is_Smoker_Cat"] = df.get("Smoking Status Now", pd.Series()).map({"Every day smoker": "Smoker_Yes", "Some days smoker": "Smoker_Yes", "Not at all smoker": "Smoker_No"}).fillna("Smoker_No")
    df["PIR_Cat"] = df.get("Income_to_Poverty_Category", "Unknown").astype(str)

    work_activity_col = "Vigorous Work Activity (Yes/No)"
    rec_activity_col = "Vigorous Recreational Activity (Yes/No)"
    df[work_activity_col] = df.get(work_activity_col, 'No').fillna('No')
    df[rec_activity_col] = df.get(rec_activity_col, 'No').fillna('No')
    conditions = [
        (df[work_activity_col] == 'Yes') & (df[rec_activity_col] == 'Yes'),
        (df[work_activity_col] == 'Yes') & (df[rec_activity_col] == 'No'),
        (df[work_activity_col] == 'No') & (df[rec_activity_col] == 'Yes'),
        (df[work_activity_col] == 'No') & (df[rec_activity_col] == 'No')
    ]
    choices = ['Work & Rec Activity', 'Work Activity Only', 'Rec Activity Only', 'No Vigorous Activity']
    df['Activity_Pattern'] = np.select(conditions, choices, default='No Vigorous Activity')
    
    pa_minute_cols = ["Mins Vigorous Work Activity", "Mins Moderate Work Activity", "Mins Walk/Bike Transport", "Mins Vigorous Rec Activity", "Mins Moderate Rec Activity"]
    existing_pa_cols = [col for col in pa_minute_cols if col in df.columns]
    if existing_pa_cols:
        df["Total Activity Minutes_Num"] = df[existing_pa_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    else:
        df["Total Activity Minutes_Num"] = 0

    
    # Define features lists
    numerical_features = [
        'Age', 'Energy (kcal)', '% Energy from Protein', 'Saturated Fat as % of Total Fat',
        'Sodium (mg)', '% Energy from Fiber', 'Total sugars (gm)', 'Convenience_Food_Sum',
        'Total Activity Minutes_Num', 'Hours Sleep Weekdays', 'Dietary fiber (gm)'
    ]
    numerical_features = [f for f in numerical_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    cat_features_pca = ['Gender_Cat', 'Is_Smoker_Cat', 'Activity_Pattern']
    cat_features_pca = [f for f in cat_features_pca if f in df.columns]
    cat_features_gower = ['Gender_Cat', 'Is_Smoker_Cat', 'PIR_Cat', 'Alcohol_Cat', 'Activity_Pattern']
    cat_features_gower = [f for f in cat_features_gower if f in df.columns]
    feature_order_for_profiling = [
        'Age', 'Gender_Cat', 'Energy (kcal)', '% Energy from Protein', 'Total sugars (gm)', 
        '% Energy from Fiber', 'Sodium (mg)', 'Saturated Fat as % of Total Fat', 'Convenience_Food_Sum', 
        'Alcohol_Cat', 'Is_Smoker_Cat', 'Activity_Pattern', 'Dietary fiber (gm)', 'Total Activity Minutes_Num', 
        'Hours Sleep Weekdays', 'PIR_Cat'
    ]
    all_gower_features = list(set(numerical_features + cat_features_gower))
    gower_feat_names = [f for f in feature_order_for_profiling if f in all_gower_features]

    # Data for Gower
    df_gower_temp = df[all_gower_features + ['SEQN']].copy()
    if numerical_features:
        df_gower_temp[numerical_features] = SimpleImputer(strategy='median').fit_transform(df_gower_temp[numerical_features])
    if cat_features_gower:
        df_gower_temp[cat_features_gower] = SimpleImputer(strategy='most_frequent').fit_transform(df_gower_temp[cat_features_gower])
    df_imputed_gower = df_gower_temp[all_gower_features].copy()
    gower_cat_indices = [col_name in cat_features_gower for col_name in df_imputed_gower.columns]

    # Data for PCA
    df_pca_input = df[numerical_features + cat_features_pca + ['SEQN']].copy()
    pca_num_pipeline = Pipeline([('imputer_num', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    pca_cat_pipeline = Pipeline([('imputer_cat', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))])
    
    pca_transformers = []
    if numerical_features: pca_transformers.append(('num', pca_num_pipeline, numerical_features))
    if cat_features_pca: pca_transformers.append(('cat', pca_cat_pipeline, cat_features_pca))

    if not pca_transformers:
        st.error("No features defined for PCA pipeline.")
        return None

    pca_preprocessor = ColumnTransformer(transformers=pca_transformers, remainder='drop', verbose_feature_names_out=True)
    
    original_indices = df_pca_input.index
    seqns = df_pca_input['SEQN'].copy()
    
    df_gower_temp_aligned = df_gower_temp.set_index('SEQN').reindex(seqns.values).reset_index()
    if not df_gower_temp_aligned.empty:
        df_imputed_gower = df_gower_temp_aligned[all_gower_features].copy()
        df_imputed_gower.index = original_indices

    # PCA Transformation
    pca_components, pca_model, n_pca_components = None, None, 0
    df_processed_pca = pd.DataFrame()
    if not df_pca_input[numerical_features + cat_features_pca].empty:
        try:
            processed_vals_pca = pca_preprocessor.fit_transform(df_pca_input[numerical_features + cat_features_pca])
            processed_pca_feat_names = pca_preprocessor.get_feature_names_out()
            df_processed_pca = pd.DataFrame(processed_vals_pca, columns=processed_pca_feat_names, index=original_indices)

            if not df_processed_pca.empty and df_processed_pca.shape[1] > 1:
                pca_model = PCA(n_components=0.95, random_state=42)
                pca_components = pca_model.fit_transform(df_processed_pca.values)
                n_pca_components = pca_components.shape[1]
            else: # Handle case with 1 or 0 features
                pca_components = df_processed_pca.values
                n_pca_components = pca_components.shape[1]
        except Exception as e:
            st.error(f"PCA preprocessing error: {e}")
            pca_components = np.array([])
    
    # Final DataFrame 
    df_profiling_aligned = df_gower_temp.set_index('SEQN').reindex(seqns.values).reset_index()
    df_profiling = pd.DataFrame()
    if not df_profiling_aligned.empty:
        df_profiling_aligned.index = original_indices
        df_profiling = df_profiling_aligned[gower_feat_names]

    # Prepare raw health indicator data
    raw_df_health_subset = df.set_index('SEQN').reindex(seqns.values).reset_index()
    if 'Diet Health' in raw_df_health_subset.columns: raw_df_health_subset['Diet Health'] = raw_df_health_subset['Diet Health'].map(ORDINAL_HEALTH_MAP)
    if 'Health Condition' in raw_df_health_subset.columns: raw_df_health_subset['Health Condition'] = raw_df_health_subset['Health Condition'].map(ORDINAL_HEALTH_MAP)
    for col in ['Ever have Diabetes', 'Ever have Hypertension', 'Ever have Heart Attack']:
        if col in raw_df_health_subset.columns: raw_df_health_subset[col] = raw_df_health_subset[col].map(YES_NO_MAP)
    if not raw_df_health_subset.empty: raw_df_health_subset.index = original_indices

    return {
        "X_pca": pca_components,
        "X_processed_for_tsne": df_processed_pca.values if not df_processed_pca.empty else None,
        "pca_model": pca_model,
        "n_pca_components": n_pca_components,
        "df_for_gower_matrix": df_imputed_gower,
        "cat_features_gower_indices": gower_cat_indices,
        "df_for_profiling": df_profiling,
        "profiling_feature_names": gower_feat_names,
        "numerical_features_selected": numerical_features,
        "categorical_features_pca_selected": cat_features_pca,
        "categorical_features_gower_selected": cat_features_gower,
        "original_indices": original_indices,
        "seqns": seqns,
        "fitted_preprocessor_pca": pca_preprocessor,
        "raw_df_subset_for_health_indicators": raw_df_health_subset
    }

    
@st.cache_data
def generate_profile_and_health_analysis(
    final_labels, profiling_df, health_data_merged,
    predefined_health_indicators_list, model_identifier_str, final_k_value):
    """
    Generates profiles and health indicator analysis 
    """
    analysis_results = {
        "input_feature_profile_summary": pd.DataFrame(),
        "health_analysis_details": [],
        "consolidated_health_summary_pivot": pd.DataFrame(),
        "cluster_sizes": pd.Series(dtype='int64')
    }
    
    # Calculates the mean of numeric features and the mode of categorical features to create a summary profile.

    if final_labels is not None and not profiling_df.empty and len(final_labels) == len(profiling_df):
        profile_df_with_labels = profiling_df.copy()
        profile_df_with_labels['Cluster'] = final_labels
        profile_df_no_noise = profile_df_with_labels[profile_df_with_labels['Cluster'] >= 0].copy()

        if not profile_df_no_noise.empty:
            numeric_cols = profile_df_no_noise.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = profile_df_no_noise.select_dtypes(exclude=np.number).columns.tolist()

            agg_dict = {}
            if 'Cluster' in numeric_cols:
                numeric_cols.remove('Cluster') 
            
            for col in numeric_cols:
                agg_dict[col] = 'mean'
            for col in categorical_cols:
                agg_dict[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            if agg_dict:
                profile_summary = profile_df_no_noise.groupby('Cluster').agg(agg_dict)
                analysis_results["input_feature_profile_summary"] = profile_summary
            
            analysis_results["cluster_sizes"] = profile_df_no_noise['Cluster'].value_counts().sort_index().rename("Count")

    # Health indicators analysis 
    if not health_data_merged.empty:
        health_pivot_data = []
        expected_cluster_ids = list(range(final_k_value)) 
        actual_clusters = sorted(health_data_merged['Cluster'].unique()) 

        for indicator in predefined_health_indicators_list:
            if indicator not in health_data_merged.columns:
                continue

            indicator_details = {"name": indicator, "stat_table": pd.DataFrame(), "p_value_text": "N/A"}
            stat_table_rows = []
            is_binary = indicator in ['Ever have Diabetes', 'Ever have Hypertension', 'Ever have Heart Attack']

            # Statistical tests 
            if len(actual_clusters) > 1:
                if is_binary:
                    health_data_merged[indicator] = pd.to_numeric(health_data_merged[indicator], errors='coerce')
                    contingency = pd.crosstab(health_data_merged['Cluster'], health_data_merged[indicator].fillna(-1))
                    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2 and contingency.sum().sum() > 0:
                        try:
                            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                            indicator_details["p_value_text"] = f"Chi-squared Test: Chi2 Stat = {chi2:.2f}, p-value = {p_val:.10f}"
                        except ValueError:
                            indicator_details["p_value_text"] = "Chi-squared test error."
                    else:
                        indicator_details["p_value_text"] = "Not enough data for Chi-squared test."
                else:
                    grouped_data = [health_data_merged[health_data_merged['Cluster'] == c_id][indicator].dropna()
                                    for c_id in actual_clusters]
                    valid_groups = [g for g in grouped_data if len(g) > 0]
                    if len(valid_groups) > 1:
                        try:
                            h_stat, p_val = stats.kruskal(*valid_groups)
                            indicator_details["p_value_text"] = f"Kruskal-Wallis H-statistic = {h_stat:.2f}, p-value = {p_val:.10f}"
                        except ValueError:
                            indicator_details["p_value_text"] = "Kruskal-Wallis test error."
                    else:
                        indicator_details["p_value_text"] = "Not enough groups with data for Kruskal-Wallis test."

            # Calculate statistics for each cluster
            for cluster_id in expected_cluster_ids:
                if cluster_id in actual_clusters:
                    cluster_data = health_data_merged[health_data_merged['Cluster'] == cluster_id][indicator].dropna()
                    n = len(cluster_data)
                    if is_binary:
                        prop = cluster_data.mean() if n > 0 else 0
                        ci_low, ci_upp = proportion_confint(count=cluster_data.sum(), nobs=n, method='wilson') if n > 0 else (np.nan, np.nan)
                        stat_table_rows.append({'Cluster': cluster_id, 'Proportion (Yes)': prop, 'N': n, '95% CI Lower': ci_low, '95% CI Upper': ci_upp})
                        health_pivot_data.append({'Cluster': cluster_id, 'Indicator': indicator, 'Value': prop})
                    else:
                        mean = cluster_data.mean() if n > 0 else np.nan
                        health_pivot_data.append({'Cluster': cluster_id, 'Indicator': indicator, 'Value': mean})
                        if n > 1:
                            sem = stats.sem(cluster_data)
                            ci_low, ci_upp = stats.t.interval(0.95, n - 1, loc=mean, scale=sem if sem > 0 else np.nan)
                        else:
                            ci_low, ci_upp = np.nan, np.nan
                        stat_table_rows.append({'Cluster': cluster_id, 'Mean': mean, 'N': n, '95% CI Lower': ci_low, '95% CI Upper': ci_upp})
                else: 
                    if is_binary:
                        stat_table_rows.append({'Cluster': cluster_id, 'Proportion (Yes)': np.nan, 'N': 0, '95% CI Lower': np.nan, '95% CI Upper': np.nan})
                    else:
                        stat_table_rows.append({'Cluster': cluster_id, 'Mean': np.nan, 'N': 0, '95% CI Lower': np.nan, '95% CI Upper': np.nan})
                    health_pivot_data.append({'Cluster': cluster_id, 'Indicator': indicator, 'Value': np.nan})

            if stat_table_rows:
                indicator_details["stat_table"] = pd.DataFrame(stat_table_rows).set_index('Cluster')
            analysis_results["health_analysis_details"].append(indicator_details)

        if health_pivot_data:
            analysis_results["consolidated_health_summary_pivot"] = pd.DataFrame(health_pivot_data).pivot(
                index='Cluster', columns='Indicator', values='Value'
            )

    return analysis_results


def style_health_table(df_to_style):
    """
    Highlighting the best value for each indicator in the health summary table
    """
    styled_df = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
    for col_name in df_to_style.columns:
        if col_name in HEALTH_INDICATOR_BEST_CRITERIA:
            criteria = HEALTH_INDICATOR_BEST_CRITERIA[col_name]
            col_series = pd.to_numeric(df_to_style[col_name], errors='coerce').dropna()
            if col_series.empty: continue
            best_val = col_series.min() if criteria == 'lower' else col_series.max()
            for idx, val in df_to_style[col_name].items():
                if pd.notna(val) and np.isclose(pd.to_numeric(val), best_val):
                    styled_df.loc[idx, col_name] = 'background-color: lightgreen'
    return styled_df

def display_model_visualizations(k, labels, method, params, pca_data, gower_matrix, tsne_input_data):
    """
    Displays the plots for the models
    """
    st.markdown(f"#### Plots for k = {k}")
    if labels is None:
        st.write(f"Failed to generate plots for k={k}.")
        return

    cols_plots = st.columns(3)
    with cols_plots[0]:
        plot_silhouette_per_cluster(
            data_for_plot=gower_matrix if params["preproc"] == "Gower" else pca_data,
            labels=labels, k_value=k, method_name=method,
            preproc_type=params["preproc"], gower_matrix=gower_matrix
        )
    with cols_plots[1]:
        plot_pca_clusters(pca_data, labels, k, method, params["preproc"])
    with cols_plots[2]:
        tsne_source_desc = "PCA" if pca_data is not None and pca_data.shape[1] > 1 else "Scaled Features"
        if tsne_input_data is not None:
            tsne_projected_data = get_tsne_projection(tsne_input_data)
            plot_tsne_clusters(tsne_projected_data, labels, k, method, params["preproc"], tsne_source_desc)
        else:
            st.caption("t-SNE input not available.")
    
def clustering_page(merged_df):
    st.title("Dietary and Lifestyle Pattern Clustering Report")
    st.markdown("""
    Welcome to the clustering experimental report. This page documents the process of identifying distinct dietary and lifestyle patterns among **adult participants (Age ≥ 18)** from the dataset.
    The analysis proceeds through several key stages:
    1.  **Data Preprocessing:** Filtering the data, engineering relevant features, and preparing them for different types of clustering algorithms.
    2.  **Algorithm Evaluation:** Running multiple clustering algorithms and evaluating their performance.
    3.  **Model Selection & Profiling:** Choosing the most promising model and interpreting the characteristics of the identified clusters.
    4.  **Health Outcome Analysis:** Examining the relationship between clusters and key health indicators.
    """)

    if merged_df is None or merged_df.empty:
        st.error("Data is not available. Please ensure data is loaded on the main page.")
        st.stop()

    #  Data Preprocessing
    st.header("1. Data Preprocessing and Feature Engineering")
    with st.expander("View Data Preprocessing Details", expanded=True):
        prep_results = prepare_features_for_clustering(merged_df)
        if prep_results is None:
            st.error("Feature preparation failed.")
            st.stop()

        pca_data = prep_results.get("X_pca")
        scaled_ohe_data_for_tsne = prep_results.get("X_processed_for_tsne")
        fitted_pca_model_object = prep_results.get("pca_model")
        num_retained_pca_components = prep_results.get("n_pca_components", 0)
        gower_feature_df = prep_results["df_for_gower_matrix"]
        gower_cat_indices = prep_results["cat_features_gower_indices"]
        profiling_df = prep_results["df_for_profiling"]
        selected_num_features = prep_results["numerical_features_selected"]
        selected_pca_cat_features = prep_results["categorical_features_pca_selected"]
        selected_gower_cat_features = prep_results["categorical_features_gower_selected"]
        health_indicators_raw_df = prep_results["raw_df_subset_for_health_indicators"]
        seqns = prep_results["seqns"]

        st.markdown(f"""
        **Objective:** To prepare a clean, informative dataset of adult participants for clustering.

        **Filtering Steps:**
        - **Population:** Kept participants aged 18 and over.
        - **Energy Intake:** Removed outliers in 'Energy (kcal)' using the 1.5 * IQR rule to exclude extreme and potentially erroneous entries.
        - **Resulting Sample Size:** **{len(profiling_df)}** individuals remained for analysis.
        """)

        st.subheader("Feature Selection & Engineering")
        st.markdown(f"""
        Features were selected and engineered to capture a holistic view of participants' demographics, diet, and lifestyle.
        - **Numerical Features:** `{', '.join(selected_num_features)}`
        - **Categorical Features (for Gower):** `{', '.join(selected_gower_cat_features)}`
        - **Categorical Features (for PCA):** `{', '.join(selected_pca_cat_features)}` (These are a subset of the Gower features suitable for one-hot encoding).
        """)

        st.subheader("Transformation Pipelines")
        st.markdown("""
        **For PCA-based Methods (K-Means, GMM, etc.):**
        A standard preprocessing pipeline was applied to handle the geometric assumptions of these algorithms.
        1.  **Imputation:** Missing numerical values were filled with the **median**, and categorical values with the **mode**.
        2.  **Scaling:** Numerical features were standardized (Z-score normalization) to give them equal weight.
        3.  **Encoding:** Categorical features were one-hot encoded to convert them into a numeric format.
        """)
        if fitted_pca_model_object and pca_data is not None and pca_data.shape[0] > 0 and num_retained_pca_components > 0:
            st.markdown(f"""
            **Principal Component Analysis (PCA):** To reduce dimensionality and multicollinearity, PCA was applied.
            - **Retained Components:** **{num_retained_pca_components}**
            - **Explained Variance:** **{fitted_pca_model_object.explained_variance_ratio_.sum()*100:.2f}%**
            - **Data Shape after PCA:** `{pca_data.shape}`
            """)
        else:
            st.warning("PCA was not applied or resulted in one or zero components.")

        st.markdown("""
        **For Gower-based Methods (K-Medoids, HAC with Gower):**
        These methods do not require extensive preprocessing, as the Gower distance metric is designed to handle mixed data types (numerical and categorical) directly.
        1.  **Imputation:** Missing values were filled using the same median/mode strategy.
        2.  **Transformation:** No scaling or encoding is needed. The data is used to compute a Gower distance matrix.
        """)

        st.subheader("PCA Diagnostics")
        st.markdown("""
        The plots below help validate the choice of the number of principal components.
        - **Cumulative Explained Variance:** Shows the total variance captured as components are added. The goal is to find a balance where a high amount of variance is explained by a manageable number of components.
        - **Scree Plot:** Displays the variance explained by each individual component. The "elbow" of the plot often suggests a cutoff point for the number of components to retain.
        """)
        plot_pca_diagnostics(prep_results.get("X_processed_for_tsne"))

        # Calculate Gower distance matrix once and store in session state
        st.session_state.gower_distance_matrix = None 
        if gower_feature_df is not None and not gower_feature_df.empty:
            with st.spinner("Calculating Gower distance matrix (this runs only once per session)..."):
                matrix = gower.gower_matrix(gower_feature_df, cat_features=gower_cat_indices)
                st.session_state.gower_distance_matrix = matrix 
                
            if st.session_state.gower_distance_matrix is not None:
                st.success(f"Gower distance matrix calculated and cached: {st.session_state.gower_distance_matrix.shape}")
            else:
                st.error("Gower distance matrix calculation failed.")
        gower_distance_matrix = st.session_state.gower_distance_matrix

    # Clustering Algorithm Experiments
    st.header("2. Clustering Algorithm Evaluation")
    algorithms_to_evaluate = {
        "K-Means (PCA)": {"preproc": "PCA", "method_short": "KMeans"},
        "GMM (PCA)": {"preproc": "PCA", "method_short": "GMM"},
        "K-Medoids (Gower)": {"preproc": "Gower", "method_short": "K-Medoids"},
        "HAC Ward (PCA)": {"preproc": "PCA", "method_short": "Agglo", "linkage": "ward"},
        "HAC Complete (Gower)": {"preproc": "Gower", "method_short": "Agglo", "linkage": "complete"},
        "HAC Average (PCA)": {"preproc": "PCA", "method_short": "Agglo", "linkage": "average"},
        "HAC Average (Gower)": {"preproc": "Gower", "method_short": "Agglo", "linkage": "average"}
    }

    st.subheader("Consolidated Validation Metrics")
    st.info("Calculating metrics for all algorithms across a range of k (2-10). This table helps compare models and identify promising candidates for further analysis.")

    if 'validation_metrics_df' not in st.session_state:
        all_metrics_data = []
        progress_bar = st.progress(0)
        for i, (method_key, params) in enumerate(algorithms_to_evaluate.items()):
            current_eval_data = pca_data if params["preproc"] == "PCA" else gower_distance_matrix
            if current_eval_data is None: continue
            

            k, sils, chs, dbs, sizes = evaluate_optimal_k(
                data_input_for_fitting=current_eval_data,
                algorithm_name=method_key, 
                preprocessing_type=params["preproc"],
                linkage_type=params.get("linkage")
            )
            for j, k_val in enumerate(OPTIMAL_K_RANGE):
                all_metrics_data.append({
                    'Method': method_key,
                    'Num_Clusters': k_val,
                    'Preprocessing': params["preproc"], 
                    'Silhouette': sils[j],
                    'Calinski-Harabasz': chs[j],
                    'Davies-Bouldin': dbs[j],
                    'Cluster_Sizes_Distribution': sizes[j]
                })
            progress_bar.progress((i + 1) / len(algorithms_to_evaluate))
        st.session_state.validation_metrics_df = pd.DataFrame(all_metrics_data)
        progress_bar.empty()

    st.dataframe(st.session_state.validation_metrics_df.style.format(precision=3))


    st.subheader("Optimal K Analysis Plots")
    st.markdown("""
    These plots visualize the internal validation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) for different numbers of clusters (k).
    - **Silhouette Score:** Higher is generally better.
    - **Calinski-Harabasz Index:** Higher is generally better.
    - **Davies-Bouldin Index:** Lower is generally better.
    """)

    if 'validation_metrics_df' in st.session_state and not st.session_state.validation_metrics_df.empty:
        results_df = st.session_state.validation_metrics_df

        # Gower optimal plots
        gower_methods = [m for m in results_df['Method'].unique() if "Gower" in m]
        if gower_methods:
            st.markdown("##### Gower-based Methods Optimal K Plots")
            num_gower_plots = len(gower_methods)
            fig, axs = plt.subplots(1, num_gower_plots, figsize=(5 * num_gower_plots, 4.5), squeeze=False)
            
            for i, method_key in enumerate(gower_methods):
                method_df = results_df[results_df['Method'] == method_key]
                plot_optimal_k_charts(
                    k_values=method_df['Num_Clusters'].tolist(),
                    sils=method_df['Silhouette'].tolist(),
                    chs=method_df['Calinski-Harabasz'].tolist(),
                    dbs=method_df['Davies-Bouldin'].tolist(),
                    method_display_name=method_key,
                    ax=axs[0, i] 
                )
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")
        # PCA optimal plots
        pca_methods = [m for m in results_df['Method'].unique() if "PCA" in m]
        if pca_methods:
            st.markdown("##### PCA-based Methods Optimal K Plots")
            for method_key in pca_methods:
                method_df = results_df[results_df['Method'] == method_key]
                plot_optimal_k_charts(
                    k_values=method_df['Num_Clusters'].tolist(),
                    sils=method_df['Silhouette'].tolist(),
                    chs=method_df['Calinski-Harabasz'].tolist(),
                    dbs=method_df['Davies-Bouldin'].tolist(),
                    method_display_name=method_key
                )
                st.markdown("---")
        else:
            st.warning("Cannot generate optimal k plots: Validation metrics data is not available.")

    #  Interpretation of Initial Results 
    st.header("3. Interpretation and Detailed Visualization")
    st.markdown("""
    This section allows for a deeper dive into the most promising models identified from the metrics table above.

    **On Interpreting Validation Metrics:**
    - **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters. Scores range from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. **However, with complex, overlapping real-world data (like human behavior), scores are often low (e.g., 0.1-0.3).** A model with a slightly lower score but more balanced and interpretable clusters may be preferable to one with a higher score that creates tiny, non-insightful "outlier" groups.
    - **Calinski-Harabasz / Davies-Bouldin:** These are other metrics measuring cluster separation. Like Silhouette, they are a guide, not an absolute truth.

    **Rationale for Model Selection:**
    - We look for a **"sweet spot"**: a model that has relatively good metric scores, produces reasonably balanced cluster sizes (avoiding a single giant cluster and many tiny ones), and ultimately yields clusters that are distinct and interpretable when we profile their features.
    - **K-Medoids (Gower)** is often a strong candidate for this type of medical/behavioral data because it naturally handles mixed data types without the assumptions of PCA and is less sensitive to outliers than K-Means.
    - **Excluded Methods (e.g., HAC with Average Linkage):** Hierarchical clustering with average linkage was excluded from the final selection. While sometimes producing high metric scores, this method is prone to a "chaining" effect, where it creates one very large, elongated cluster while leaving many individuals as tiny, separate clusters. This is not useful for creating meaningful, group-level personas, so it was removed in favor of methods that produce more balanced groupings.
    """)

    promising_models_to_plot = {
        "K-Means (PCA)": {"preproc": "PCA", "method_short": "KMeans", "k_values": [2, 3, 4, 5]},
        "GMM (PCA)": {"preproc": "PCA", "method_short": "GMM", "k_values": [2, 3, 4, 5]},
        "HAC Ward (PCA)": {"preproc": "PCA", "method_short": "Agglo", "linkage": "ward", "k_values": [2, 3, 4, 5]},
        "K-Medoids (Gower)": {"preproc": "Gower", "method_short": "K-Medoids", "k_values": [2, 3, 4, 5]},
    }

    for method_key, params in promising_models_to_plot.items():
        with st.expander(f"View Detailed Plots for {method_key}"):
            for k_target in params["k_values"]:
                st.markdown(f"#### Plots for k = {k_target}")
                data_to_fit = pca_data if params["preproc"] == "PCA" else gower_feature_df
                gower_matrix_for_fit = gower_distance_matrix if params["preproc"] == "Gower" else None

                _, fitted_labels = fit_clustering_model(
                    data_fit=data_to_fit, full_method_name=method_key, k=k_target,
                    preproc_type=params["preproc"], linkage_type=params.get("linkage"),
                    gower_dist_matrix=gower_matrix_for_fit
                )

                if fitted_labels is not None:
                    cols_plots = st.columns(3)
                    with cols_plots[0]:
                        if params["preproc"] == "Gower":
                            plot_silhouette_per_cluster(gower_matrix_for_fit, fitted_labels, k_target, method_key, params["preproc"], gower_matrix=gower_matrix_for_fit)
                        else: # PCA
                            plot_silhouette_per_cluster(pca_data, fitted_labels, k_target, method_key, params["preproc"], gower_matrix=None)
                    with cols_plots[1]:
                        plot_pca_clusters(pca_data, fitted_labels, k_target, method_key, params["preproc"])
                    with cols_plots[2]:
                        tsne_input_source_data = pca_data if pca_data is not None and pca_data.shape[1] > 1 else scaled_ohe_data_for_tsne
                        tsne_source_description = "PCA" if pca_data is not None and pca_data.shape[1] > 1 else "Scaled Features"
                        if tsne_input_source_data is not None:
                            tsne_projected_data = get_tsne_projection(tsne_input_source_data)
                            plot_tsne_clusters(tsne_projected_data, fitted_labels, k_target, method_key, params["preproc"], tsne_source_description)
                        else: st.caption("t-SNE input not available.")
                else:
                    st.write(f"Failed to generate plots for k={k_target}.")
                st.markdown("---")

    # Final Model Selection and Profiling
    st.header("4. Final Model Selection and Cluster Profiling")
    results_df = st.session_state.get('validation_metrics_df')
    if results_df is None or results_df.empty:
        st.warning("Clustering results are not available for selection.")
        st.stop()

    # Improved model selection UI
    available_methods = sorted([m for m in results_df['Method'].unique() if m in promising_models_to_plot])
    selected_method = st.selectbox(
        "**Step 1: Select a clustering method for final analysis**",
        options=available_methods,
        index=available_methods.index("K-Medoids (Gower)") if "K-Medoids (Gower)" in available_methods else 0
    )

    possible_ks = promising_models_to_plot.get(selected_method, {}).get("k_values", [2, 3, 4, 5])
    selected_k = st.radio(
        "**Step 2: Select the number of clusters (k)**",
        options=possible_ks,
        horizontal=True
    )

    final_model_details = results_df[(results_df['Method'] == selected_method) & (results_df['Num_Clusters'] == selected_k)]
    if final_model_details.empty:
        st.stop()

    final_model_details = final_model_details.iloc[0]
    final_method_str = final_model_details['Method']
    final_k = int(final_model_details['Num_Clusters'])
    final_preproc_type = final_model_details['Preprocessing']
    final_algo_params = algorithms_to_evaluate.get(final_method_str)

    st.success(f"**Selected for Final Analysis:** `{final_method_str}` with `k={final_k}`")

    # Refit the final model
    final_data_for_fitting = pca_data if final_preproc_type == "PCA" else gower_feature_df
    final_gower_matrix_for_fitting = gower_distance_matrix if final_preproc_type == "Gower" else None

    with st.spinner(f"Generating final labels for {final_method_str} (k={final_k})..."):
        _, final_labels = fit_clustering_model(
            data_fit=final_data_for_fitting, full_method_name=final_method_str, k=final_k,
            preproc_type=final_preproc_type, linkage_type=final_algo_params.get("linkage"),
            gower_dist_matrix=final_gower_matrix_for_fitting
        )

    if final_labels is not None:
        st.subheader("Cluster Stability Analysis (via Bootstrap Resampling)")
        st.markdown("""
        This analysis assesses how robust the clusters are. It works by repeatedly resampling the dataset, re-running the clustering, and comparing the new clusters to the original ones using the **Adjusted Rand Index (ARI)**.
        - **High Mean ARI (e.g., > 0.7):** Good stability. The clusters are generally reproducible.
        - **Moderate Mean ARI (e.g., 0.5-0.7):** Acceptable stability. The core structure of the clusters shows some consistency, but minor variations might occur.
        - **Low Mean ARI (e.g., < 0.5):** Poor stability. The clusters may be more sensitive to sampling variations and should be interpreted with caution.
        """)

        if st.button("Run Bootstrap Stability Analysis", key=f"run_stability_{final_method_str}_{final_k}"):
            progress_bar_stability = st.progress(0, text="Starting bootstrap analysis...")
            mean_ari, std_ari, num_runs = calculate_bootstrap_stability(
                data_for_fitting=final_data_for_fitting, original_labels=final_labels, model_name=final_method_str,
                k_clusters=final_k, preproc_type=final_preproc_type, hac_linkage=final_algo_params.get("linkage"),
                gower_dist_matrix=final_gower_matrix_for_fitting, n_bootstraps=300, progress_bar=progress_bar_stability
            )

            if pd.notna(mean_ari):
                st.success(f"**Stability Analysis Complete** ({num_runs} successful runs)")
                col1, col2 = st.columns(2)
                col1.metric(label="Mean Adjusted Rand Index (ARI)", value=f"{mean_ari:.3f}")
                col2.metric(label="Standard Deviation of ARI", value=f"{std_ari:.3f}")
            else:
                st.error("Could not complete the stability analysis.")

        # Profiling and Health Analysis 
        labels_df_for_merge = pd.DataFrame({'Cluster': final_labels, 'SEQN': seqns})
        health_data_merged_for_cache = pd.merge(health_indicators_raw_df, labels_df_for_merge, on="SEQN", how="inner")
        health_data_merged_for_cache = health_data_merged_for_cache[health_data_merged_for_cache['Cluster'] >= 0]

        model_cache_id = f"{final_method_str}_{final_k}_{final_preproc_type}"
        analysis_results_cached = generate_profile_and_health_analysis(
            final_labels, profiling_df, health_data_merged_for_cache,
            PREDEFINED_HEALTH_INDICATORS, model_cache_id, final_k
        )

        st.subheader("Cluster Profiles (Mean/Mode of Input Features)")
        st.markdown("This table shows the central tendency for each input feature within each cluster. It is the primary tool for understanding and naming the clusters.")
        cluster_summary_display = analysis_results_cached.get("input_feature_profile_summary")
        if cluster_summary_display is not None and not cluster_summary_display.empty:
            num_cols = cluster_summary_display.select_dtypes(include=np.number).columns
            st.dataframe(cluster_summary_display.style.format({col: "{:.2f}" for col in num_cols}, na_rep="-"))
            st.subheader("Cluster Sizes")
            st.dataframe(analysis_results_cached.get("cluster_sizes"))
        else:
            st.warning("Input feature profiles could not be generated.")

        st.subheader("Health Indicator Analysis by Cluster")
        st.markdown("Here, we explore whether cluster membership is associated with different health outcomes. For each indicator, we present a plot, a statistical test (Kruskal-Wallis for continuous, Chi-squared for binary), and a summary table.")

        health_analysis_details = analysis_results_cached.get("health_analysis_details", [])
        for indicator_info in health_analysis_details:
            indicator_name = indicator_info.get("name")
            if not indicator_name: continue

            st.markdown(f"#### {indicator_name}")
            col1, col2 = st.columns([2, 3])
            with col1:
                indicator_data_plot = health_data_merged_for_cache[['Cluster', indicator_name]].copy()
                is_binary_plot = indicator_name in ['Ever have Diabetes', 'Ever have Hypertension', 'Ever have Heart Attack']
                if is_binary_plot:
                    indicator_data_plot[indicator_name] = pd.to_numeric(indicator_data_plot[indicator_name], errors='coerce')
                    prop_counts = indicator_data_plot.groupby('Cluster')[indicator_name].value_counts(normalize=True).mul(100).unstack(fill_value=0)
                    if 1.0 in prop_counts.columns:
                        fig, ax = plt.subplots(figsize=(6,4))
                        prop_counts[1.0].plot(kind='bar', ax=ax, color='skyblue')
                        ax.set_ylabel(f"% with Condition")
                        ax.set_title(f"Prevalence by Cluster")
                        plt.setp(ax.get_xticklabels(), rotation=0)
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    fig, ax = plt.subplots(figsize=(6,4))
                    sns.boxplot(x='Cluster', y=indicator_name, data=indicator_data_plot, ax=ax, palette="viridis")
                    ax.set_title(f"Distribution by Cluster")
                    st.pyplot(fig)
                    plt.close(fig)
            with col2:
                st.markdown(f"**Statistical Test:** {indicator_info.get('p_value_text', 'N/A')}")
                stat_table_df = indicator_info.get("stat_table")
                if stat_table_df is not None and not stat_table_df.empty:
                    num_cols_stat = stat_table_df.select_dtypes(include=np.number).columns
                    st.table(stat_table_df.style.format({col: "{:.2f}" for col in num_cols_stat}, na_rep="-"))

            st.markdown("---")

        consolidated_health_summary = analysis_results_cached.get("consolidated_health_summary_pivot")
        if consolidated_health_summary is not None and not consolidated_health_summary.empty:
            st.subheader("Consolidated Health Indicator Summary")
            st.markdown("This table provides a high-level overview of all health indicators across the clusters. Cells highlighted in **green** indicate the 'best' value for that indicator (e.g., lowest BMI, highest HDL).")
            def style_health_table(df_to_style):
                styled_df = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
                for col_name in df_to_style.columns:
                    if col_name in HEALTH_INDICATOR_BEST_CRITERIA:
                        criteria = HEALTH_INDICATOR_BEST_CRITERIA[col_name]
                        col_series = pd.to_numeric(df_to_style[col_name], errors='coerce').dropna()
                        if col_series.empty: continue
                        best_val = col_series.min() if criteria == 'lower' else col_series.max()
                        for idx, val in df_to_style[col_name].items():
                            if pd.notna(val) and np.isclose(pd.to_numeric(val), best_val):
                                styled_df.loc[idx, col_name] = 'background-color: darkgreen'
                return styled_df

            num_cols_pivot = consolidated_health_summary.select_dtypes(include=np.number).columns
            st.dataframe(consolidated_health_summary.style.apply(style_health_table, axis=None).format({col: "{:.2f}" for col in num_cols_pivot}, na_rep="-"))

        st.subheader("Cluster Profiles (Mean/Mode of Input Features)")
        st.markdown("This table shows the central tendency for each input feature within each cluster. It is the primary tool for understanding and naming the clusters.")
        cluster_summary_display = analysis_results_cached.get("input_feature_profile_summary")
        if cluster_summary_display is not None and not cluster_summary_display.empty:
            num_cols = cluster_summary_display.select_dtypes(include=np.number).columns
            st.dataframe(cluster_summary_display.style.format({col: "{:.2f}" for col in num_cols}, na_rep="-"))
            st.subheader("Cluster Sizes")
            st.dataframe(analysis_results_cached.get("cluster_sizes"))
        else:
            st.warning("Input feature profiles could not be generated.")

    else:
        st.error("Failed to generate final labels for the selected model. Profiling cannot proceed.")

    #  Final Interpretation
    st.header("5. Final Interpretation")

    if selected_method == "K-Means (PCA)" and selected_k == 3:
        st.markdown(f"Based on the analysis of the **K-Means (PCA) (k=3)** model, we can now synthesize the findings to create descriptive cluster personas.")


        st.subheader("Interpretation of K-Means (PCA) Clusters (k=3)")
        st.markdown("""
        Based on the input features and health outcomes, the three clusters identified by K-Means (PCA) with k=3 can be interpreted as follows:

        **Cluster 0: The Male High Consumers with Moderate Risk**
        - **Input Features:** This cluster is entirely male, with an average age around 45. They have the highest energy intake, sugar consumption, and sodium intake. Their diet includes a fair amount of convenience foods, and they engage in a moderate level of physical activity.
        - **Health Outcomes:** They present average to slightly elevated health risks—higher blood pressure and BMI than Cluster 1, but lower than Cluster 2. Their medication usage is also in the mid-range.

        **Cluster 1: The Active Young Females**
        - **Input Features:** Exclusively female and the youngest group (around 32). Despite consuming the most convenience foods, likely influenced by lower income levels, they maintain the highest physical activity levels and consume fewer overall calories and sugar compared to Cluster 0.
        - **Health Outcomes:** This cluster displays the best overall health indicators: lowest risks of diabetes, hypertension, and heart attack, as well as the lowest BMI, Glycohemoglobin (HbA1c), blood pressure, and triglyceride levels. They take the fewest medications.
        - Characterized by younger, active women whose high activity levels and moderate intake contribute to better health outcomes, even in the face of higher convenience food consumption.

        **Cluster 2: The Older Sedentary Females with Elevated Health Risks**
        - **Input Features:** This group is entirely female and the oldest (around 58). They have the lowest energy, sugar, and sodium intake, consume the fewest convenience foods, and are very sedentary. They also belong to the highest income bracket.
        - **Health Outcomes:** This cluster faces the most significant health challenges, such as highest BMI, HbA1c, blood pressure, and disease prevalence. They rely on the greatest number of medications.
        - Representing older, less active women with higher socioeconomic status but substantially worse health outcomes, likely due to low activity levels and age-related factors.
        """)

    else:
        st.markdown("""
        **Final Interpretation is available for K-Means (PCA) with k=3.**
        Please select "K-Means (PCA)" as the clustering method and "3" for the number of clusters above to view detailed cluster personas.
        For other methods or numbers of clusters, you can use the tables above (Input Features, Health Outcomes) to derive your own interpretations.
        """)